"""Browse the remote G0 manifest in a local WebGL page, without training.

Example:
  python3 -m experiments.vae_sdf.view_meshes

The server binds only to 127.0.0.1. Meshes are loaded on demand through SSH
using the fixed manifest and audit in /data/hi3d/g0-prep.
"""

from __future__ import annotations

import argparse
import json
import shlex
import struct
import subprocess
import webbrowser
import zlib
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


HTML = Path(__file__).with_name("mesh_browser.html")
COMPARE_HTML = Path(__file__).with_name("g0b_compare.html")


class RemoteSource:
    def __init__(self, host: str, remote_dir: str, remote_python: str, timeout: int,
                 dataset: str, batch_root: str | None = None):
        if not host or host.startswith("-"):
            raise ValueError("SSH host must be a host name or user@host")
        self.host = host
        self.root = remote_dir.rstrip("/")
        self.python = remote_python
        self.timeout = timeout
        self.batch_root = batch_root or f"{self.root}/g0-a1a2-r128-cpu-v1"
        if dataset not in {"balanced50", "usable50", "first50"}:
            raise ValueError("Unknown G0 dataset")
        self.manifest_name = f"manifest-{dataset}.json"
        self.audit_name = f"mesh-load-{dataset}.json"

    def run(self, command: list[str]) -> bytes:
        result = subprocess.run(
            ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             self.host, shlex.join(command)],
            capture_output=True, timeout=self.timeout, check=False,
        )
        if result.returncode:
            detail = result.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"Remote command failed ({result.returncode}): {detail[:500]}")
        return result.stdout

    def json_file(self, filename: str) -> dict:
        return json.loads(self.run(["cat", f"{self.root}/{filename}"]))

    def batch_summary(self) -> dict:
        return json.loads(self.run(["cat", f"{self.batch_root}/summary.json"]))

    def g0b_summary(self, g0b_root: str) -> dict:
        return json.loads(self.run(["cat", f"{g0b_root}/summary.json"]))

    def mesh(self, index: int) -> bytes:
        library_paths = ":".join((f"{self.root}/vendor-full", f"{self.root}/vae3d", self.root))
        return self.run([
            "env", f"PYTHONPATH={library_paths}", self.python,
            "-m", "experiments.vae_sdf.remote_mesh",
            "--manifest", f"{self.root}/{self.manifest_name}",
            "--audit", f"{self.root}/{self.audit_name}",
            "--index", str(index),
        ])

    def target(self, index: int, stage: str, kind: str, axis: str = "z",
               slice_index: int = 64) -> bytes:
        if (stage not in {"A1", "A2"} or kind not in {"mesh", "slice"}
                or axis not in {"x", "y", "z"}):
            raise ValueError("Invalid target request")
        library_paths = ":".join((f"{self.root}/vendor-full", f"{self.root}/vae3d", self.root))
        return self.run([
            "env", f"PYTHONPATH={library_paths}",
            f"TMPDIR={self.root}/tmp", f"XDG_CACHE_HOME={self.root}/cache",
            self.python, "-m", "experiments.vae_sdf.remote_target",
            "--batch-root", self.batch_root, "--index", str(index),
            "--stage", stage, "--kind", kind, "--axis", axis,
            "--slice-index", str(slice_index),
        ])

    def g0b_mesh(self, index: int, extractor: str, g0b_root: str) -> bytes:
        if extractor not in {"mc", "flexicubes"}:
            raise ValueError("Invalid G0b extractor")
        library_paths = ":".join((f"{self.root}/vendor-full", f"{self.root}/vae3d", self.root))
        return self.run([
            "env", f"PYTHONPATH={library_paths}", self.python,
            "-m", "experiments.vae_sdf.remote_g0b_mesh",
            "--root", g0b_root, "--manifest", f"{self.root}/{self.manifest_name}",
            "--index", str(index), "--extractor", extractor,
        ])

    def g0b_bundle(self, index: int, g0b_root: str, preview_faces: int) -> bytes:
        library_paths = ":".join((f"{self.root}/vendor-full", f"{self.root}/vae3d", self.root))
        compressed = self.run([
            "env", f"PYTHONPATH={library_paths}", self.python,
            "-m", "experiments.vae_sdf.remote_g0b_bundle",
            "--root", g0b_root, "--manifest", f"{self.root}/{self.manifest_name}",
            "--audit", f"{self.root}/{self.audit_name}", "--index", str(index),
            "--preview-faces", str(preview_faces),
        ])
        payload = zlib.decompress(compressed)
        if payload[:4] != b"G0B2" or len(payload) < 40:
            raise ValueError("Invalid G0b mesh bundle")
        lengths = struct.unpack_from("<III", payload, 4)
        if any(length < 8 for length in lengths) or len(payload) != 40 + sum(lengths):
            raise ValueError("Incomplete G0b mesh bundle")
        return payload


def catalog(manifest: dict, audit: dict, batch: dict | None = None,
            g0b: dict | None = None) -> dict:
    entries = manifest.get("assets", [])
    results = audit.get("assets", [])
    if not entries or len(entries) != len(results):
        raise ValueError("Manifest and mesh audit have different asset counts")
    rows = []
    for index, (asset, result) in enumerate(zip(entries, results)):
        if (result.get("index") != index or result.get("id") != asset.get("id")
                or result.get("path") != asset.get("path")):
            raise ValueError(f"Manifest and mesh audit differ at index {index}")
        rows.append({
            "index": index, "id": asset["id"], "name": Path(asset["path"]).name,
            "category": asset.get("category", "unclassified"),
            "status": result["status"], "error": result.get("error"),
            "faces": result.get("faces"), "vertices": result.get("vertices"),
            "watertight": result.get("watertight_geometry"),
            "seconds": result.get("seconds"),
        })
    target_resolution = None
    if batch is not None:
        paired = batch.get("assets", [])
        if len(paired) != len(rows) or batch.get("denominator") != len(rows):
            raise ValueError("A1/A2 batch differs from the selected manifest")
        for index, (row, result, asset) in enumerate(zip(rows, paired, entries)):
            if (result.get("index") != index or result.get("id") != row["id"]
                    or result.get("source") != asset["path"]):
                raise ValueError(f"A1/A2 batch differs at index {index}")
            if result.get("status") == "success":
                metrics = result["metrics"]
                row["target"] = {
                    "status": "success", "stages": {
                        stage: {"chamfer_l1": metrics["stages"][stage]["chamfer_l1"],
                                "fscore_0_005": metrics["stages"][stage]["fscore"]["0.005"]["fscore"],
                                "fscore_0_01": metrics["stages"][stage]["fscore"]["0.01"]["fscore"],
                                "faces": metrics["stages"][stage]["faces"]}
                        for stage in ("A1", "A2")},
                    "changed_samples": metrics["cache_check"]["field_changed_samples"],
                    "added_negative_samples": metrics["cache_check"]["added_negative_samples"],
                }
                target_resolution = metrics["cache_check"]["resolution"]
            else:
                row["target"] = {"status": result.get("status", "pending")}
    if g0b is not None:
        compared = g0b.get("assets", [])
        if len(compared) != len(rows) or g0b.get("denominator") != len(rows):
            raise ValueError("G0b batch differs from selected manifest")
        for index, (row, result, asset) in enumerate(zip(rows, compared, entries)):
            if (result.get("index") != index or result.get("id") != row["id"]
                    or result.get("source") != asset["path"]
                    or result.get("source_sha256") != asset["sha256"]):
                raise ValueError(f"G0b batch differs at index {index}")
            row["g0b"] = {"status": result.get("status"),
                          "a0_status": result.get("a0_status"),
                          "a0_concerns": result.get("a0_concerns", [])}
            if result.get("status") == "success":
                metrics = result["metrics"]
                row["g0b"].update({
                    "inverted_crossing_cells": result["audit"]["a4_inverted_crossing_cells"],
                    "mc": metrics["MC_vs_source"],
                    "flexicubes": metrics["FlexiCubes_vs_source"],
                    "flexicubes_vs_mc": metrics["FlexiCubes_vs_MC"],
                })
    return {"rows": rows, "selection": manifest.get("selection", {}),
            "audit_resolution": audit.get("resolution"), "target_resolution": target_resolution,
            "has_targets": batch is not None, "has_g0b": g0b is not None}


def handler_for(source: RemoteSource, data: dict, g0b_root: str | None = None,
                preview_cache: Path | None = None):
    @lru_cache(maxsize=2)
    def mesh(index: int) -> bytes:
        return source.mesh(index)

    @lru_cache(maxsize=4)
    def target_mesh(index: int, stage: str) -> bytes:
        return source.target(index, stage, "mesh")

    @lru_cache(maxsize=6)
    def g0b_mesh(index: int, extractor: str) -> bytes:
        return source.g0b_mesh(index, extractor, g0b_root)

    @lru_cache(maxsize=4)
    def g0b_bundle(index: int) -> bytes:
        if preview_cache is not None:
            compressed = (preview_cache / f"bundle_{index:02d}.zlib").read_bytes()
            payload = zlib.decompress(compressed)
            if payload[:4] != b"G0B2" or len(payload) < 40:
                raise ValueError("Invalid local G0b preview bundle")
            lengths = struct.unpack_from("<III", payload, 4)
            if any(length < 8 for length in lengths) or len(payload) != 40 + sum(lengths):
                raise ValueError("Incomplete local G0b preview bundle")
            return payload
        return source.g0b_bundle(index, g0b_root, 6000)

    @lru_cache(maxsize=1)
    def g0b_full_bundle(index: int) -> bytes:
        return source.g0b_bundle(index, g0b_root, 0)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            path = request.path
            try:
                if path == "/":
                    body, mime = HTML.read_bytes(), "text/html; charset=utf-8"
                elif path == "/compare" and data["has_g0b"]:
                    body, mime = COMPARE_HTML.read_bytes(), "text/html; charset=utf-8"
                elif path == "/api/catalog":
                    body, mime = json.dumps(data, ensure_ascii=False).encode(), "application/json"
                elif path.startswith("/bundle/") and data["has_g0b"]:
                    value = path[8:]
                    if not value.isdigit():
                        self.send_error(404)
                        return
                    index = int(value)
                    if (index >= len(data["rows"])
                            or data["rows"][index].get("g0b", {}).get("status") != "success"):
                        self.send_error(404, "G0b bundle not available")
                        return
                    body, mime = g0b_bundle(index), "application/octet-stream"
                elif path.startswith("/bundle-full/") and data["has_g0b"]:
                    value = path[13:]
                    if not value.isdigit():
                        self.send_error(404)
                        return
                    index = int(value)
                    if (index >= len(data["rows"])
                            or data["rows"][index].get("g0b", {}).get("status") != "success"):
                        self.send_error(404, "G0b full bundle not available")
                        return
                    body, mime = g0b_full_bundle(index), "application/octet-stream"
                elif path.startswith("/g0b/") and data["has_g0b"]:
                    parts = path.strip("/").split("/")
                    if (len(parts) != 3 or not parts[1].isdigit()
                            or parts[2] not in {"mc", "flexicubes"}):
                        self.send_error(404)
                        return
                    index, extractor = int(parts[1]), parts[2]
                    if (index >= len(data["rows"])
                            or data["rows"][index].get("g0b", {}).get("status") != "success"):
                        self.send_error(404, "G0b mesh not available")
                        return
                    body, mime = g0b_mesh(index, extractor), "application/octet-stream"
                elif path.startswith("/mesh/") and path[6:].isdigit():
                    index = int(path[6:])
                    if index >= len(data["rows"]) or data["rows"][index]["status"] != "loaded":
                        self.send_error(404, "Mesh not available")
                        return
                    body, mime = mesh(index), "application/octet-stream"
                elif path.startswith("/target/") and data["has_targets"]:
                    parts = path.strip("/").split("/")
                    if (len(parts) != 4 or not parts[1].isdigit()
                            or parts[2] not in {"A1", "A2"}
                            or parts[3] not in {"mesh", "slice"}):
                        self.send_error(404)
                        return
                    index, stage, kind = int(parts[1]), parts[2], parts[3]
                    if (index >= len(data["rows"]) or
                            data["rows"][index].get("target", {}).get("status") != "success"):
                        self.send_error(404, "Target not available")
                        return
                    if kind == "mesh":
                        body = target_mesh(index, stage)
                    else:
                        query = parse_qs(request.query)
                        axis = query.get("axis", ["z"])[0]
                        value = query.get("index", ["64"])[0]
                        if (axis not in {"x", "y", "z"} or not value.isdigit()
                                or int(value) >= data["target_resolution"]):
                            self.send_error(400, "Invalid slice")
                            return
                        body = source.target(index, stage, "slice", axis, int(value))
                    mime = "application/octet-stream"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                self.send_error(502, str(exc)[:500])

        def log_message(self, format, *args):
            if len(args) >= 2 and str(args[1]) != "200":
                super().log_message(format, *args)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="le.chang@124.156.97.173")
    parser.add_argument("--remote-dir", default="/data/hi3d/g0-prep")
    parser.add_argument("--remote-python", default="/data/hi3d/hi3d-validation/.venv/bin/python")
    parser.add_argument("--port", type=int, default=0, help="Local port; 0 picks a free port")
    parser.add_argument("--timeout", type=int, default=180, help="SSH mesh load timeout in seconds")
    parser.add_argument("--dataset", choices=("balanced50", "usable50", "first50"),
                        default="balanced50",
                        help="G0 manifest to browse (default: balanced50)")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--batch-root", help="Remote A1/A2 batch root for balanced50")
    parser.add_argument("--g0b-root", help="Remote G0b batch root for balanced50")
    parser.add_argument("--preview-cache", type=Path,
                        help="Local snapshot with catalog.json and compressed G0b preview bundles")
    args = parser.parse_args()
    if args.timeout < 1 or not 0 <= args.port <= 65535:
        parser.error("timeout must be positive and port must be in 0..65535")
    source = RemoteSource(args.host, args.remote_dir, args.remote_python, args.timeout,
                          args.dataset, args.batch_root)
    g0b_root = args.g0b_root or f"{source.root}/g0b-a4-r128-v1"
    if args.preview_cache:
        if args.dataset != "balanced50":
            parser.error("Local G0b preview cache only supports balanced50")
        data = json.loads((args.preview_cache / "catalog.json").read_text())
        if (not data.get("has_g0b") or len(data.get("rows", [])) != 50
                or any(not (args.preview_cache / f"bundle_{index:02d}.zlib").is_file()
                       for index in range(50))):
            parser.error("Incomplete local G0b preview cache")
    else:
        batch = source.batch_summary() if args.dataset == "balanced50" else None
        g0b = source.g0b_summary(g0b_root) if args.dataset == "balanced50" else None
        data = catalog(source.json_file(source.manifest_name), source.json_file(source.audit_name), batch, g0b)
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 handler_for(source, data, g0b_root, args.preview_cache))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"G0 mesh viewer: {url} — {len(data['rows'])} assets. Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
