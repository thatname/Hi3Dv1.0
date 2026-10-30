"""Read fixed four-route caches for the local G0 comparison page.

This module never accepts an arbitrary cache path from the browser. A1/A2 are
shared between normal modes; A3/A4 must match their own resolved route config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

ROUTES = {"mc_flat", "mc_smooth", "flexicubes_flat", "flexicubes_smooth"}
STAGES = {"source", "A1", "A2", "A3", "A4"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _cache(root: Path, asset: dict, index: int, route: str, stage: str):
    if stage in {"A1", "A2"}:
        return root / "shared" / asset["id"] / f"R128-{stage.lower()}-{route.split('_')[0]}" / "raw_mesh.ply"
    parent = root / asset["id"] / "R128"
    for candidate in sorted(parent.glob(f"{route}-*")):
        config_file = candidate / "resolved_config.json"
        if not config_file.is_file():
            continue
        config = json.loads(config_file.read_text())
        if (config.get("index") == index and config.get("asset_id") == asset["id"]
                and config.get("source_sha256") == asset["sha256"]
                and config.get("route_id") == route and config.get("resolution") == 128
                and config.get("sampling_offset") == .5):
            path = candidate / stage.lower() / "raw_mesh.ply"
            return path
    return None


def _entry(root, asset, index, route, stage):
    path = _cache(root, asset, index, route, stage)
    if path is None:
        return {"status": "missing"}
    if path.is_file():
        if stage in {"A1", "A2"}:
            identity = json.loads((path.parent / "identity.json").read_text())
            if (identity.get("asset_id") != asset["id"]
                    or identity.get("source_sha256") != asset["sha256"]
                    or identity.get("stage") != stage
                    or identity.get("extractor") != route.split("_")[0]
                    or identity.get("sampling_offset") != .5):
                raise ValueError("Shared A1/A2 extraction identity mismatch")
        answer = {"status": "ready", "sha256": _sha(path), "bytes": path.stat().st_size}
        if stage in {"A3", "A4"}:
            parent = path.parent
            answer["audit"] = json.loads((parent / "audit.json").read_text())
            answer["metrics"] = json.loads((parent / "metrics.json").read_text())
        elif stage in {"A1", "A2"}:
            answer["metrics"] = json.loads((path.parent / "metrics.json").read_text())
        return answer
    if path.parent.with_name(path.parent.name + ".partial").exists():
        return {"status": "running"}
    if stage in {"A1", "A2"}:
        failures = root / "shared-failures.jsonl"
        if failures.is_file():
            for line in reversed(failures.read_text().splitlines()):
                failure = json.loads(line)
                if (failure["index"] == index and failure["stage"] == stage
                        and failure["extractor"] == route.split("_")[0]):
                    return {"status": "failed", "error": failure["error"]}
    if stage in {"A3", "A4"}:
        status_file = root / "batch-status.json"
        if status_file.is_file():
            run = json.loads(status_file.read_text()).get("runs", {}).get(f"{index}:{route}", {})
            if run.get("status") in {"running", "failed"}:
                return {"status": run["status"], "log": run.get("log")}
    return {"status": "missing"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--kind", choices=("catalog", "mesh"), required=True)
    parser.add_argument("--route", choices=sorted(ROUTES), default="mc_flat")
    parser.add_argument("--stage", choices=sorted(STAGES), default="source")
    parser.add_argument("--preview-faces", type=int, default=6000)
    args = parser.parse_args()
    assets = json.loads(args.manifest.read_text())["assets"]
    if len(assets) != 50 or not 0 <= args.index < 50 or args.preview_faces < 0:
        parser.error("Expected fixed balanced50 index and nonnegative preview face budget")
    asset = assets[args.index]
    if args.kind == "catalog":
        cells = {}
        for route in sorted(ROUTES):
            cells[route] = {stage: _entry(args.root, asset, args.index, route, stage)
                            for stage in ("A1", "A2", "A3", "A4")}
        result = {"index": args.index, "asset_id": asset["id"],
                  "source_sha256": asset["sha256"], "resolution": 128,
                  "sampling_offset": .5, "cells": cells}
        sys.stdout.write(json.dumps(result))
        return 0

    import numpy as np
    import trimesh
    from .remote_g0b_bundle import encode_mesh, preview_mesh
    if args.stage == "source":
        from vae3d.data import load_mesh
        mesh, center, scale = load_mesh(asset["path"], normalization_extent=126 / 128)
        mesh.vertices = np.asarray(mesh.vertices) * scale + center
    else:
        path = _cache(args.root, asset, args.index, args.route, args.stage)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"{args.index} {args.route} {args.stage} is unavailable")
        mesh = trimesh.load(path, process=False)
    mesh = preview_mesh(mesh, args.preview_faces)
    payload = encode_mesh(mesh)
    sys.stdout.buffer.write(zlib.compress(payload, level=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
