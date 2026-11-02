"""One A1/A2 worker that claims assets individually and keeps going.

The batch entry point runs a fixed list per process, so a slow asset makes its
whole slice late and an idle worker cannot help a busy one. Here the asset is the
work unit: a worker claims the next unclaimed asset, builds it in-process (no
per-asset interpreter start), records the result, and immediately claims again.
Claim files make the hand-off safe between workers; a claim left behind by a dead
worker expires so the asset is retried.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import open3d as o3d
import torch

import vae3d.back_shell as back_shell

from .evaluate_initial_targets import evaluate_pair
from .prepare_initial_targets import run as prepare_one
from .run_initial_batch import summarize


def _cpu_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene(
        nthreads=int(os.environ.get("VAE3D_RAY_THREADS", "1")))
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                  o3d.utility.Vector3iVector(mesh.faces))))
    return scene


back_shell.cpu_scene = _cpu_scene


def claim(claims: Path, index: int, stale_seconds: float) -> bool:
    """Create the claim file atomically; steal it when the owner looks dead."""

    marker = claims / f"{index:05d}.claim"
    try:
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        try:
            if time.time() - marker.stat().st_mtime > stale_seconds:
                marker.write_text(str(os.getpid()))
                return True
        except OSError:
            pass
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--load-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True,
                        help="ordered list of asset indices still to build")
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-sites", type=int, default=6_000_000)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many assets (0 = until the queue ends)")
    args = parser.parse_args()

    args.claims.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("VAE3D_TORCH_THREADS", "1")))
    manifest = json.loads(args.manifest.read_bytes())
    assets = manifest["assets"]
    plan = json.loads((args.output_root / "plan.json").read_text(encoding="utf-8"))
    results = args.output_root / "assets"
    results.mkdir(parents=True, exist_ok=True)
    queue = [int(line) for line in args.queue.read_text().split()]

    done = 0
    pause = args.claims.parent / "pause"
    for index in queue:
        if args.limit and done >= args.limit:
            break
        if pause.exists():
            break
        result_path = results / f"{index:02d}.json"
        if result_path.is_file():
            continue
        if not claim(args.claims, index, args.stale_seconds):
            continue
        asset = assets[index]
        started = time.perf_counter()
        row = {"index": index, "id": asset["id"], "source": asset["path"],
               "source_sha256": asset["sha256"], "status": "failed"}
        try:
            one = argparse.Namespace(
                manifest=args.manifest, index=index, vae3d_root=args.vae3d_root,
                output_root=args.output_root / "targets", stage="both",
                resolution=args.resolution, raycast_backend="cpu", distance_backend="cpu",
                max_sites=args.max_sites, verify_reference=False,
            )
            output = prepare_one(one)
            pair = (Path(output["A1"]["cache"]), Path(output["A2"]["cache"]))
            row["caches"] = {"A1": str(pair[0]), "A2": str(pair[1])}
            row["metrics"] = evaluate_pair(pair[0], pair[1], Path(asset["path"]),
                                           resolution=args.resolution,
                                           samples=args.samples, seed=20261029)
            row["status"] = "success"
        except Exception as error:  # noqa: BLE001 - one asset must not kill the worker
            row["error"] = f"{type(error).__name__}: {error}"
            row["traceback"] = traceback.format_exc()[-600:]
        row["seconds"] = round(time.perf_counter() - started, 1)
        temporary = result_path.with_name(result_path.name + ".tmp")
        temporary.write_text(json.dumps(row), encoding="utf-8")
        temporary.replace(result_path)
        try:
            (args.claims / f"{index:05d}.claim").unlink()
        except OSError:
            pass
        done += 1
        print(json.dumps({"index": index, "status": row["status"],
                          "seconds": row["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())