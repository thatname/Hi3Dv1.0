"""A4-side driver for the G3 full-pool experiment: run, then prune.

The pool cannot hold every asset's A1/A2 plus A3/A4 at once (~6 TB), so each
successful asset is pruned back to its A4 directory immediately: the A1/A2 entry
and the A3 fit are deleted, and the A0 audit only needs identity here because no
sign probe was run on the pool. One process per whitelisted GPU, disjoint index
shards, resumable by checking which assets already have an A4 target.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .run_four_route_asset import GPU_WHITELIST, ROUTES, SUPPORTED_RESOLUTIONS


def a4_present(root: Path, asset_id: str, resolution: int, route: str) -> bool:
    base = root / asset_id / f"R{resolution}"
    if not base.is_dir():
        return False
    for cache in base.glob(f"{route}-*"):
        target = cache / "a4" / "target.npz"
        mesh = cache / "a4" / "raw_mesh.ply"
        if target.is_file() and mesh.is_file():
            return True
    return False


def prune(a1a2_root: Path, four_root: Path, asset_id: str, resolution: int,
          route: str, keep_a1a2: bool) -> dict:
    removed = []
    if not keep_a1a2:
        stage_dir = a1a2_root / "targets" / asset_id
        if stage_dir.is_dir():
            shutil.rmtree(stage_dir)
            removed.append(str(stage_dir))
    base = four_root / asset_id / f"R{resolution}"
    if base.is_dir():
        for cache in base.glob(f"{route}-*"):
            for stage in ("a3",):
                folder = cache / stage
                if folder.is_dir():
                    shutil.rmtree(folder)
                    removed.append(str(folder))
    return {"removed": removed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--a1a2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=GPU_WHITELIST)
    parser.add_argument("--route", default="flexicubes_smooth", choices=tuple(ROUTES))
    parser.add_argument("--resolution", type=int, default=256, choices=SUPPORTED_RESOLUTIONS)
    parser.add_argument("--a3-steps", type=int, default=30)
    parser.add_argument("--a3-samples", type=int, default=8192)
    parser.add_argument("--a4-steps", type=int, default=200)
    parser.add_argument("--image-resolution", type=int, default=4096)
    parser.add_argument("--metric-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-a1a2", action="store_true")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != str(args.physical_gpu):
        raise ValueError(f"CUDA_VISIBLE_DEVICES={visible!r} disagrees with --physical-gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    args.records.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    assets = manifest["assets"]
    for index in args.indices:
        asset_id = assets[index]["id"]
        record_path = args.records / f"{index:05d}.json"
        if a4_present(args.output_root, asset_id, args.resolution, args.route):
            if not record_path.is_file():
                record_path.write_text(json.dumps(
                    {"index": index, "id": asset_id, "status": "skipped_present"}) + "\n")
            continue
        log = args.records / f"{index:05d}.log"
        command = [sys.executable, "-m", "experiments.vae_sdf.run_four_route_asset",
                   "--manifest", str(args.manifest), "--a1a2-summary", str(args.a1a2_summary),
                   "--a0-audit", str(args.a0_audit), "--vae3d-root", str(args.vae3d_root),
                   "--output-root", str(args.output_root), "--route", args.route,
                   "--index", str(index), "--resolution", str(args.resolution),
                   "--a3-steps", str(args.a3_steps), "--a3-samples", str(args.a3_samples),
                   "--a4-steps", str(args.a4_steps),
                   "--image-resolution", str(args.image_resolution),
                   "--metric-samples", str(args.metric_samples), "--seed", str(args.seed),
                   "--resume"]
        started = time.time()
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        row = {"index": index, "id": asset_id, "physical_gpu": args.physical_gpu,
               "exit_code": result.returncode, "seconds": round(time.time() - started, 1),
               "status": "success" if result.returncode == 0 else "failed"}
        if result.returncode == 0:
            row.update(prune(args.a1a2_root, args.output_root, asset_id, args.resolution,
                             args.route, args.keep_a1a2))
        else:
            tail = log.read_text().strip().splitlines()[-5:]
            row["error"] = " | ".join(tail)[:500]
        record_path.write_text(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("index", "id", "status", "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())