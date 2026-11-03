"""Retry the A4 assets the main stream gave up on, one at a time.

They are in the ledger because they failed at least twice under the production
settings: most raise a CUDA fault inside the rasterizer at 4096 px views, one is
a 12.7 M-face mesh that ran for eighty minutes. Retrying them here keeps them off
the shard schedule, runs them sequentially so a single bad asset cannot occupy a
card for the whole pool, and drops the render resolution - the target is a
voxel-grid field, so 2048 px views change the fit only marginally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-root", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--image-resolution", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="Retry only these indices instead of the whole ledger")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    wanted = set(args.indices) if args.indices else None

    args.logs.mkdir(parents=True, exist_ok=True)
    report_path = args.logs / "retry-skipped.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else []

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    paths = {
        "--manifest": args.manifest,
        "--a1a2-summary": args.a1a2_summary,
        "--a0-audit": args.a0_audit,
        "--vae3d-root": args.vae3d_root,
        "--output-root": args.output_root,
    }
    for row in ledger:
        index = row["index"]
        if wanted is not None and index not in wanted:
            continue
        asset_id = assets[index]["id"]
        cache = args.a1a2_root / "targets" / asset_id
        if not cache.is_dir():
            print(json.dumps({"index": index, "status": "no_a1a2_cache"}), flush=True)
            report.append({"index": index, "status": "no_a1a2_cache"})
            continue
        log_path = args.logs / f"retry-{index:05d}.log"
        command = [sys.executable, "-B", "-m", "experiments.vae_sdf.run_four_route_asset"]
        for key, value in paths.items():
            command += [key, str(value)]
        command += ["--route", "flexicubes_smooth", "--index", str(index),
                    "--resolution", str(args.resolution),
                    "--image-resolution", str(args.image_resolution),
                    "--a3-samples", "8192", "--metric-samples", "4096",
                    "--seed", "42", "--device", "cuda:0", "--resume"]
        started = time.time()
        with log_path.open("w") as stream:
            try:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        env=env, timeout=args.timeout_seconds, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        row_out = {"index": index, "id": asset_id, "status": "success" if code == 0 else code,
                   "seconds": round(time.time() - started, 1),
                   "image_resolution": args.image_resolution}
        print(json.dumps(row_out), flush=True)
        report.append(row_out)
        report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())