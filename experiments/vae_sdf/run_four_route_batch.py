"""Sequential, resumable single-GPU driver for the fixed four-route G0 subset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .run_four_route_asset import ROUTES


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--routes", nargs="+", choices=tuple(ROUTES), default=tuple(ROUTES))
    parser.add_argument("--a3-steps", type=int, default=30)
    parser.add_argument("--a3-samples", type=int, default=8192)
    parser.add_argument("--a4-steps", type=int, default=200)
    parser.add_argument("--image-resolution", type=int, default=4096)
    parser.add_argument("--metric-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise ValueError("Batch driver requires physical GPU 7 only")
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "batch-status.json"
    identity = {"indices": list(args.indices), "routes": list(args.routes),
                "a3_steps": args.a3_steps, "a3_samples": args.a3_samples,
                "a4_steps": args.a4_steps, "image_resolution": args.image_resolution,
                "metric_samples": args.metric_samples, "seed": args.seed,
                "physical_gpu": 7}
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status["identity"] != identity:
            raise ValueError("Existing batch status has a different protocol")
    else:
        status = {"identity": identity, "runs": {}, "started_at": time.time()}
        save(status_path, status)
    for index in args.indices:
        for route in args.routes:
            key = f"{index}:{route}"
            if status["runs"].get(key, {}).get("status") == "success":
                continue
            command = [sys.executable, "-m", "experiments.vae_sdf.run_four_route_asset",
                "--manifest", str(args.manifest), "--a1a2-summary", str(args.a1a2_summary),
                "--a0-audit", str(args.a0_audit), "--vae3d-root", str(args.vae3d_root),
                "--output-root", str(args.output_root), "--route", route, "--index", str(index),
                "--a3-steps", str(args.a3_steps), "--a3-samples", str(args.a3_samples),
                "--a4-steps", str(args.a4_steps), "--image-resolution", str(args.image_resolution),
                "--metric-samples", str(args.metric_samples), "--seed", str(args.seed), "--resume"]
            log = args.output_root / f"index{index}-{route}.log"
            status["runs"][key] = {"status": "running", "command": command,
                                   "log": str(log), "started_at": time.time()}
            save(status_path, status)
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        check=False)
            status["runs"][key].update(status="success" if result.returncode == 0 else "failed",
                                       exit_code=result.returncode, finished_at=time.time())
            save(status_path, status)
            print(f"{key}: {status['runs'][key]['status']} ({result.returncode})", flush=True)
    status["finished_at"] = time.time()
    save(status_path, status)
    return 0 if all(run["status"] == "success" for run in status["runs"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
