"""Sequential, resumable, multi-GPU driver for the fixed four-route G0 subset.

Several processes may share one output root: each worker takes a disjoint index
subset on its own whitelisted physical GPU, and every status update re-reads the
status file under an exclusive lock so concurrent workers cannot clobber each
other. The physical card is recorded per run, never in the hashed target config,
so one protocol keeps one cache identity whichever GPU produced it.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .run_four_route_asset import GPU_WHITELIST, ROUTES


def save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@contextlib.contextmanager
def status_lock(root: Path):
    """Exclusive lock so parallel workers update one status file safely."""
    with (root / ".batch-status.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def protocol_identity(args) -> dict:
    return {"routes": list(args.routes), "a3_steps": args.a3_steps,
            "a3_samples": args.a3_samples, "a4_steps": args.a4_steps,
            "image_resolution": args.image_resolution,
            "metric_samples": args.metric_samples, "seed": args.seed}


def mutate_status(root: Path, identity: dict, gpu: int, mutate) -> dict:
    """Load-modify-save the shared status file under the exclusive lock.

    Legacy status files recorded indices and the physical GPU inside identity;
    those keys are ignored so an existing root stays resumable.
    """
    path = root / "batch-status.json"
    with status_lock(root):
        if path.exists():
            status = json.loads(path.read_text())
            legacy = {k: v for k, v in status["identity"].items()
                      if k not in ("indices", "physical_gpu")}
            if legacy != identity:
                raise ValueError("Existing batch status has a different protocol")
        else:
            status = {"identity": identity, "runs": {}, "started_at": time.time()}
        status.setdefault("runs", {})
        # Coverage is derived from the runs actually recorded, so parallel shards
        # compose into one complete index list.
        status["indices"] = sorted({int(key.split(":")[0]) for key in status["runs"]})
        status.setdefault("physical_gpus", [])
        if gpu not in status["physical_gpus"]:
            status["physical_gpus"] = sorted(set(status["physical_gpus"]) | {gpu})
        mutate(status)
        save(path, status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--routes", nargs="+", choices=tuple(ROUTES), default=tuple(ROUTES))
    parser.add_argument("--physical-gpu", type=int, default=7, choices=GPU_WHITELIST)
    parser.add_argument("--a3-steps", type=int, default=30)
    parser.add_argument("--a3-samples", type=int, default=8192)
    parser.add_argument("--a4-steps", type=int, default=200)
    parser.add_argument("--image-resolution", type=int, default=4096)
    parser.add_argument("--metric-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != str(args.physical_gpu):
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES={visible!r} disagrees with --physical-gpu {args.physical_gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    args.output_root.mkdir(parents=True, exist_ok=True)
    identity = protocol_identity(args)
    mutate_status(args.output_root, identity, args.physical_gpu, lambda status: None)
    for index in args.indices:
        for route in args.routes:
            key = f"{index}:{route}"
            current = mutate_status(args.output_root, identity, args.physical_gpu,
                                    lambda status: None)
            if current["runs"].get(key, {}).get("status") == "success":
                continue
            command = [sys.executable, "-m", "experiments.vae_sdf.run_four_route_asset",
                "--manifest", str(args.manifest), "--a1a2-summary", str(args.a1a2_summary),
                "--a0-audit", str(args.a0_audit), "--vae3d-root", str(args.vae3d_root),
                "--output-root", str(args.output_root), "--route", route, "--index", str(index),
                "--a3-steps", str(args.a3_steps), "--a3-samples", str(args.a3_samples),
                "--a4-steps", str(args.a4_steps), "--image-resolution", str(args.image_resolution),
                "--metric-samples", str(args.metric_samples), "--seed", str(args.seed), "--resume"]
            log = args.output_root / f"index{index}-{route}.log"
            started = time.time()

            def start(status, key=key, command=command, log=log, started=started):
                status["runs"][key] = {"status": "running", "command": command,
                                       "log": str(log), "started_at": started,
                                       "physical_gpu": args.physical_gpu}

            mutate_status(args.output_root, identity, args.physical_gpu, start)
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        check=False)
            finished = time.time()

            def finish(status, key=key, result=result, finished=finished):
                status["runs"][key].update(
                    status="success" if result.returncode == 0 else "failed",
                    exit_code=result.returncode, finished_at=finished)
                # A shard is finished only when every recorded run reached a terminal state.
                if all(run.get("status") in {"success", "failed"}
                       for run in status["runs"].values()):
                    status["finished_at"] = finished

            mutate_status(args.output_root, identity, args.physical_gpu, finish)
            print(f"{key}: {'success' if result.returncode == 0 else 'failed'} "
                  f"({result.returncode}) gpu{args.physical_gpu}", flush=True)
    status = mutate_status(args.output_root, identity, args.physical_gpu, lambda value: None)
    return 0 if all(run["status"] == "success" for run in status["runs"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())