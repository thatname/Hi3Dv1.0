"""After the GPU 7 batch finishes, build shared meshes and fixed-protocol audits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--a4-summary", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise ValueError("Post-run GPU checks are pinned to physical GPU 7")
    status_path = args.root / "post-status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    status = {"state": "waiting_for_batch", "started_at": time.time(), "steps": {}}
    save(status_path, status)
    while True:
        batch = json.loads((args.root / "batch-status.json").read_text())
        if batch.get("finished_at") is not None:
            break
        time.sleep(30)
    status["state"] = "running"
    status["batch_counts"] = {kind: sum(x["status"] == kind for x in batch["runs"].values())
                              for kind in ("success", "failed")}
    save(status_path, status)
    asset_count = len(json.loads(args.manifest.read_text())["assets"])
    commands = {
        "finite_difference": [sys.executable, "-m", "experiments.vae_sdf.four_route_probe",
            "--a4-summary", str(args.a4_summary),
            "--index", str(min(48, asset_count - 1)), "--finite-difference"],
        "shared_a1a2": [sys.executable, "-m", "experiments.vae_sdf.build_four_route_shared",
            "--manifest", str(args.manifest), "--a1a2-summary", str(args.a1a2_summary),
            "--output-root", str(args.root), "--indices", *map(str, range(asset_count))],
        "paired_summary": [sys.executable, "-m", "experiments.vae_sdf.summarize_four_route_batch",
            "--manifest", str(args.manifest), "--root", str(args.root),
            "--output", str(args.root / "paired-summary.json")],
    }
    for name, command in commands.items():
        log = args.root / f"post-{name}.log"
        status["steps"][name] = {"status": "running", "command": command,
                                  "log": str(log), "started_at": time.time()}
        save(status_path, status)
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        status["steps"][name].update(status="success" if result.returncode == 0 else "failed",
                                      exit_code=result.returncode, finished_at=time.time())
        save(status_path, status)
        print(f"{name}: {status['steps'][name]['status']}", flush=True)
    status["state"] = "complete"
    status["finished_at"] = time.time()
    save(status_path, status)
    return 0 if all(x["status"] == "success" for x in status["steps"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
