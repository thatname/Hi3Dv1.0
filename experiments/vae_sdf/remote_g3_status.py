"""Read G3 pipeline progress for the local viewer without touching any GPU state.

Reports the four stages that matter before training starts: the manifest/split,
the frozen capacity policy, the A1/A2 stage and the A4 target stage. Everything
here is read-only and cheap enough to poll.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def read_json(path: Path, *, optional: bool = False):
    for attempt in range(4):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if optional:
                return None
            raise
        except json.JSONDecodeError:
            if attempt == 3:
                raise
            time.sleep(.05)


def du_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def count_dirs(path: Path) -> int:
    if not path.is_dir():
        return 0


    return sum(1 for entry in os.scandir(path) if entry.is_dir())


def running(pattern: str) -> int:
    try:
        out = subprocess.run(["pgrep", "-fc", pattern], capture_output=True, text=True,
                             check=False)
        return int(out.stdout.strip() or 0)
    except (OSError, ValueError):
        return 0


def a4_progress(root: Path, resolution: int, window: float = 900.0) -> dict:
    """Count assets whose A4 target exists, and how many finished recently."""

    present = 0
    recent = 0
    cutoff = time.time() - window
    if root.is_dir():
        for asset in os.scandir(root):
            if not asset.is_dir():
                continue
            for cache in (Path(asset.path) / f"R{resolution}").glob("flexicubes_smooth-*"):
                target = cache / "a4" / "target.npz"
                if target.is_file():
                    present += 1
                    if target.stat().st_mtime >= cutoff:
                        recent += 1
                    break
    records = root.parent / "logs" / "a4-records"
    by_gpu: dict[str, int] = {}
    newest: list[tuple[float, Path]] = []
    for record_root in sorted(records.glob("gpu*")):
        files = list(record_root.glob("*.json"))
        by_gpu[record_root.name] = len(files)
        newest.extend((path.stat().st_mtime, path) for path in files)
    newest.sort(reverse=True)
    seconds = []
    for _, path in newest[:300]:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(row.get("seconds"), (int, float)) and row.get("status") == "success":
            seconds.append(row["seconds"])
    return {"present": present, "recent": recent, "records_by_gpu": by_gpu,
            "seconds_median": sorted(seconds)[len(seconds) // 2] if seconds else None,
            "seconds_count": len(seconds)}


def scan_rate(directory: Path, cutoff: float, suffix: str = ".json") -> tuple[int, int]:
    """Total and recently touched files in one directory, in a single scan."""

    total = 0
    recent = 0
    if not directory.is_dir():
        return 0, 0
    for entry in os.scandir(directory):
        if not entry.is_file() or not entry.name.endswith(suffix):
            continue
        total += 1
        try:
            if entry.stat().st_mtime >= cutoff:
                recent += 1
        except OSError:
            pass
    return total, recent


def eta_block(done: int, total: int, recent: int, window: float) -> dict:
    remaining = max(0, total - done)
    per_second = recent / window if window > 0 else 0.0
    if per_second <= 0:
        return {"per_minute": 0.0, "remaining": remaining, "eta_hours": None,
                "finish_local": None}
    hours = remaining / per_second / 3600
    finish = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + hours * 3600))
    return {"per_minute": round(per_second * 60, 1), "remaining": remaining,
            "eta_hours": round(hours, 1), "finish_local": finish}


def snapshot(experiment: Path, resolution: int = 256) -> dict:
    experiment = experiment.resolve()
    if not experiment.is_relative_to(Path("/data/hi3d").resolve()):
        raise ValueError("G3 experiment must be inside /data/hi3d")
    manifest = read_json(experiment / "manifest" / "manifest-g3-full.json", optional=True)
    split = read_json(experiment / "manifest" / "split-g3-full-v2-capacity.json", optional=True)
    split_v1 = read_json(experiment / "manifest" / "split-g3-full-v1.json", optional=True)
    policy = read_json(experiment / "manifest" / "capacity-policy-v1.json", optional=True)
    build = read_json(experiment / "manifest" / "build-summary.json", optional=True)
    a1a2_summary = read_json(experiment / "targets" / "a1a2" / "summary.json", optional=True)
    a1a2_counts = (a1a2_summary or {}).get("counts", {})

    state = {
        "schema": "g3_pipeline_v1",
        "experiment": str(experiment),
        "resolution": resolution,
        "manifest": {
            "assets": len((manifest or {}).get("assets", [])),
            "counts": (build or {}).get("split"),
            "loaded": (build or {}).get("loaded"),
            "failed": (build or {}).get("failed"),
            "degenerate": (build or {}).get("degenerate"),
            "faces_median": (build or {}).get("faces_median"),
            "manifest_sha256": (build or {}).get("manifest_sha256"),
        },
        "split": {"train": len(split["train_ids"]), "val": len(split["val_ids"]),
                  "test": len(split["test_ids"])} if split else None,
        "split_v1": {"train": len(split_v1["train_ids"]), "val": len(split_v1["val_ids"]),
                     "test": len(split_v1["test_ids"])} if split_v1 else None,
        "policy": None if not policy else {
            "threshold_cells": policy["threshold_cells"],
            "excluded_total": policy["excluded_total"],
            "excluded_fraction": policy["excluded_fraction"],
            "model": policy["model"],
            "p90_abs_rel_error": policy["calibration_error"]["p90_abs_rel_error"],
        },
        "a1a2": {"counts": a1a2_counts or None,
                 "seconds_median": ((a1a2_summary or {}).get("seconds_median")),
                 "running": running("experiments.vae_sdf.run_initial_batch")},
        "a4": a4_progress(experiment / "targets" / "a4", resolution),
        "disk": {"targets_bytes": du_bytes(experiment / "targets")},
    }
    state["a4"]["running"] = running("experiments.vae_sdf.run_g3_a4_batch")
    state["a4"]["target"] = state["manifest"]["assets"]

    # live rates come from file timestamps, so the estimate reflects what the run
    # is actually doing rather than what it was configured to do
    window = 900.0
    cutoff = time.time() - window
    a1a2_done, a1a2_recent = scan_rate(experiment / "targets" / "a1a2" / "assets", cutoff)
    total = state["manifest"]["assets"]
    state["rates"] = {
        "window_minutes": window / 60,
        "a1a2": {**eta_block(a1a2_done, total, a1a2_recent, window),
                 "recent": a1a2_recent, "done": a1a2_done},
        "a4": {**eta_block(state["a4"]["present"], total,
                           state["a4"].get("recent", 0), window),
               "recent": state["a4"].get("recent", 0),
               "done": state["a4"]["present"]},
    }
    stage_hours = [state["rates"][stage]["eta_hours"] for stage in ("a1a2", "a4")
                   if state["rates"][stage]["eta_hours"] is not None]
    finish = max(stage_hours) if stage_hours else None
    state["rates"]["overall_eta_hours"] = finish
    state["rates"]["overall_finish_local"] = (
        time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + finish * 3600))
        if finish is not None else None)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.experiment, args.resolution), ensure_ascii=False,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())