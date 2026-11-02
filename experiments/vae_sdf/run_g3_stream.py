"""Run A1/A2 and A4 as decoupled producer and consumer pools.

The group pipeline coupled the stages: A1/A2 could only build assets that still
needed an A4 target, and A4 only start after a whole group's initial fields
existed, so each side idled while the other worked.

Here the asset is the work unit. A fixed pool of A1/A2 workers claims one asset at
a time and immediately takes the next, so a slow asset no longer delays a dozen
others. Every finished field becomes available to the A4 pool, which runs on its
own shard of GPUs and skips assets the producers have not reached yet. Production
pauses on a byte budget for the intermediate cache, so the slower stage becomes
the pacer instead of the disk filling up.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

from .run_initial_batch import summarize

PRODUCER_MODULE = "experiments.vae_sdf.run_g3_producer_worker"
CONSUMER_MODULE = "experiments.vae_sdf.run_g3_a4_batch"
ROUTE = "flexicubes_smooth"
DEFAULT_ASSET_MB = 135.0


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 2 ** 30


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def count_files(directory: Path, suffix: str = ".json") -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for entry in os.scandir(directory)
               if entry.is_file() and entry.name.endswith(suffix))


def a4_present(root: Path, asset_id: str, resolution: int, route: str = ROUTE) -> bool:
    base = root / asset_id / f"R{resolution}"
    if not base.is_dir():
        return False
    return any((cache / "a4" / "target.npz").is_file()
               for cache in base.glob(f"{route}-*"))


def chunk(items: list[int], size: int) -> list[list[int]]:
    return [items[start:start + size] for start in range(0, len(items), size)]


def mean_asset_mb(root: Path, sample: int = 12) -> float:
    """Average size of a finished A1/A2 entry, for the byte budget."""

    target_root = root / "targets"
    if not target_root.is_dir():
        return DEFAULT_ASSET_MB
    sizes = []
    for entry in os.scandir(target_root):
        if not entry.is_dir():
            continue
        total = 0
        for path in Path(entry.path).rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        if total:
            sizes.append(total / 2 ** 20)
        if len(sizes) >= sample:
            break
    return round(sum(sizes) / len(sizes), 1) if sizes else DEFAULT_ASSET_MB


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--load-audit", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--a1a2-root", type=Path, required=True)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-sites", type=int, default=6_000_000)
    parser.add_argument("--producers", type=int, default=22)
    parser.add_argument("--consumers", type=int, default=10)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--slice-size", type=int, default=100)
    parser.add_argument("--pending-gb", type=float, default=400.0,
                        help="pause producing above this much intermediate cache")
    parser.add_argument("--min-free-gb", type=float, default=150.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--skip-indices", type=int, nargs="*", default=[],
                        help="Manifest indices deliberately left without an A4 target; a "
                             "batch has no per-asset timeout, so one pathological asset is "
                             "otherwise re-offered after every failure and stalls its shard")
    parser.add_argument("--summary-refresh-seconds", type=float, default=60.0)
    args = parser.parse_args()

    for path in (args.a1a2_root, args.targets_root, args.records, args.logs):
        path.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_bytes())
    assets = manifest["assets"]
    results_dir = args.a1a2_root / "assets"
    summary_path = args.a1a2_root / "summary.json"
    claims = args.a1a2_root / "claims"
    pause_flag = args.a1a2_root / "pause"
    queue_path = args.a1a2_root / "producer-queue.txt"

    todo = [index for index in range(len(assets))
            if not (results_dir / f"{index:02d}.json").is_file()]
    queue_path.write_text(" ".join(str(index) for index in todo), encoding="utf-8")
    skip = {index for index in args.skip_indices if 0 <= index < len(assets)}
    outstanding = sum(1 for index in range(len(assets))
                      if index not in skip
                      and not a4_present(args.targets_root, assets[index]["id"],
                                         args.resolution))
    asset_mb = mean_asset_mb(args.a1a2_root)

    print(json.dumps({"event": "start", "assets": len(assets), "producer_queue": len(todo),
                      "outstanding_targets": outstanding, "mean_asset_mb": asset_mb,
                      "pending_gb": args.pending_gb,
                      "producers": args.producers, "consumers": args.consumers}), flush=True)

    env = dict(os.environ)
    python = sys.executable
    stop = threading.Event()
    summary_lock = threading.Lock()

    def refresh_summary() -> None:
        with summary_lock:
            plan = json.loads((args.a1a2_root / "plan.json").read_text(encoding="utf-8"))
            write_json(summary_path, summarize(assets, args.a1a2_root, plan))

    def refresher() -> None:
        while not stop.is_set():
            try:
                refresh_summary()
            except Exception as error:  # noqa: BLE001 - keep the loop alive
                print(json.dumps({"event": "summary_error",
                                  "error": f"{type(error).__name__}: {error}"}), flush=True)
            stop.wait(args.summary_refresh_seconds)

    producers: dict[int, subprocess.Popen] = {}
    consumers: dict[int, tuple[subprocess.Popen, list[int]]] = {}
    inflight_indices: set[int] = set()
    serial = {"producer": 0, "consumer": 0}

    def start_producer() -> None:
        serial["producer"] += 1
        number = serial["producer"]
        log = (args.logs / f"produce-{number:04d}.log").open("a")
        command = [python, "-B", "-m", PRODUCER_MODULE,
                   "--manifest", str(args.manifest), "--load-audit", str(args.load_audit),
                   "--vae3d-root", str(args.vae3d_root),
                   "--output-root", str(args.a1a2_root),
                   "--queue", str(queue_path), "--claims", str(claims),
                   "--resolution", str(args.resolution), "--max-sites", str(args.max_sites)]
        producers[number] = subprocess.Popen(command, stdout=log,
                                             stderr=subprocess.STDOUT, env=env)

    def start_consumer(group: list[int]) -> None:
        serial["consumer"] += 1
        number = serial["consumer"]
        gpu = args.gpus[number % len(args.gpus)]
        log = (args.logs / f"consume-{number:04d}.log").open("a")
        command = [python, "-B", "-m", CONSUMER_MODULE,
                   "--manifest", str(args.manifest), "--a1a2-summary", str(summary_path),
                   "--a0-audit", str(args.a0_audit), "--vae3d-root", str(args.vae3d_root),
                   "--a1a2-root", str(args.a1a2_root),
                   "--output-root", str(args.targets_root),
                   "--records", str(args.records / f"stream{number:03d}"),
                   "--physical-gpu", str(gpu), "--resolution", str(args.resolution),
                   "--indices", *[str(i) for i in group]]
        inflight_indices.update(group)
        consumers[number] = (subprocess.Popen(command, stdout=log,
                                              stderr=subprocess.STDOUT, env=env), group)

    def unclaimed_left() -> int:
        count = 0
        for index in todo:
            if (results_dir / f"{index:02d}.json").is_file():
                continue
            if not (claims / f"{index:05d}.claim").exists():
                count += 1
        return count

    watcher = threading.Thread(target=refresher, daemon=True)
    try:
        if not summary_path.is_file():
            refresh_summary()
        watcher.start()
        while True:
            for number, process in list(producers.items()):
                if process.poll() is not None:
                    del producers[number]
            for number, (process, group) in list(consumers.items()):
                if process.poll() is None:
                    continue
                del consumers[number]
                inflight_indices.difference_update(group)

            fields = count_files(results_dir)
            targets = sum(1 for entry in os.scandir(args.targets_root) if entry.is_dir()) \
                if args.targets_root.is_dir() else 0
            pending = max(0, fields - targets)
            pending_gb = pending * asset_mb / 1024
            disk_free = free_gb(args.targets_root)
            over_budget = pending_gb >= args.pending_gb or disk_free < args.min_free_gb
            if over_budget:
                pause_flag.write_text("paused\n", encoding="utf-8")
            elif pause_flag.exists():
                pause_flag.unlink()

            # A4 only ever gets assets whose field already exists; scanning for
            # them is much cheaper than keeping a queue in sync with the producers
            if len(consumers) < args.consumers:
                ready = []
                for entry in os.scandir(results_dir):
                    name = entry.name
                    if not name.endswith(".json"):
                        continue
                    index = int(name[:-5])
                    if index in inflight_indices or index in skip:
                        continue
                    if not a4_present(args.targets_root, assets[index]["id"],
                                      args.resolution):
                        ready.append(index)
                    if len(ready) >= args.slice_size:
                        break
                if ready:
                    start_consumer(ready)

            missing_fields = len(assets) - fields
            if not over_budget and len(producers) < args.producers and missing_fields > 0:
                while len(producers) < args.producers and missing_fields > 0:
                    start_producer()
                    missing_fields = len(assets) - count_files(results_dir)

            print(json.dumps({"event": "status", "fields": fields, "targets": targets,
                              "pending": pending, "pending_gb": round(pending_gb, 1),
                              "free_gb": round(disk_free, 1), "paused": over_budget,
                              "producers": len(producers), "consumers": len(consumers),
                              "missing_fields": missing_fields}), flush=True)

            if not producers and not consumers:
                # nothing is running: settle whether any work is actually left
                fields = count_files(results_dir)
                remaining = sum(1 for index in range(len(assets))
                                if index not in skip
                                and (results_dir / f"{index:02d}.json").is_file()
                                and not a4_present(args.targets_root, assets[index]["id"],
                                                   args.resolution))
                if len(assets) == fields and remaining == 0:
                    print(json.dumps({"event": "finished", "fields": fields,
                                      "targets": targets}), flush=True)
                    return 0
                if len(assets) == fields and remaining > 0:
                    print(json.dumps({"event": "stalled",
                                      "remaining_targets": remaining}), flush=True)
                    return 1
            stop.wait(args.poll_seconds)
    except KeyboardInterrupt:
        return 1
    except Exception as error:  # noqa: BLE001 - report what broke, then stop
        print(json.dumps({"event": "failed", "error": f"{type(error).__name__}: {error}",
                          "traceback": traceback.format_exc()[-800:]}), flush=True)
        return 1
    finally:
        stop.set()
        try:
            pause_flag.write_text("stopping\n", encoding="utf-8")
        except OSError:
            pass
        for process in list(producers.values()):
            process.terminate()
        for process, _ in list(consumers.values()):
            process.terminate()
        watcher.join(timeout=30)
        try:
            refresh_summary()
        except Exception:  # noqa: BLE001 - best effort on the way out
            pass


if __name__ == "__main__":
    raise SystemExit(main())