"""Drive the G3 target pipeline: A1/A2 on many cores, A4 on many cards, overlapped.

The pool cannot hold every asset's A1/A2 plus its A4 cache at once, so work runs
in index groups: A1/A2 builds a group while A4 consumes the group before it, and
each successful A4 prunes its A1/A2 entry immediately. Index shards keep the two
stages busy in parallel without ever letting the intermediate cache run ahead of
what the cards can drain.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .run_initial_batch import summarize


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 2 ** 30


def chunks(items: list[int], parts: int) -> list[list[int]]:
    parts = max(1, min(parts, len(items)))
    size = -(-len(items) // parts)
    return [items[start:start + size] for start in range(0, len(items), size)]


def run(command: list[str], log: Path, env: dict) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        stream.flush()
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                              env=env, check=False).returncode


def a4_present(root: Path, asset_id: str, resolution: int) -> bool:
    base = root / asset_id / f"R{resolution}"
    if not base.is_dir():
        return False
    return any((cache / "a4" / "target.npz").is_file()
               for cache in base.glob("flexicubes_smooth-*"))


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
    parser.add_argument("--indices", type=int, nargs="+",
                        help="defaults to every manifest index whose A4 target is missing")
    parser.add_argument("--group", type=int, default=400)
    parser.add_argument("--a1a2-workers", type=int, default=64)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-sites", type=int, default=6_000_000)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--a4-steps", type=int, default=200)
    parser.add_argument("--max-groups", type=int, default=0,
                        help="stop after this many groups, for smoke runs")
    parser.add_argument("--sequential", action="store_true",
                        help="do not overlap A1/A2 with A4 (debugging only)")
    parser.add_argument("--min-free-gb", type=float, default=60.0,
                        help="stop cleanly before a group when the volume runs this low")
    parser.add_argument("--prefetch-groups", type=int, default=4,
                        help="how many groups of A1/A2 may be built ahead of A4")
    parser.add_argument("--prefetch-interval", type=float, default=10.0,
                        help="seconds between refills of the A1/A2 buffer")
    parser.add_argument("--producer-workers", type=int, default=3,
                        help="how many groups may be built at once; one group at a time "
                             "leaves the CPU idle while a group's tail drains")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    assets = manifest["assets"]
    indices = args.indices if args.indices else [
        index for index, asset in enumerate(assets)
        if not a4_present(args.targets_root, asset["id"], args.resolution)]
    groups = chunks(indices, -(-max(1, len(indices)) // max(1, args.group)))
    if args.max_groups:
        groups = groups[:args.max_groups]
    print(json.dumps({"indices": len(indices), "groups": len(groups),
                      "group_size": args.group, "a1a2_workers": args.a1a2_workers,
                      "gpus": args.gpus}), flush=True)

    env = dict(os.environ)
    python = sys.executable
    summary_path = args.a1a2_root / "summary.json"
    args.a1a2_root.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)

    def refresh_summary() -> None:
        write_json(summary_path, summarize(assets, args.a1a2_root,
                                           json.loads((args.a1a2_root / "plan.json").read_text())))

    def run_a1a2(group: list[int], tag: str) -> dict:
        shards = chunks(group, args.a1a2_workers)
        base = [python, "-B", "-m", "experiments.vae_sdf.run_initial_batch_single_thread",
                "--manifest", str(args.manifest), "--load-audit", str(args.load_audit),
                "--vae3d-root", str(args.vae3d_root), "--output-root", str(args.a1a2_root),
                "--resolution", str(args.resolution), "--max-sites", str(args.max_sites),
                "--samples", str(args.samples), "--no-per-asset-summary", "--indices"]
        started = time.time()
        # one shard publishes plan.json first, so the others never race to create it;
        # it keeps running in the background instead of blocking the whole group
        first_log = (args.logs / f"a1a2-{tag}-000.log").open("a")
        first = subprocess.Popen(base + [str(i) for i in shards[0]], stdout=first_log,
                                 stderr=subprocess.STDOUT, env=env)
        plan_path = args.a1a2_root / "plan.json"
        deadline = time.time() + 180
        while not plan_path.is_file() and time.time() < deadline:
            if first.poll() is not None:
                break
            time.sleep(1)
        rest = [subprocess.Popen(base + [str(i) for i in shard],
                                stdout=(args.logs / f"a1a2-{tag}-{number:03d}.log").open("a"),
                                stderr=subprocess.STDOUT, env=env)
                for number, shard in enumerate(shards[1:], 1)]
        codes = [process.wait() for process in rest] + [first.wait()]
        if any(code not in (0, 1) for code in codes):
            raise RuntimeError(f"A1/A2 shard failed: {sorted(set(codes))}")
        refresh_summary()
        return {"stage": "a1a2", "group": tag, "assets": len(group),
                "seconds": round(time.time() - started, 1)}

    def run_a4(group: list[int], tag: str) -> dict:
        shards = chunks(group, len(args.gpus))
        started = time.time()
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            list(pool.map(lambda pair: run(
                [python, "-B", "-m", "experiments.vae_sdf.run_g3_a4_batch",
                 "--manifest", str(args.manifest), "--a1a2-summary", str(summary_path),
                 "--a0-audit", str(args.a0_audit), "--vae3d-root", str(args.vae3d_root),
                 "--a1a2-root", str(args.a1a2_root), "--output-root", str(args.targets_root),
                 "--records", str(args.records / f"gpu{pair[0]}"),
                 "--physical-gpu", str(pair[0]), "--resolution", str(args.resolution),
                 "--a4-steps", str(args.a4_steps),
                 "--indices", *[str(i) for i in pair[1]]],
                args.logs / f"a4-{tag}-gpu{pair[0]}.log", env),
                zip(args.gpus, shards)))
        return {"stage": "a4", "group": tag, "assets": len(group),
                "seconds": round(time.time() - started, 1)}

    try:
        stop = threading.Event()
        futures: dict[int, object] = {}
        finished_a1a2 = {int(path.stem) for path in
                          (args.a1a2_root / "assets").glob("*.json")}
        guard = threading.Lock()

        def needs_a1a2(number: int) -> bool:
            # a group whose A1/A2 results all exist needs no worker at all;
            # re-running it would only spawn dozens of idle interpreters
            return number < len(groups) and any(
                index not in finished_a1a2 for index in groups[number])

        def inflight() -> int:
            with guard:
                return sum(1 for future in futures.values() if not future.done())

        def top_up() -> None:
            """Keep the CPU stage fed regardless of what the cards are doing.

            Tying the refill to "an A4 group just finished" starved the CPU for the
            rest of a 30+ minute group whenever the queued groups were mostly done.
            """

            while not stop.is_set():
                try:
                    if (inflight() < args.prefetch_groups
                            and free_gb(args.targets_root) >= args.min_free_gb):
                        for number in range(len(groups)):
                            if inflight() >= args.prefetch_groups:
                                break
                            with guard:
                                known = number in futures
                            if known or not needs_a1a2(number):
                                continue
                            future = producer.submit(run_a1a2, groups[number],
                                                     f"g{number:03d}")
                            future.add_done_callback(
                                lambda _f, group=groups[number]: finished_a1a2.update(group))
                            with guard:
                                futures[number] = future
                except Exception as error:  # noqa: BLE001 - the watcher must not die
                    print(json.dumps({"event": "top_up_error",
                                      "error": f"{type(error).__name__}: {error}"}),
                          flush=True)
                stop.wait(args.prefetch_interval)

        with ThreadPoolExecutor(max_workers=max(1, args.producer_workers)) as producer:
            def wait_group(number: int) -> None:
                while True:
                    with guard:
                        future = futures.get(number)
                    if future is None:
                        if not needs_a1a2(number):
                            return
                        stop.wait(5)
                        continue
                    future.result()
                    return

            watcher = threading.Thread(target=top_up, daemon=True)
            watcher.start()
            try:
                for number, group in enumerate(groups):
                    tag = f"g{number:03d}"
                    wait_group(number)
                    low = free_gb(args.targets_root) < args.min_free_gb
                    print(json.dumps({"event": "a4_start", "group": tag,
                                      "assets": len(group),
                                      "a1a2_inflight": sorted(futures)}), flush=True)
                    print(json.dumps(run_a4(group, tag)), flush=True)
                    if low:
                        print(json.dumps({"event": "stopped_low_disk", "group": tag,
                                          "free_gb": round(free_gb(args.targets_root), 1),
                                          "min_free_gb": args.min_free_gb}), flush=True)
                        return 1
            finally:
                stop.set()
                watcher.join(timeout=30)
    except Exception as error:  # noqa: BLE001 - the driver must report what broke
        print(json.dumps({"event": "failed", "error": f"{type(error).__name__}: {error}",
                          "traceback": traceback.format_exc()[-800:]}), flush=True)
        return 1
    refresh_summary()
    counts = summarize(assets, args.a1a2_root,
                       json.loads((args.a1a2_root / "plan.json").read_text()))["counts"]
    finished = sum(1 for asset in assets
                   if a4_present(args.targets_root, asset["id"], args.resolution))
    print(json.dumps({"event": "done", "a1a2": counts, "a4_present": finished}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())