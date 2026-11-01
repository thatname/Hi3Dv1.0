"""Compact G3 A4 caches so the full pool fits on the 2 TB volume.

Two lossless reductions: the eight-wide `cell_indices` array is dropped (it is a
third of every target and is rebuilt exactly from `cells` and `coords` on load),
and `raw_mesh.ply` is dropped for training-split assets, which never need a mesh
for metrics. Held-out meshes are kept so evaluation can still be run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def compact_one(target: Path, drop_mesh: Path | None) -> dict:
    freed = 0
    steps = []
    started = time.perf_counter()
    with np.load(target, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if "cell_indices" in arrays:
        reduced = {name: value for name, value in arrays.items() if name != "cell_indices"}
        temporary = target.with_name("target.compact.npz")
        np.savez_compressed(temporary, **reduced)
        before = target.stat().st_size
        freed += before - temporary.stat().st_size
        os.replace(temporary, target)
        steps.append("cell_indices")
    if drop_mesh is not None and drop_mesh.is_file():
        freed += drop_mesh.stat().st_size
        drop_mesh.unlink()
        steps.append("mesh")
    return {"freed": freed, "steps": steps, "seconds": time.perf_counter() - started}


def sweep(targets: Path, resolution: int, keep_ids: set[str]) -> dict:
    compacted = meshes = saved = 0
    touched = 0
    for asset in sorted(targets.iterdir()):
        if not asset.is_dir():
            continue
        for cache in (asset / f"R{resolution}").glob("flexicubes_smooth-*/a4"):
            target = cache / "target.npz"
            # a finished stage is renamed into place, so audit.json implies the
            # npz is complete and safe to rewrite
            if not target.is_file() or not (cache / "audit.json").is_file():
                continue
            drop_mesh = None if asset.name in keep_ids else cache / "raw_mesh.ply"
            with np.load(target, allow_pickle=False) as probe:
                has_indices = "cell_indices" in probe.files
            if not has_indices and (drop_mesh is None or not drop_mesh.is_file()):
                continue
            row = compact_one(target, drop_mesh)
            compacted += int(has_indices)
            meshes += int("mesh" in row["steps"])
            saved += row["freed"]
            touched += 1
    return {"assets_touched": touched, "cell_indices_dropped": compacted,
            "meshes_removed": meshes, "bytes_freed": saved,
            "gib_freed": round(saved / 2 ** 30, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True,
                        help="split whose val/test assets keep their mesh")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--watch", action="store_true",
                        help="keep sweeping while targets are still being produced")
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--idle-rounds", type=int, default=10,
                        help="stop watching after this many rounds with nothing to do")
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    keep_ids = set(split["val_ids"]) | set(split["test_ids"])
    idle = 0
    while True:
        row = sweep(args.targets_root, args.resolution, keep_ids)
        print(json.dumps({"event": "sweep", **row}), flush=True)
        if not args.watch:
            return 0
        idle = idle + 1 if row["assets_touched"] == 0 else 0
        if idle >= args.idle_rounds:
            print(json.dumps({"event": "idle_exit", "rounds": idle}), flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())