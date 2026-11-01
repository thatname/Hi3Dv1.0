"""Stratified loadability audit of the Objaverse raw mesh pool.

Walks the pool, records extension/size counts, then tries to load a per-extension
sample through the same loader the target pipeline uses (vae3d.data.load_mesh,
trimesh based) so the corpus' real failure modes and per-format cost are known
before any target generation is planned.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vae3d.data import load_mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/TRELLIS/datasets/ObjaverseXL_github/"
                                     "raw/github/objects"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-extension", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()

    started = time.perf_counter()
    by_extension: dict[str, list[Path]] = defaultdict(list)
    total_bytes = 0
    for path in args.root.iterdir():
        if not path.is_file():
            continue
        total_bytes += path.stat().st_size
        by_extension[path.suffix.lower() or "<none>"].append(path)

    summary = {
        "root": str(args.root),
        "files": sum(len(v) for v in by_extension.values()),
        "bytes": total_bytes,
        "by_extension": {k: len(v) for k, v in sorted(by_extension.items(),
                                                      key=lambda kv: -len(kv[1]))},
        "sample_per_extension": args.per_extension,
        "resolution": args.resolution,
        "extensions": {},
    }

    rng = random.Random(args.seed)
    for extension, paths in sorted(by_extension.items(), key=lambda kv: -len(kv[1])):
        sample = rng.sample(paths, min(args.per_extension, len(paths)))
        rows = []
        for path in sorted(sample):
            entry = {"file": path.name, "bytes": path.stat().st_size}
            try:
                begin = time.perf_counter()
                mesh, center, scale = load_mesh(
                    str(path), normalization_extent=(args.resolution - 2) / args.resolution)
                entry.update({
                    "status": "loaded",
                    "seconds": time.perf_counter() - begin,
                    "faces": int(len(mesh.faces)),
                    "vertices": int(len(mesh.vertices)),
                    "scale": float(scale),
                })
                extent = np.asarray(mesh.bounds[1]) - np.asarray(mesh.bounds[0])
                entry["normalized_extent"] = [round(float(v), 6) for v in extent]
            except Exception as error:  # noqa: BLE001 - audit wants every failure mode
                entry.update({"status": "failed",
                              "error": f"{type(error).__name__}: {error}"[:300],
                              "traceback": traceback.format_exc()[-400:]})
            rows.append(entry)
        loaded = [r for r in rows if r["status"] == "loaded"]
        summary["extensions"][extension] = {
            "pool": len(paths),
            "sampled": len(rows),
            "loaded": len(loaded),
            "load_rate": len(loaded) / len(rows) if rows else None,
            "median_seconds": (float(np.median([r["seconds"] for r in loaded]))
                               if loaded else None),
            "median_faces": (int(np.median([r["faces"] for r in loaded]))
                             if loaded else None),
            "rows": rows,
        }
        print(json.dumps({k: v for k, v in summary["extensions"][extension].items()
                          if k != "rows"}), flush=True)

    summary["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": summary["files"], "bytes": summary["bytes"],
                      "elapsed_seconds": summary["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())