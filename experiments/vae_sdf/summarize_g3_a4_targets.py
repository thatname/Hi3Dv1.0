"""Write the A4-only paired summary for the G3 pool.

`summarize_four_route_batch` insists on a completed a3 + a4 pair, but the pool
driver prunes every a3 as soon as its a4 succeeds. The training and evaluation
entry points only need the A4 entry (cache, audit, metrics) plus a provenance
check that every cache belongs to the manifest asset at that index, so this
writer produces the same schema with the a3 stage absent and its comparison
fields omitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .g1_targets import ROUTE
from .preflight import sha256

COMMON_CONFIG = ("resolution", "sampling_offset", "route_id", "extractor", "normal_mode",
                 "delta_limit", "target_thickness_pixels", "depth_weight", "a4_steps")


def percentile(values: list[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def build(manifest_path: Path, targets_root: Path, resolution: int, route: str) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assets = manifest["assets"]
    rows = []
    reference = None
    for index, asset in enumerate(assets):
        matches = list((targets_root / asset["id"] / f"R{resolution}").glob(f"{route}-*"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {route} cache for asset {index}, "
                             f"found {len(matches)}")
        cache = matches[0]
        config = json.loads((cache / "resolved_config.json").read_text(encoding="utf-8"))
        if (config["index"] != index or config["route_id"] != route
                or config["source_sha256"] != asset["sha256"]):
            raise ValueError(f"Cache identity mismatch at {index}: {cache}")
        common = {key: config.get(key) for key in COMMON_CONFIG}
        if reference is None:
            reference = common
        elif common != reference:
            raise ValueError(f"Protocol differs at {index}: {common} vs {reference}")
        stage = cache / "a4"
        for name in ("target.npz", "audit.json", "metrics.json"):
            if not (stage / name).is_file():
                raise ValueError(f"Incomplete a4 at {index}: missing {name}")
        rows.append({"index": index, "asset_id": asset["id"], "source_sha256": asset["sha256"],
                     "routes": {route: {
                         "status": "success", "cache": str(cache),
                         "config_sha256": cache.name.rsplit("-", 1)[-1],
                         "a4": {"audit": json.loads((stage / "audit.json").read_text(
                             encoding="utf-8")),
                                "metrics": json.loads((stage / "metrics.json").read_text(
                                    encoding="utf-8"))}}}})

    audit = [row["routes"][route]["a4"] for row in rows]
    output = {
        "schema": "g0_four_route_paired_summary_v1",
        "root": str(targets_root),
        "resolution": resolution,
        "batch_finished": True,
        "total_assets": len(assets),
        "paired_assets": len(rows),
        "paired_indices": [row["index"] for row in rows],
        "routes_requested": [route],
        "routes": {route: {
            "success": len(rows),
            "failed": 0,
            "pending_or_running": 0,
            "chamfer_l1_median": percentile(
                [x["metrics"]["chamfer_l1"] for x in audit], 50),
            "chamfer_l1_p90": percentile([x["metrics"]["chamfer_l1"] for x in audit], 90),
            "normal_consistency_median": percentile(
                [x["metrics"]["normal_consistency"] for x in audit], 50),
            "fscore_0_005_median": percentile(
                [x["metrics"]["fscore"]["0.005"]["fscore"] for x in audit], 50),
            "fscore_0_01_median": percentile(
                [x["metrics"]["fscore"]["0.01"]["fscore"] for x in audit], 50),
            "boundary_edges_median": percentile(
                [x["metrics"]["boundary_edges"] for x in audit], 50),
            "nonmanifold_edges_median": percentile(
                [x["metrics"]["nonmanifold_edges"] for x in audit], 50),
            "assets_with_inverted_crossing_cells": sum(
                x["audit"]["grid_quality"]["inverted_crossing_cells"] > 0 for x in audit),
            "a4_seconds_median": percentile([x["audit"]["seconds"] for x in audit], 50),
            "a4_peak_allocated_bytes_max": max(
                (x["audit"]["peak_cuda_allocated_bytes"] for x in audit), default=None),
        }},
        "assets": rows,
        "a3_note": "a3 is pruned as soon as a4 succeeds for the pool run, so no a3 "
                   "comparison fields exist in this summary",
        "manifest": manifest_path.name,
        "manifest_sha256": sha256(manifest_path),
        "self_intersection": "not_measured",
        "peak_driver_memory": "not_measured",
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--route", default=ROUTE)
    args = parser.parse_args()
    if args.output.resolve().parent != args.targets_root.resolve():
        # the training/evaluation loader derives the cache root from the summary's
        # own directory, so a summary written elsewhere would be rejected there
        parser.error("--output must be written inside --targets-root")

    summary = build(args.manifest, args.targets_root, args.resolution, args.route)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("assets", "routes")}, ensure_ascii=False))
    print(json.dumps(summary["routes"][args.route], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())