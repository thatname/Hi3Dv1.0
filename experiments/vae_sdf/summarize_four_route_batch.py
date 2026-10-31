"""Audit four-route caches for a fixed manifest and summarize paired raw-mesh metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROUTES = ("mc_flat", "mc_smooth", "flexicubes_flat", "flexicubes_smooth")
COMMON_CONFIG = ("source_sha256", "manifest_sha256", "a2_target_sha256",
                 "support_sha256", "resolution", "sampling_offset",
                 "normalization_center", "normalization_scale", "a3_steps",
                 "a3_samples", "a4_steps", "image_resolution", "views",
                 "held_out_views", "seed", "metric_samples", "delta_limit",
                 "allow_sign_changes_a4", "target_thickness_pixels", "depth_weight")


def percentile(values, q):
    import numpy as np

    return float(np.percentile(values, q)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assets = json.loads(args.manifest.read_text())["assets"]
    status = json.loads((args.root / "batch-status.json").read_text())
    if not assets or status["identity"]["indices"] != list(range(len(assets))):
        raise ValueError("Batch status must cover every manifest asset in order")
    routes = tuple(status["identity"]["routes"])
    if (not routes or len(set(routes)) != len(routes)
            or any(route not in ROUTES for route in routes)):
        raise ValueError("Batch status lists unsupported or duplicate routes")
    rows = []
    for index, asset in enumerate(assets):
        row = {"index": index, "asset_id": asset["id"], "source_sha256": asset["sha256"],
               "routes": {}}
        reference_config = None
        for route in routes:
            run = status["runs"].get(f"{index}:{route}", {"status": "pending"})
            entry = {"status": run["status"], "log": run.get("log")}
            if run["status"] == "success":
                matches = list((args.root / asset["id"] / "R128").glob(f"{route}-*"))
                if len(matches) != 1:
                    raise ValueError(f"Expected one cache for {index}:{route}, found {len(matches)}")
                cache = matches[0]
                config = json.loads((cache / "resolved_config.json").read_text())
                if (config["index"] != index or config["route_id"] != route
                        or config["source_sha256"] != asset["sha256"]):
                    raise ValueError(f"Cache identity mismatch: {index}:{route}")
                common = {key: config[key] for key in COMMON_CONFIG}
                if reference_config is None:
                    reference_config = common
                elif common != reference_config:
                    raise ValueError(f"Four-route protocol differs at {index}:{route}")
                entry["cache"] = str(cache)
                entry["config_sha256"] = cache.name.rsplit("-", 1)[-1]
                for stage in ("a3", "a4"):
                    folder = cache / stage
                    if not all((folder / name).is_file() for name in
                               ("raw_mesh.ply", "target.npz", "audit.json", "metrics.json", "trace.json")):
                        raise ValueError(f"Incomplete {index}:{route}:{stage}")
                    entry[stage] = {"audit": json.loads((folder / "audit.json").read_text()),
                                    "metrics": json.loads((folder / "metrics.json").read_text())}
            row["routes"][route] = entry
        rows.append(row)
    paired = [row for row in rows if all(row["routes"][route]["status"] == "success"
                                          for route in routes)]
    output = {"schema": "g0_four_route_paired_summary_v1", "root": str(args.root),
              "batch_finished": status.get("finished_at") is not None,
              "total_assets": len(assets), "paired_assets": len(paired),
              "paired_indices": [row["index"] for row in paired],
              "routes_requested": list(routes), "routes": {}, "assets": rows,
              "normal_mode_comparison_note": "Training/held-out normal losses use each route's normal mode; do not rank them across modes without a common evaluator.",
              "self_intersection": "not_measured", "peak_driver_memory": "not_measured"}
    for route in routes:
        successful = [row for row in rows if row["routes"][route]["status"] == "success"]
        def vals(key, stage="a4", population=paired):
            return [float(row["routes"][route][stage]["metrics"][key]) for row in population]
        a4 = [row["routes"][route]["a4"] for row in paired]
        output["routes"][route] = {
            "success": len(successful),
            "failed": sum(row["routes"][route]["status"] == "failed" for row in rows),
            "pending_or_running": sum(row["routes"][route]["status"] not in {"success", "failed"}
                                      for row in rows),
            "chamfer_l1_median": percentile(vals("chamfer_l1"), 50),
            "chamfer_l1_p90": percentile(vals("chamfer_l1"), 90),
            "normal_consistency_median": percentile(vals("normal_consistency"), 50),
            "fscore_0_005_median": percentile([
                x["metrics"]["fscore"]["0.005"]["fscore"] for x in a4], 50),
            "fscore_0_01_median": percentile([
                x["metrics"]["fscore"]["0.01"]["fscore"] for x in a4], 50),
            "boundary_edges_median": percentile(vals("boundary_edges"), 50),
            "nonmanifold_edges_median": percentile(vals("nonmanifold_edges"), 50),
            "assets_with_inverted_crossing_cells": sum(
                x["audit"]["grid_quality"]["inverted_crossing_cells"] > 0 for x in a4),
            "a4_better_chamfer_than_a3": sum(
                row["routes"][route]["a4"]["metrics"]["chamfer_l1"] <
                row["routes"][route]["a3"]["metrics"]["chamfer_l1"] for row in paired),
            "a4_seconds_median": percentile([
                x["audit"]["seconds"] for x in a4], 50),
            "a4_peak_allocated_bytes_max": max((x["audit"]["peak_cuda_allocated_bytes"] for x in a4), default=None),
            "a4_peak_reserved_bytes_max": max((x["audit"].get("peak_cuda_reserved_bytes", 0) for x in a4), default=None),
        }
    if args.output:
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
    else:
        print(json.dumps({key: value for key, value in output.items() if key != "assets"},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
