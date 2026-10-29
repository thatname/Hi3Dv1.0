"""Read back the fixed 50-asset A4 targets, flags, and paired metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def check(root: Path, manifest_path: Path, a3_path: Path, a0_path: Path) -> dict:
    summary = json.loads((root / "summary.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    manifest = json.loads(manifest_path.read_text())["assets"]
    a3_rows = json.loads(a3_path.read_text())["assets"]
    a0_rows = json.loads(a0_path.read_text())["assets"]
    if (not all(len(x) == 50 for x in (manifest, a3_rows, a0_rows, summary["assets"]))
            or summary["counts"] != {"success": 50, "failed": 0, "pending": 0}
            or plan["a0_policy"] != "retain_flags_no_exclusion"
            or plan["steps"] != 200 or plan["image_resolution"] != 4096):
        raise ValueError("A4 batch inputs, plan, or counts differ")
    failures = []
    chamfer_changes = []
    held_changes = {name: [] for name in ("silhouette", "depth", "normal")}
    held_before = {name: [] for name in held_changes}
    held_after = {name: [] for name in held_changes}
    max_abs_delta = 0.0
    sign_change_assets = 0
    sign_change_points = 0
    inverted_crossing_assets = []
    total_seconds = 0.0
    peak_torch_bytes = 0
    for index, (asset, a3_row, flag, row) in enumerate(
            zip(manifest, a3_rows, a0_rows, summary["assets"])):
        if (row["index"] != index or row["id"] != asset["id"]
                or row["status"] != "success" or row["a0_status"] != flag["status"]
                or row["a0_concerns"] != flag.get("concerns", [])):
            failures.append(f"{index}: identity or A0 flag mismatch")
            continue
        cache = Path(row["cache"])
        if not cache.is_dir() or not all((cache / name).is_file() for name in (
            "target.npz", "raw_mesh.ply", "a3_classic_raw_mesh.ply",
            "resolved_config.json", "audit.json", "trace.json", "metrics.json")):
            failures.append(f"{index}: incomplete cache")
            continue
        config = json.loads((cache / "resolved_config.json").read_text())
        audit = json.loads((cache / "audit.json").read_text())
        trace = json.loads((cache / "trace.json").read_text())
        if (config["asset_id"] != asset["id"] or config["a0_status"] != flag["status"]
                or config["a3_cache"] != a3_row["cache"]
                or config["steps"] != 200 or config["image_resolution"] != 4096
                or config["resolution"] != 128 or not config["allow_sign_changes"]
                or config["freeze_boundary"] or len(trace) != 21
                or trace[-1]["step"] != 200
                or audit["render"]["render_backend"] != "cuda"
                or audit["render"]["held_out_views"] != 12):
            failures.append(f"{index}: configuration, trace, or render audit mismatch")
            continue
        with np.load(cache / "target.npz", allow_pickle=False) as fitted:
            coords = fitted["coords"]
            features = fitted["features"]
            cells = fitted["cells"]
            indices = fitted["cell_indices"]
        with np.load(Path(a3_row["cache"]) / "target.npz", allow_pickle=False) as initial:
            same_support = (np.array_equal(coords, initial["coords"])
                            and np.array_equal(cells, initial["cells"])
                            and np.array_equal(indices, initial["cell_indices"]))
            old = initial["features"]
        if (not same_support or features.shape != old.shape
                or not np.isfinite(features).all()
                or np.max(np.abs(features[:, 1:])) > 0.50001
                or audit["sign_changes_from_a3"] != np.count_nonzero(
                    (features[:, 0] < 0) != (old[:, 0] < 0))):
            failures.append(f"{index}: paired field/support invariant mismatch")
            continue
        max_abs_delta = max(max_abs_delta, float(np.max(np.abs(features[:, 1:]))))
        sign_change_assets += audit["sign_changes_from_a3"] > 0
        sign_change_points += audit["sign_changes_from_a3"]
        if audit["grid_quality"]["inverted_crossing_cells"]:
            inverted_crossing_assets.append((index, audit["grid_quality"]["inverted_crossing_cells"]))
        total_seconds += row["seconds"]
        peak_torch_bytes = max(peak_torch_bytes, audit["torch_peak_allocated_bytes"])
        before, after = row["metrics"]["A3_classic"], row["metrics"]["A4"]
        chamfer_changes.append((index, after["chamfer_l1"] - before["chamfer_l1"]))
        for name in held_changes:
            old_value = audit["render"]["before"][name]
            new_value = audit["render"]["after"][name]
            held_before[name].append(old_value)
            held_after[name].append(new_value)
            held_changes[name].append(new_value - old_value)
    if failures:
        raise ValueError("; ".join(failures))
    return {
        "checked": len(chamfer_changes),
        "a0_flag_counts": {status: sum(r["a0_status"] == status for r in summary["assets"])
                           for status in sorted({r["a0_status"] for r in summary["assets"]})},
        "max_abs_delta_voxels": max_abs_delta,
        "sign_change_assets": sign_change_assets,
        "sign_change_points_total": sign_change_points,
        "inverted_crossing_assets": inverted_crossing_assets,
        "inverted_crossing_cells_total": sum(n for _, n in inverted_crossing_assets),
        "sum_asset_seconds": total_seconds,
        "peak_torch_allocated_bytes": peak_torch_bytes,
        "groups": summary["groups"],
        "chamfer_change_mean": float(np.mean([x[1] for x in chamfer_changes])),
        "chamfer_change_median": float(np.median([x[1] for x in chamfer_changes])),
        "held_before_mean": {name: float(np.mean(values)) for name, values in held_before.items()},
        "held_after_mean": {name: float(np.mean(values)) for name, values in held_after.items()},
        "held_change_mean": {name: float(np.mean(values)) for name, values in held_changes.items()},
        "held_better_count": {name: sum(v < 0 for v in values) for name, values in held_changes.items()},
        "worst_chamfer_regressions": sorted(chamfer_changes, key=lambda x: x[1], reverse=True)[:5],
        "best_chamfer_improvements": sorted(chamfer_changes, key=lambda x: x[1])[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a3-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.manifest, args.a3_summary, args.a0_audit),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
