"""Read back every fixed G0 A3 cache and summarize the paired A2/A3 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def check(root: Path, manifest_path: Path, a1a2_path: Path, a0_path: Path) -> dict:
    summary = json.loads((root / "summary.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    manifest = json.loads(manifest_path.read_text())["assets"]
    a1a2 = json.loads(a1a2_path.read_text())["assets"]
    a0 = json.loads(a0_path.read_text())["assets"]
    if (len(manifest) != len(a1a2) or len(manifest) != len(a0)
            or len(manifest) != len(summary["assets"]) or len(manifest) != 50
            or summary["counts"] != {"success": 50, "failed": 0, "pending": 0}
            or plan["a0_policy"] != "retain_flags_no_exclusion"):
        raise ValueError("Fixed 50-asset inputs or A3 counts differ")
    deltas, normal_deltas, fscore_deltas = [], [], []
    failures = []
    max_delta = 0.0
    total_seconds = 0.0
    peak_bytes = 0
    for index, (asset, initial_row, flag, row) in enumerate(
            zip(manifest, a1a2, a0, summary["assets"])):
        if (row["index"] != index or row["id"] != asset["id"]
                or row["status"] != "success" or row["a0_status"] != flag["status"]
                or row["a0_concerns"] != flag.get("concerns", [])):
            failures.append(f"{index}: identity or A0 flag mismatch")
            continue
        cache = Path(row["cache"])
        if not cache.is_dir() or not all((cache / name).is_file() for name in (
            "target.npz", "raw_mesh.ply", "a2_classic_raw_mesh.ply",
            "resolved_config.json", "audit.json", "trace.json", "metrics.json")):
            failures.append(f"{index}: incomplete cache")
            continue
        config = json.loads((cache / "resolved_config.json").read_text())
        audit = json.loads((cache / "audit.json").read_text())
        if (config["asset_id"] != asset["id"] or config["a0_status"] != flag["status"]
                or config["steps"] != 30 or config["fit_samples"] != 8192
                or config["resolution"] != 128 or audit["sign_changes"] != 0
                or audit["grid_quality"]["inverted_cells"] != 0
                or audit["grid_quality"]["inverted_crossing_cells"] != 0):
            failures.append(f"{index}: config or grid audit mismatch")
            continue
        with np.load(cache / "target.npz", allow_pickle=False) as fitted:
            coord = fitted["coords"]
            field = fitted["features"]
            cells = fitted["cells"]
            cell_indices = fitted["cell_indices"]
        with np.load(Path(initial_row["caches"]["A2"]) / "target.npz",
                     allow_pickle=False) as initial:
            same_support = (np.array_equal(coord, initial["coords"])
                            and np.array_equal(cells, initial["cells"])
                            and np.array_equal(cell_indices, initial["cell_indices"]))
            original_field = initial["features"]
        if (not same_support or field.shape != original_field.shape
                or not np.isfinite(field).all()
                or not np.array_equal(field[:, 0] < 0, original_field[:, 0] < 0)
                or not np.array_equal(field[:, 0] == 0, original_field[:, 0] == 0)
                or np.max(np.abs(field[:, 1:])) > 0.50001):
            failures.append(f"{index}: paired field/support invariant mismatch")
            continue
        max_delta = max(max_delta, float(np.max(np.abs(field[:, 1:]))))
        total_seconds += row["seconds"]
        peak_bytes = max(peak_bytes, audit["torch_peak_allocated_bytes"])
        before, after = row["metrics"]["A2_classic"], row["metrics"]["A3"]
        deltas.append((index, after["chamfer_l1"] - before["chamfer_l1"]))
        normal_deltas.append(after["normal_consistency"] - before["normal_consistency"])
        fscore_deltas.append(after["fscore"]["0.005"]["fscore"]
                             - before["fscore"]["0.005"]["fscore"])
    if failures:
        raise ValueError("; ".join(failures))
    group = summary["groups"]
    return {
        "checked": len(deltas), "a0_flag_counts": {
            status: sum(row["a0_status"] == status for row in summary["assets"])
            for status in sorted({row["a0_status"] for row in summary["assets"]})},
        "max_abs_delta_voxels": max_delta,
        "sum_asset_seconds": total_seconds,
        "peak_torch_allocated_bytes": peak_bytes,
        "groups": group,
        "chamfer_change_mean": float(np.mean([x[1] for x in deltas])),
        "chamfer_change_median": float(np.median([x[1] for x in deltas])),
        "normal_change_mean": float(np.mean(normal_deltas)),
        "fscore_0.005_change_mean": float(np.mean(fscore_deltas)),
        "worst_chamfer_regressions": sorted(deltas, key=lambda x: x[1], reverse=True)[:5],
        "best_chamfer_improvements": sorted(deltas, key=lambda x: x[1])[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.manifest, args.a1a2_summary, args.a0_audit),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
