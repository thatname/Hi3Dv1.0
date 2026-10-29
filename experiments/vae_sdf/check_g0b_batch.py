"""Read back G0b outputs and audit packing, extractor, and flag invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def check(root: Path, manifest_path: Path, a4_path: Path) -> dict:
    summary = json.loads((root / "summary.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    assets = json.loads(manifest_path.read_text())["assets"]
    a4_rows = json.loads(a4_path.read_text())["assets"]
    if (not all(len(x) == 50 for x in (assets, a4_rows, summary["assets"]))
            or summary["counts"] != {"success": 50, "failed": 0, "pending": 0}
            or plan["a0_policy"] != "retain_flags_no_exclusion"
            or plan["quality_gate"] != "deferred_by_user"):
        raise ValueError("G0b fixed inputs, plan, or counts differ")
    failures = []
    errors = {name: [] for name in (
        "field_error", "delta_error", "shared_point_error",
        "coordinate_error", "same_mc_vertex_error")}
    chamfer_changes = []
    fscore_changes = []
    normal_changes = []
    face_changes = []
    boundary_changes = []
    nonmanifold_changes = []
    watertight_pairs = []
    inverted = []
    total_seconds = 0.0
    for index, (asset, a4, row) in enumerate(zip(assets, a4_rows, summary["assets"])):
        if (row["index"] != index or row["id"] != asset["id"]
                or row["status"] != "success"
                or row["source_sha256"] != asset["sha256"]
                or row["a0_status"] != a4["a0_status"]
                or row["a0_concerns"] != a4["a0_concerns"]):
            failures.append(f"{index}: identity or A0 flag mismatch")
            continue
        cache = Path(row["cache"])
        if not cache.is_dir() or not all((cache / name).is_file() for name in (
            "mc_raw_mesh.ply", "flexicubes_raw_mesh.ply",
            "resolved_config.json", "audit.json", "metrics.json")):
            failures.append(f"{index}: incomplete cache")
            continue
        config = json.loads((cache / "resolved_config.json").read_text())
        audit = json.loads((cache / "audit.json").read_text())
        metrics = json.loads((cache / "metrics.json").read_text())
        if (config["schema"] != "g0b_field_interface_v1"
                or config["asset_id"] != asset["id"]
                or config["a4_cache"] != a4["cache"]
                or config["resolution"] != 128
                or config["sampling_offset"] != 0.5
                or config["corner_order_hi3d_from_vae"] != [0, 1, 3, 2, 4, 5, 7, 6]
                or config["point_deform_raw_scale"] != 2.0
                or config["a0_status"] != a4["a0_status"]
                or audit != row["audit"] or metrics != row["metrics"]
                or audit["a4_inverted_crossing_cells"]
                != a4["audit"]["grid_quality"]["inverted_crossing_cells"]
                or audit["raw_mesh_coordinate_frame"] != "source"
                or audit["raw_mesh_postprocessing"] != "none"
                or audit["center_points"] != audit["hi3d_shared_vertices"]
                or audit["cells"] != audit["hi3d_shared_cubes"]
                or not audit["same_mc_faces_exact"]):
            failures.append(f"{index}: config or extractor audit mismatch")
            continue
        if any(audit[name] > 1e-6 for name in errors):
            failures.append(f"{index}: lossless packing tolerance exceeded")
            continue
        for name in errors:
            errors[name].append(audit[name])
        mc, flex = metrics["MC_vs_source"], metrics["FlexiCubes_vs_source"]
        if (mc["chamfer_l1"] is None or flex["chamfer_l1"] is None
                or mc["faces"] != audit["mc_raw_faces"]
                or flex["faces"] != audit["flexicubes_raw_faces"]):
            failures.append(f"{index}: empty or inconsistent extracted mesh")
            continue
        chamfer_changes.append((index, flex["chamfer_l1"] - mc["chamfer_l1"]))
        normal_changes.append((index, flex["normal_consistency"] - mc["normal_consistency"]))
        fscore_changes.append((index, flex["fscore"]["0.005"]["fscore"]
                               - mc["fscore"]["0.005"]["fscore"]))
        face_changes.append((index, flex["faces"] - mc["faces"]))
        boundary_changes.append((index, flex["boundary_edges"] - mc["boundary_edges"]))
        nonmanifold_changes.append((index, flex["nonmanifold_edges"] - mc["nonmanifold_edges"]))
        watertight_pairs.append((index, mc["watertight"], flex["watertight"]))
        if audit["a4_inverted_crossing_cells"]:
            inverted.append((index, audit["a4_inverted_crossing_cells"]))
        total_seconds += row["seconds"]
    if failures:
        raise ValueError("; ".join(failures))
    return {
        "checked": len(chamfer_changes),
        "a0_flag_counts": {status: sum(r["a0_status"] == status for r in summary["assets"])
                           for status in sorted({r["a0_status"] for r in summary["assets"]})},
        "max_errors": {name: max(values) for name, values in errors.items()},
        "inverted_crossing_assets": inverted,
        "inverted_crossing_cells_total": sum(n for _, n in inverted),
        "sum_asset_seconds": total_seconds,
        "groups": summary["groups"],
        "flexicubes_better_chamfer": sum(d < 0 for _, d in chamfer_changes),
        "flexicubes_worse_chamfer": sum(d > 0 for _, d in chamfer_changes),
        "chamfer_change_mean": float(np.mean([d for _, d in chamfer_changes])),
        "chamfer_change_median": float(np.median([d for _, d in chamfer_changes])),
        "worst_chamfer_regressions": sorted(chamfer_changes, key=lambda x: x[1], reverse=True)[:8],
        "best_chamfer_improvements": sorted(chamfer_changes, key=lambda x: x[1])[:8],
        "fscore_0.005_improved": sum(d > 0 for _, d in fscore_changes),
        "fscore_0.005_regressed": sum(d < 0 for _, d in fscore_changes),
        "normal_consistency_improved": sum(d > 0 for _, d in normal_changes),
        "normal_consistency_regressed": sum(d < 0 for _, d in normal_changes),
        "face_change_range": [min(d for _, d in face_changes), max(d for _, d in face_changes)],
        "boundary_edges_increased": [i for i, d in boundary_changes if d > 0],
        "nonmanifold_edges_increased": [i for i, d in nonmanifold_changes if d > 0],
        "mc_watertight": sum(a for _, a, _ in watertight_pairs),
        "flexicubes_watertight": sum(b for _, _, b in watertight_pairs),
        "watertight_lost": [i for i, a, b in watertight_pairs if a and not b],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a4-summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.manifest, args.a4_summary),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
