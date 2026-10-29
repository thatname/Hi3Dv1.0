"""Derive scale-aware, paired A1/A2 statistics from a completed G0 batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def stats(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def paired(values_a1: list[float], values_a2: list[float], *, lower_is_better: bool) -> dict:
    difference = [b - a for a, b in zip(values_a1, values_a2)]
    tol = 1e-12
    return {"a1": stats(values_a1), "a2": stats(values_a2),
            "a2_minus_a1": stats(difference),
            "a2_better": sum(d < -tol if lower_is_better else d > tol for d in difference),
            "a2_worse": sum(d > tol if lower_is_better else d < -tol for d in difference),
            "tied": sum(abs(d) <= tol for d in difference)}


def analyze(summary_path: Path) -> dict:
    data = summary_path.read_bytes()
    summary = json.loads(data)
    rows = summary["assets"]
    if (summary["denominator"] != 50 or len(rows) != 50
            or summary["counts"] != {"success": 50, "failed": 0, "pending": 0}
            or any(row["index"] != index or row["status"] != "success"
                   for index, row in enumerate(rows))):
        raise ValueError("This analysis requires a complete, ordered 50/50 batch")
    result = {"schema": "g0_a1a2_paired_analysis_v1",
              "summary": str(summary_path), "summary_sha256": hashlib.sha256(data).hexdigest(),
              "groups": {}, "assets": []}
    for row in rows:
        one = {"index": row["index"], "id": row["id"],
               "group": "closed_candidate" if row["index"] < 20 else "nonclosed",
               "seconds": row["seconds"], "stages": {}}
        for stage in ("A1", "A2"):
            metric = row["metrics"]["stages"][stage]
            one["stages"][stage] = {
                "chamfer_over_source_bbox_diagonal":
                    metric["chamfer_l1"] / metric["reference_bbox_diagonal"],
                "fscore_0.005": metric["fscore"]["0.005"]["fscore"],
                "fscore_0.01": metric["fscore"]["0.01"]["fscore"],
                "normal_consistency": metric["normal_consistency"],
                "boundary_edges": metric["boundary_edges"],
                "nonmanifold_edges": metric["nonmanifold_edges"],
            }
        one["added_shell_samples"] = row["metrics"]["cache_check"]["added_negative_samples"]
        one["changed_field_samples"] = row["metrics"]["cache_check"]["field_changed_samples"]
        result["assets"].append(one)
    for name, group in (("all", result["assets"]),
                        ("closed_candidate", result["assets"][:20]),
                        ("nonclosed", result["assets"][20:])):
        section = {"denominator": len(group), "seconds": stats([r["seconds"] for r in group]),
                   "total_seconds": sum(r["seconds"] for r in group),
                   "shell_added_assets": sum(r["added_shell_samples"] > 0 for r in group),
                   "field_changed_assets": sum(r["changed_field_samples"] > 0 for r in group)}
        for metric in ("chamfer_over_source_bbox_diagonal", "fscore_0.005",
                       "fscore_0.01", "normal_consistency"):
            a1 = [r["stages"]["A1"][metric] for r in group]
            a2 = [r["stages"]["A2"][metric] for r in group]
            section[metric] = paired(a1, a2, lower_is_better=metric.startswith("chamfer"))
        for stage in ("A1", "A2"):
            section[f"{stage}_meshes_with_boundary_edges"] = sum(
                r["stages"][stage]["boundary_edges"] > 0 for r in group)
            section[f"{stage}_meshes_with_nonmanifold_edges"] = sum(
                r["stages"][stage]["nonmanifold_edges"] > 0 for r in group)
        result["groups"][name] = section
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.summary)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary_sha256": result["summary_sha256"],
                      "count": len(result["assets"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
