"""Run/resume G0b packing and extractor comparison on the fixed 50 A4 caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

from .preflight import sha256
from .run_initial_batch import save_json


SCHEMA = "g0b_field_interface_batch_v1"


def load_plan(args):
    manifest_bytes = args.manifest.read_bytes()
    a4_bytes = args.a4_summary.read_bytes()
    assets = json.loads(manifest_bytes)["assets"]
    a4_rows = json.loads(a4_bytes)["assets"]
    if len(assets) != 50 or len(a4_rows) != 50:
        raise ValueError("Expected fixed 50-asset inputs")
    for index, (asset, row) in enumerate(zip(assets, a4_rows)):
        if (row["index"] != index or row["id"] != asset["id"]
                or row["status"] != "success"
                or row["source_sha256"] != asset["sha256"]):
            raise ValueError(f"A4/manifest mismatch at {index}")
    module_dir = Path(__file__).parent
    plan = {
        "schema": SCHEMA,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "a4_summary": str(args.a4_summary),
        "a4_summary_sha256": hashlib.sha256(a4_bytes).hexdigest(),
        "vae3d_root": str(args.vae3d_root),
        "hi3d_root": str(args.hi3d_root),
        "metric_samples": args.metric_samples,
        "seed": args.seed,
        "device": args.device,
        "a0_policy": "retain_flags_no_exclusion",
        "quality_gate": "deferred_by_user",
        "code_sha256": {name: sha256(module_dir / name) for name in (
            "g0b_field_adapter.py", "run_g0b_asset.py", "run_g0b_batch.py",
            "evaluate_initial_targets.py")},
        "asset_count": 50,
    }
    return assets, a4_rows, plan


def summarize(assets, root, plan):
    import numpy as np

    rows = []
    for index, asset in enumerate(assets):
        path = root / "assets" / f"{index:02d}.json"
        row = json.loads(path.read_text()) if path.is_file() else {
            "index": index, "id": asset["id"], "status": "pending"}
        if row["index"] != index or row["id"] != asset["id"]:
            raise ValueError(f"Result identity mismatch at {index}")
        rows.append(row)
    counts = {name: sum(r["status"] == name for r in rows)
              for name in ("success", "failed", "pending")}
    groups = {}
    for label, selection in (("all", rows), ("closed_candidate", rows[:20]),
                             ("nonclosed", rows[20:]),
                             ("a0_needs_review", [r for r in rows if r.get("a0_status") == "needs_review"])):
        good = [r for r in selection if r["status"] == "success"]
        group = {"denominator": len(selection), "success": len(good),
                 "failed": sum(r["status"] == "failed" for r in selection),
                 "pending": sum(r["status"] == "pending" for r in selection),
                 "inverted_crossing_assets": sum(
                     r["audit"]["a4_inverted_crossing_cells"] > 0 for r in good)}
        for name in ("MC_vs_source", "FlexiCubes_vs_source", "FlexiCubes_vs_MC"):
            values = [r["metrics"][name]["chamfer_l1"] for r in good]
            values = [v for v in values if v is not None]
            group[name + "_chamfer"] = (
                {"mean": float(np.mean(values)), "median": float(np.median(values)),
                 "count": len(values)} if values else None)
        comparable = [r for r in good if all(
            r["metrics"][name]["chamfer_l1"] is not None
            for name in ("MC_vs_source", "FlexiCubes_vs_source"))]
        group["flexicubes_better_chamfer"] = sum(
            r["metrics"]["FlexiCubes_vs_source"]["chamfer_l1"]
            < r["metrics"]["MC_vs_source"]["chamfer_l1"] for r in comparable)
        group["comparable_chamfer"] = len(comparable)
        groups[label] = group
    return {"schema": SCHEMA,
            "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
            "denominator": 50, "counts": counts, "groups": groups, "assets": rows}


def run(args):
    from .run_g0b_asset import run as run_one

    assets, a4_rows, plan = load_plan(args)
    plan_path = args.output_root / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing G0b plan differs; use another output root")
    else:
        save_json(plan_path, plan)
    for index, (asset, a4) in enumerate(zip(assets, a4_rows)):
        if args.max_assets is not None and index >= args.max_assets:
            break
        result_path = args.output_root / "assets" / f"{index:02d}.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous["status"] == "success" or not args.retry_failed:
                continue
        started = time.perf_counter()
        row = {"index": index, "id": asset["id"], "source": asset["path"],
               "source_sha256": asset["sha256"], "a0_status": a4["a0_status"],
               "a0_concerns": a4["a0_concerns"], "status": "failed"}
        try:
            one = argparse.Namespace(
                manifest=args.manifest, a4_summary=args.a4_summary,
                vae3d_root=args.vae3d_root, hi3d_root=args.hi3d_root,
                index=index, output_root=args.output_root / "targets",
                metric_samples=args.metric_samples, seed=args.seed, device=args.device)
            result = run_one(one)
            row.update(cache=result["cache"], audit=result["audit"],
                       metrics=result["metrics"], status="success")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        row["seconds"] = time.perf_counter() - started
        save_json(result_path, row)
        save_json(args.output_root / "summary.json", summarize(assets, args.output_root, plan))
        print(f"{index + 1}/50 {row['status']} {asset['id']} {row['seconds']:.1f}s"
              + (f" {row['error']}" if row["status"] == "failed" else ""), flush=True)
    result = summarize(assets, args.output_root, plan)
    save_json(args.output_root / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a4-summary", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--hi3d-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metric-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-assets", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["counts"]), flush=True)
    return 0 if result["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
