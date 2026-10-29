"""Run and resume fixed 50-asset G0 A3 geometry fitting on one CUDA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

from .preflight import VAE3D_COMMIT, sha256
from .run_initial_batch import save_json


SCHEMA = "g0_a3_batch_v1"


def load_plan(args):
    manifest_bytes = args.manifest.read_bytes()
    summary_bytes = args.a1a2_summary.read_bytes()
    audit_bytes = args.a0_audit.read_bytes()
    assets = json.loads(manifest_bytes)["assets"]
    initial = json.loads(summary_bytes)["assets"]
    audit = json.loads(audit_bytes)["assets"]
    if not (len(assets) == len(initial) == len(audit) == 50):
        raise ValueError("Expected 50 fixed assets and 50 A1/A2 and A0 rows")
    for index, (asset, row, flag) in enumerate(zip(assets, initial, audit)):
        if (row["index"] != index or row["id"] != asset["id"]
                or row["status"] != "success" or flag["index"] != index
                or flag["id"] != asset["id"]):
            raise ValueError(f"Input mismatch at index {index}")
    head = subprocess.check_output(
        ["git", "-C", str(args.vae3d_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != VAE3D_COMMIT:
        raise ValueError(f"Wrong vae3d commit: {head}")
    module_dir = Path(__file__).parent
    plan = {
        "schema": SCHEMA, "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "a1a2_summary": str(args.a1a2_summary),
        "a1a2_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "a0_audit": str(args.a0_audit),
        "a0_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "vae3d_root": str(args.vae3d_root), "vae3d_commit": head,
        "resolution": args.resolution, "steps": args.steps,
        "fit_samples": args.fit_samples, "metric_samples": args.metric_samples,
        "seed": args.seed, "device": args.device,
        "extractor": "vae3d classic MC, same for A2 and A3",
        "a0_policy": "retain_flags_no_exclusion",
        "code_sha256": {name: sha256(module_dir / name) for name in (
            "prepare_a3_targets.py", "run_a3_batch.py", "evaluate_initial_targets.py")},
        "asset_count": len(assets),
    }
    return assets, audit, plan


def summarize(assets, audit, root, plan):
    import numpy as np

    rows = []
    for index, (asset, flag) in enumerate(zip(assets, audit)):
        result = root / "assets" / f"{index:02d}.json"
        if result.is_file():
            row = json.loads(result.read_text())
            if row["index"] != index or row["id"] != asset["id"]:
                raise ValueError(f"Result mismatch at index {index}")
        else:
            row = {"index": index, "id": asset["id"], "status": "pending",
                   "a0_status": flag["status"], "a0_concerns": flag.get("concerns", [])}
        rows.append(row)
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("success", "failed", "pending")}
    groups = {}
    for label, selection in (("all", rows), ("closed_candidate", rows[:20]),
                             ("nonclosed", rows[20:]),
                             ("a0_needs_review", [r for r in rows if r["a0_status"] == "needs_review"])):
        good = [r for r in selection if r["status"] == "success"]
        group = {"denominator": len(selection),
                 "success": len(good),
                 "failed": sum(r["status"] == "failed" for r in selection),
                 "pending": sum(r["status"] == "pending" for r in selection)}
        for stage in ("A2_classic", "A3"):
            group[stage] = {}
            for name, access in (
                ("chamfer_l1", lambda m: m["chamfer_l1"]),
                ("normal_consistency", lambda m: m["normal_consistency"]),
                ("fscore_0.005", lambda m: m["fscore"]["0.005"]["fscore"]),
                ("fscore_0.01", lambda m: m["fscore"]["0.01"]["fscore"]),
            ):
                values = [access(r["metrics"][stage]) for r in good]
                values = [v for v in values if v is not None]
                group[stage][name] = ({"mean": float(np.mean(values)),
                                      "median": float(np.median(values)),
                                      "count": len(values)} if values else None)
        comparable = [r for r in good if all(
            r["metrics"][s]["chamfer_l1"] is not None for s in ("A2_classic", "A3"))]
        group["A3_better_chamfer"] = sum(
            r["metrics"]["A3"]["chamfer_l1"] < r["metrics"]["A2_classic"]["chamfer_l1"]
            for r in comparable)
        group["comparable_chamfer"] = len(comparable)
        groups[label] = group
    return {"schema": SCHEMA, "plan_sha256": hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "denominator": len(rows), "counts": counts, "groups": groups, "assets": rows}


def run(args):
    from .prepare_a3_targets import run as prepare_one

    assets, audit, plan = load_plan(args)
    plan_path = args.output_root / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing A3 plan differs; use a new output root")
    else:
        save_json(plan_path, plan)
    for index, (asset, flag) in enumerate(zip(assets, audit)):
        if args.max_assets is not None and index >= args.max_assets:
            break
        result_path = args.output_root / "assets" / f"{index:02d}.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous["status"] == "success" or not args.retry_failed:
                continue
        started = time.perf_counter()
        row = {"index": index, "id": asset["id"], "source": asset["path"],
               "source_sha256": asset["sha256"], "a0_status": flag["status"],
               "a0_concerns": flag.get("concerns", []), "status": "failed"}
        try:
            one = argparse.Namespace(
                manifest=args.manifest, a1a2_summary=args.a1a2_summary,
                a0_audit=args.a0_audit, vae3d_root=args.vae3d_root,
                index=index, output_root=args.output_root / "targets",
                resolution=args.resolution, steps=args.steps,
                fit_samples=args.fit_samples, metric_samples=args.metric_samples,
                seed=args.seed, device=args.device)
            result = prepare_one(one)
            row.update(cache=result["cache"], audit=result["audit"],
                       metrics=result["metrics"], status="success")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        row["seconds"] = time.perf_counter() - started
        save_json(result_path, row)
        save_json(args.output_root / "summary.json", summarize(assets, audit, args.output_root, plan))
        print(f"{index + 1}/50 {row['status']} {asset['id']} {row['seconds']:.1f}s"
              + (f" {row['error']}" if row['status'] == "failed" else ""), flush=True)
    summary = summarize(assets, audit, args.output_root, plan)
    save_json(args.output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fit-samples", type=int, default=8192)
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
