"""Run and resume fixed 50-asset G0 A4 consistent-depth render fitting."""

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


SCHEMA = "g0_a4_batch_v1"


def load_plan(args):
    manifest_bytes = args.manifest.read_bytes()
    a3_bytes = args.a3_summary.read_bytes()
    a0_bytes = args.a0_audit.read_bytes()
    assets = json.loads(manifest_bytes)["assets"]
    a3_rows = json.loads(a3_bytes)["assets"]
    a0_rows = json.loads(a0_bytes)["assets"]
    if not (len(assets) == len(a3_rows) == len(a0_rows) == 50):
        raise ValueError("Expected fixed 50-asset inputs")
    for index, (asset, a3, flag) in enumerate(zip(assets, a3_rows, a0_rows)):
        if (a3["index"] != index or a3["id"] != asset["id"]
                or a3["status"] != "success" or flag["index"] != index
                or flag["id"] != asset["id"] or a3["a0_status"] != flag["status"]):
            raise ValueError(f"Input mismatch at index {index}")
    vae3d_commit = subprocess.check_output(
        ["git", "-C", str(args.vae3d_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if vae3d_commit != VAE3D_COMMIT:
        raise ValueError(f"Wrong vae3d commit: {vae3d_commit}")
    raster_commit = subprocess.check_output(
        ["git", "-C", str(args.nvdiffrast_root), "rev-parse", "HEAD"], text=True
    ).strip()
    module_dir = Path(__file__).parent
    plan = {
        "schema": SCHEMA, "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "a3_summary": str(args.a3_summary),
        "a3_summary_sha256": hashlib.sha256(a3_bytes).hexdigest(),
        "a0_audit": str(args.a0_audit),
        "a0_audit_sha256": hashlib.sha256(a0_bytes).hexdigest(),
        "vae3d_root": str(args.vae3d_root), "vae3d_commit": vae3d_commit,
        "nvdiffrast_root": str(args.nvdiffrast_root),
        "nvdiffrast_commit": raster_commit,
        "resolution": args.resolution, "steps": args.steps,
        "image_resolution": args.image_resolution,
        "metric_samples": args.metric_samples, "seed": args.seed,
        "device": args.device, "render_backend": "cuda",
        "extractor": "vae3d classic MC for A3 and A4",
        "a0_policy": "retain_flags_no_exclusion",
        "code_sha256": {name: sha256(module_dir / name) for name in (
            "prepare_a4_targets.py", "run_a4_batch.py",
            "prepare_a3_targets.py", "evaluate_initial_targets.py")},
        "asset_count": len(assets),
    }
    return assets, a0_rows, plan


def summarize(assets, a0_rows, root, plan):
    import numpy as np

    rows = []
    for index, (asset, flag) in enumerate(zip(assets, a0_rows)):
        path = root / "assets" / f"{index:02d}.json"
        row = json.loads(path.read_text()) if path.exists() else {
            "index": index, "id": asset["id"], "status": "pending",
            "a0_status": flag["status"], "a0_concerns": flag.get("concerns", []),
        }
        if row["index"] != index or row["id"] != asset["id"]:
            raise ValueError(f"Result mismatch at index {index}")
        rows.append(row)
    counts = {state: sum(row["status"] == state for row in rows)
              for state in ("success", "failed", "pending")}
    groups = {}
    for label, selection in (("all", rows), ("closed_candidate", rows[:20]),
                             ("nonclosed", rows[20:]),
                             ("a0_needs_review", [r for r in rows if r["a0_status"] == "needs_review"])):
        good = [row for row in selection if row["status"] == "success"]
        group = {"denominator": len(selection), "success": len(good),
                 "failed": sum(r["status"] == "failed" for r in selection),
                 "pending": sum(r["status"] == "pending" for r in selection)}
        for stage in ("A3_classic", "A4"):
            group[stage] = {}
            for name, get in (
                ("chamfer_l1", lambda m: m["chamfer_l1"]),
                ("normal_consistency", lambda m: m["normal_consistency"]),
                ("fscore_0.005", lambda m: m["fscore"]["0.005"]["fscore"]),
                ("fscore_0.01", lambda m: m["fscore"]["0.01"]["fscore"]),
            ):
                values = [get(r["metrics"][stage]) for r in good]
                values = [v for v in values if v is not None]
                group[stage][name] = ({"mean": float(np.mean(values)),
                                      "median": float(np.median(values)),
                                      "count": len(values)} if values else None)
        comparable = [r for r in good if all(
            r["metrics"][s]["chamfer_l1"] is not None for s in ("A3_classic", "A4"))]
        group["A4_better_chamfer"] = sum(
            r["metrics"]["A4"]["chamfer_l1"] < r["metrics"]["A3_classic"]["chamfer_l1"]
            for r in comparable)
        group["comparable_chamfer"] = len(comparable)
        group["inverted_crossing_assets"] = sum(
            r["audit"]["grid_quality"]["inverted_crossing_cells"] > 0 for r in good)
        groups[label] = group
    return {"schema": SCHEMA, "plan_sha256": hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "denominator": len(rows), "counts": counts, "groups": groups,
        "assets": rows}


def run(args):
    from .prepare_a4_targets import run as prepare_one

    assets, a0_rows, plan = load_plan(args)
    plan_path = args.output_root / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing A4 plan differs; use a new output root")
    else:
        save_json(plan_path, plan)
    for index, (asset, flag) in enumerate(zip(assets, a0_rows)):
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
                manifest=args.manifest, a3_summary=args.a3_summary,
                a0_audit=args.a0_audit, vae3d_root=args.vae3d_root,
                nvdiffrast_root=args.nvdiffrast_root, index=index,
                output_root=args.output_root / "targets", resolution=args.resolution,
                steps=args.steps, image_resolution=args.image_resolution,
                metric_samples=args.metric_samples, seed=args.seed,
                device=args.device)
            result = prepare_one(one)
            row.update(cache=result["cache"], audit=result["audit"],
                       metrics=result["metrics"], status="success")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        row["seconds"] = time.perf_counter() - started
        save_json(result_path, row)
        save_json(args.output_root / "summary.json",
                  summarize(assets, a0_rows, args.output_root, plan))
        print(f"{index + 1}/50 {row['status']} {asset['id']} {row['seconds']:.1f}s"
              + (f" {row['error']}" if row['status'] == "failed" else ""), flush=True)
    result = summarize(assets, a0_rows, args.output_root, plan)
    save_json(args.output_root / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a3-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--nvdiffrast-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--image-resolution", type=int, default=4096)
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
