"""Run and resume the G0 A1/A2 CPU baseline for a fixed manifest.

The manifest length is whatever the supplied file declares; the load audit must
match it asset for asset. The original balanced-50 batch and any later extension
(e.g. the 100-asset G2 pretest) therefore use the same code path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from .evaluate_initial_targets import PROTOCOL, evaluate_pair
from .preflight import VAE3D_COMMIT, sha256


SCHEMA = "g0_a1a2_batch_v1"


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_inputs(args) -> tuple[list[dict], dict]:
    manifest_bytes = args.manifest.read_bytes()
    assets = json.loads(manifest_bytes)["assets"]
    audit_bytes = args.load_audit.read_bytes()
    audit = json.loads(audit_bytes)
    if not assets or len(audit["assets"]) != len(assets) or audit["resolution"] != args.resolution:
        raise ValueError("Manifest and load audit must match at this resolution")
    for index, (asset, checked) in enumerate(zip(assets, audit["assets"])):
        if (checked["index"] != index or checked["id"] != asset["id"]
                or checked["path"] != asset["path"] or checked["status"] != "loaded"):
            raise ValueError(f"Manifest/load-audit mismatch at index {index}")
    head = subprocess.check_output(
        ["git", "-C", str(args.vae3d_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != VAE3D_COMMIT:
        raise ValueError(f"Wrong vae3d commit: {head}")
    module_dir = Path(__file__).parent
    code_hashes = {name: sha256(module_dir / name) for name in (
        "initial_fields.py", "prepare_initial_targets.py", "check_initial_targets.py",
        "evaluate_initial_targets.py", "run_initial_batch.py")}
    plan = {"schema": SCHEMA, "manifest": str(args.manifest),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "load_audit": str(args.load_audit),
            "load_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "vae3d_root": str(args.vae3d_root), "vae3d_commit": head,
            "resolution": args.resolution, "raycast_backend": "cpu",
            "distance_backend": "cpu", "support_stages": ["A1", "A2"],
            "extractor": "skimage.measure.marching_cubes (Lewiner)",
            "samples_per_direction": args.samples, "seed": args.seed,
            "threshold_fractions": [0.005, 0.01], "metric_protocol": PROTOCOL,
            "code_sha256": code_hashes, "asset_count": len(assets)}
    return assets, plan


def cached_pair(target_root: Path, asset: dict, plan: dict) -> tuple[Path, Path] | None:
    base = target_root / asset["id"] / f"R{plan['resolution']}"
    found = {}
    if not base.is_dir():
        return None
    for path in base.iterdir():
        config_file = path / "resolved_config.json"
        if not config_file.is_file():
            continue
        config = json.loads(config_file.read_text())
        if (config.get("source_sha256") != asset["sha256"]
                or config.get("manifest_sha256") != plan["manifest_sha256"]
                or config.get("resolution") != plan["resolution"]
                or config.get("raycast_backend") != "cpu"
                or config.get("distance_backend") != "cpu"
                or config.get("support_stages") != ["A1", "A2"]):
            continue
        stage = config.get("stage")
        if stage in {"A1", "A2"}:
            if stage in found:
                raise ValueError(f"Duplicate {stage} cache for {asset['id']}")
            found[stage] = path
    if not found:
        return None
    if set(found) != {"A1", "A2"}:
        raise ValueError(f"Incomplete A1/A2 cache pair for {asset['id']}: {sorted(found)}")
    return found["A1"], found["A2"]


def summarize(assets: list[dict], result_root: Path, plan: dict) -> dict:
    import numpy as np

    rows = []
    for index, asset in enumerate(assets):
        path = result_root / "assets" / f"{index:02d}.json"
        if path.is_file():
            rows.append(json.loads(path.read_text()))
        else:
            rows.append({"index": index, "id": asset["id"], "status": "pending"})
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("success", "failed", "pending")}
    summary = {"schema": SCHEMA, "plan_sha256": hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "denominator": len(assets), "counts": counts, "assets": rows,
        "groups": {}}
    for label, selection in (
        ("all", rows),
        ("closed_candidate", [row for asset, row in zip(assets, rows)
                              if asset.get("topology_candidate") == "watertight_candidate"]),
        ("nonclosed", [row for asset, row in zip(assets, rows)
                       if asset.get("topology_candidate") != "watertight_candidate"]),
    ):
        complete = [row for row in selection if row["status"] == "success"]
        group = {"denominator": len(selection), "success": len(complete),
                 "failed": sum(row["status"] == "failed" for row in selection),
                 "pending": sum(row["status"] == "pending" for row in selection)}
        for stage in ("A1", "A2"):
            metrics = [row["metrics"]["stages"][stage] for row in complete]
            stage_stats = {}
            for name in ("chamfer_l1", "normal_consistency"):
                values = [m[name] for m in metrics if m[name] is not None]
                stage_stats[name] = {"mean": float(np.mean(values)),
                                     "median": float(np.median(values)),
                                     "count": len(values)} if values else None
            for fraction in ("0.005", "0.01"):
                values = [m["fscore"][fraction]["fscore"] for m in metrics
                          if isinstance(m["fscore"][fraction], dict)]
                stage_stats[f"fscore_{fraction}"] = {
                    "mean": float(np.mean(values)), "median": float(np.median(values)),
                    "count": len(values)} if values else None
            group[stage] = stage_stats
        if complete:
            group["A2_better_chamfer"] = sum(
                row["metrics"]["stages"]["A2"]["chamfer_l1"]
                < row["metrics"]["stages"]["A1"]["chamfer_l1"]
                for row in complete if all(row["metrics"]["stages"][s]["chamfer_l1"] is not None
                                            for s in ("A1", "A2")))
        summary["groups"][label] = group
    return summary


def run(args) -> dict:
    from .prepare_initial_targets import run as prepare_one

    assets, plan = load_inputs(args)
    plan_path = args.output_root / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing batch plan differs; use a new output root")
    else:
        save_json(plan_path, plan)
    target_root = args.output_root / "targets"
    selected = set(args.indices) if args.indices else None
    for index, asset in enumerate(assets):
        if args.max_assets is not None and index >= args.max_assets:
            break
        if selected is not None and index not in selected:
            continue
        result_path = args.output_root / "assets" / f"{index:02d}.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous["status"] == "success" or not args.retry_failed:
                continue
        started = time.perf_counter()
        row = {"index": index, "id": asset["id"], "source": asset["path"],
               "source_sha256": asset["sha256"], "status": "failed"}
        try:
            pair = cached_pair(target_root, asset, plan)
            if pair is None:
                one = argparse.Namespace(
                    manifest=args.manifest, index=index, vae3d_root=args.vae3d_root,
                    output_root=target_root, stage="both", resolution=args.resolution,
                    raycast_backend="cpu", distance_backend="cpu", max_sites=args.max_sites,
                    verify_reference=args.verify_reference,
                )
                output = prepare_one(one)
                pair = Path(output["A1"]["cache"]), Path(output["A2"]["cache"])
            row["caches"] = {"A1": str(pair[0]), "A2": str(pair[1])}
            row["metrics"] = evaluate_pair(
                pair[0], pair[1], Path(asset["path"]), resolution=args.resolution,
                samples=args.samples, seed=args.seed,
            )
            row["status"] = "success"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        row["seconds"] = time.perf_counter() - started
        save_json(result_path, row)
        if not args.no_per_asset_summary:
            # rewriting the whole summary after every asset is O(N^2) in file reads,
            # which costs hours once the manifest reaches tens of thousands of rows
            summary = summarize(assets, args.output_root, plan)
            save_json(args.output_root / "summary.json", summary)
        print(f"{index + 1}/{len(assets)} {row['status']} {asset['id']} {row['seconds']:.1f}s"
              + (f" {row['error']}" if row['status'] == "failed" else ""), flush=True)
    summary = summarize(assets, args.output_root, plan)
    save_json(args.output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--load-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--max-sites", type=int, default=2_000_000)
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--max-assets", type=int, help="Run only the first N assets for a smoke test")
    parser.add_argument("--indices", type=int, nargs="+",
                        help="Run only these manifest indices; every other asset stays pending")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--no-per-asset-summary", action="store_true",
                        help="Skip the in-loop summary rewrite; the caller summarizes instead")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary["counts"]))
    return 0 if summary["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
