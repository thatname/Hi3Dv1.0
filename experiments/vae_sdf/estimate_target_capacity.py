"""Predict per-asset R256 support cell counts from cheap mesh geometry.

Training memory scales with the number of finest-level support cells, so the
capacity policy needs a cell estimate for every candidate before any target is
generated. The estimator is calibrated on assets whose A4 audit already reports
`grid_quality.cells`, using only features available from a single mesh load
(normalized surface area, bounding-box extents, face count).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import traceback
from pathlib import Path

import numpy as np

from vae3d.data import load_mesh


def features(path: Path, resolution: int) -> dict:
    mesh, _, scale = load_mesh(str(path),
                               normalization_extent=(resolution - 2) / resolution)
    extent = np.asarray(mesh.bounds[1], dtype=float) - np.asarray(mesh.bounds[0])
    ex, ey, ez = (float(v) for v in extent)
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "scale": float(scale),
        "extent": [ex, ey, ez],
        "area": float(mesh.area),
        "bbox_area": 2.0 * (ex * ey + ey * ez + ez * ex),
        "diagonal": float(np.linalg.norm(extent)),
    }


def measure(row: dict, resolution: int) -> dict:
    try:
        out = features(Path(row["path"]), resolution)
        out.update({"id": row["id"], "status": "loaded"})
    except Exception as error:  # noqa: BLE001 - capacity audit records every failure
        out = {"id": row["id"], "status": "failed",
               "error": f"{type(error).__name__}: {error}"[:200],
               "traceback": traceback.format_exc()[-300:]}
    return out


def run(rows: list[dict], resolution: int, workers: int) -> list[dict]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda row: measure(row, resolution), rows))


# ---------------------------------------------------------------- calibration

def fits(samples: list[dict]) -> dict:
    """Fit candidate predictors on assets with a measured cell count."""

    area = np.array([s["area"] for s in samples], dtype=float)
    bbox = np.array([s["bbox_area"] for s in samples], dtype=float)
    faces = np.array([s["faces"] for s in samples], dtype=float)
    cells = np.array([s["measured_cells"] for s in samples], dtype=float)

    def report(name, pred):
        ratio = pred / cells
        return {"name": name,
                "median_ratio": float(np.median(ratio)),
                "p10_ratio": float(np.percentile(ratio, 10)),
                "p90_ratio": float(np.percentile(ratio, 90)),
                "median_abs_rel_error": float(np.median(np.abs(ratio - 1.0))),
                "p90_abs_rel_error": float(np.percentile(np.abs(ratio - 1.0), 90))}

    out = {}

    # area through the origin: cells ~ k * area
    k_area = float(np.sum(area * cells) / np.sum(area * area))
    out["area_through_origin"] = {"k": k_area,
                                  **report("area_through_origin", k_area * area)}

    # area scaled by the median cell-per-area ratio: robust to the heavy tail
    k_ratio = float(np.median(cells / area))
    out["area_median_ratio"] = {"k": k_ratio,
                                **report("area_median_ratio", k_ratio * area)}

    # affine in area
    slope, intercept = np.polyfit(area, cells, 1)
    out["area_affine"] = {"slope": float(slope), "intercept": float(intercept),
                          **report("area_affine", slope * area + intercept)}

    # bbox area through the origin: cells ~ k * bbox_area
    k_bbox = float(np.sum(bbox * cells) / np.sum(bbox * bbox))
    out["bbox_through_origin"] = {"k": k_bbox,
                                  **report("bbox_through_origin", k_bbox * bbox)}

    # log-log on area: log(cells) = b * log(area) + a
    b, a = np.polyfit(np.log(area), np.log(cells), 1)
    out["area_loglog"] = {"a": float(a), "b": float(b),
                          **report("area_loglog", np.exp(a) * area ** b)}

    # log-log on faces
    b2, a2 = np.polyfit(np.log(faces), np.log(cells), 1)
    out["faces_loglog"] = {"a": float(a2), "b": float(b2),
                           **report("faces_loglog", np.exp(a2) * faces ** b2)}

    # log-log on area and faces jointly
    design = np.stack([np.ones_like(area), np.log(area), np.log(faces)], axis=1)
    coef, *_ = np.linalg.lstsq(design, np.log(cells), rcond=None)
    pred = np.exp(design @ coef)
    out["area_faces_loglog"] = {"coef": [float(v) for v in coef],
                                **report("area_faces_loglog", pred)}
    return out


def predict_one(model: dict, sample: dict) -> float:
    """Apply the selected fit to one feature row."""

    fit = model["fits"][model["selected"]]
    area, faces = sample["area"], sample["faces"]
    if model["selected"].endswith("_loglog") and "coef" in fit:
        c0, c1, c2 = fit["coef"]
        return float(np.exp(c0 + c1 * np.log(area) + c2 * np.log(faces)))
    if model["selected"].endswith("_loglog"):
        return float(np.exp(fit["a"]) * area ** fit["b"])
    if model["selected"].endswith("_affine"):
        return float(fit["slope"] * area + fit["intercept"])
    return float(fit["k"] * area)


def load_measured(asset_root: Path, limit: int) -> list[dict]:
    """Read measured cells from four-route targets whose A4 audit exists."""

    rows = []
    for asset_dir in sorted(asset_root.iterdir()):
        if not asset_dir.is_dir():
            continue
        for route_dir in asset_dir.glob("R*/flexicubes_smooth-*"):
            audit = route_dir / "a4" / "audit.json"
            if not audit.is_file():
                continue
            try:
                data = json.loads(audit.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows.append({"id": asset_dir.name,
                         "path": None,
                         "resolution": int(route_dir.parent.name[1:]),
                         "measured_cells": int(data["grid_quality"]["cells"])})
        if limit and len(rows) >= limit:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "predict", "policy"), required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--assets", type=Path,
                        help="manifest whose assets carry the mesh paths")
    parser.add_argument("--measured", type=Path,
                        help="four-route target root with a4 audits")
    parser.add_argument("--id-map", type=Path,
                        help="g2 manifest giving id -> path for calibration meshes")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--threshold-cells", type=int, default=4_000_000)
    parser.add_argument("--calibration", type=Path,
                        help="calibrate-mode output reused by predict mode")
    parser.add_argument("--split", type=Path,
                        help="asset split filtered by the capacity policy")
    parser.add_argument("--output-split", type=Path,
                        help="split written after the over-budget assets are dropped")
    args = parser.parse_args()

    if args.mode == "policy":
        # the manifest now carries the normalized surface area, so the whole pool
        # can be screened without loading a single mesh again
        model = json.loads(args.calibration.read_text(encoding="utf-8"))
        manifest = json.loads(args.assets.read_text(encoding="utf-8"))
        split = json.loads(args.split.read_text(encoding="utf-8"))
        predicted = {a["id"]: predict_one(model, {"area": a["area"], "faces": a["faces"]})
                     for a in manifest["assets"]}
        over = {i: c for i, c in predicted.items() if c > args.threshold_cells}
        kept = {part: [i for i in split[f"{part}_ids"] if i not in over]
                for part in ("train", "val", "test")}
        index_of = {a["id"]: n for n, a in enumerate(manifest["assets"])}
        filtered = {"schema": split["schema"],
                    "manifest": split["manifest"],
                    "manifest_sha256": split["manifest_sha256"],
                    "rule": f"{split['rule']}; assets whose predicted R256 support cells "
                            f"exceed {args.threshold_cells} are dropped from every partition",
                    "capacity_policy": args.output.name,
                    **{f"{part}_indices": [index_of[i] for i in kept[part]]
                       for part in ("train", "val", "test")},
                    **{f"{part}_ids": kept[part] for part in ("train", "val", "test")}}
        summary = {
            "schema": "g3_capacity_policy_v1",
            "resolution": args.resolution,
            "threshold_cells": args.threshold_cells,
            "rule": f"an asset is excluded when predicted R256 support cells exceed "
                    f"{args.threshold_cells}; the estimate is cells = "
                    f"exp({model['fits'][model['selected']]['a']:.6f}) * area^"
                    f"{model['fits'][model['selected']]['b']:.6f} fitted on the measured "
                    f"A4 grid_quality.cells of {model['joined']} g2 assets",
            "model": model["selected"],
            "model_fit": model["fits"][model["selected"]],
            "calibration_error": {"median_abs_rel_error":
                                  model["fits"][model["selected"]]["median_abs_rel_error"],
                                  "p90_abs_rel_error":
                                  model["fits"][model["selected"]]["p90_abs_rel_error"]},
            "manifest": args.assets.name,
            "manifest_sha256": split["manifest_sha256"],
            "assets_total": len(manifest["assets"]),
            "excluded_total": len(over),
            "excluded_fraction": len(over) / len(manifest["assets"]) if manifest else None,
            "split_before": {part: len(split[f"{part}_ids"]) for part in kept},
            "split_after": {part: len(v) for part, v in kept.items()},
            "predicted_cells": {"median": float(np.median(list(predicted.values()))),
                                "p90": float(np.percentile(list(predicted.values()), 90)),
                                "p99": float(np.percentile(list(predicted.values()), 99)),
                                "max": float(max(predicted.values()))},
            "excluded": [{"id": i, "predicted_cells": round(c, 1)}
                         for i, c in sorted(over.items(), key=lambda kv: -kv[1])],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        args.output_split.write_text(json.dumps(filtered, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "excluded"}, indent=1))
        return 0

    if args.mode == "calibrate":
        measured = [r for r in load_measured(args.measured, args.limit)
                    if r["resolution"] == args.resolution]
        paths = {a["id"]: a["path"] for a in
                 json.loads(args.id_map.read_text(encoding="utf-8"))["assets"]}
        rows = [{"id": r["id"], "path": paths[r["id"]]} for r in measured
                if r["id"] in paths]
        samples = run(rows, args.resolution, args.workers)
        by_id = {r["id"]: r for r in measured}
        joined = [{**s, "measured_cells": by_id[s["id"]]["measured_cells"]}
                  for s in samples if s["status"] == "loaded"]
        fits_summary = fits(joined)
        # the simplest fit that survives the tail wins: prefer pure-area models,
        # and only fall back to the area+faces fit when neither tail holds up
        area_only = [k for k in fits_summary if k.startswith("area")
                     and k != "area_faces_loglog"]
        selected = min(area_only, key=lambda k: fits_summary[k]["p90_abs_rel_error"])
        if fits_summary[selected]["p90_abs_rel_error"] > 0.35:
            selected = "area_faces_loglog"
        model = {"fits": fits_summary, "selected": selected}
        predicted = [predict_one(model, s) for s in joined]
        measured_over = sum(1 for s in joined
                            if s["measured_cells"] > args.threshold_cells)
        predicted_over = sum(1 for c in predicted if c > args.threshold_cells)
        summary = {"mode": "calibrate", "resolution": args.resolution,
                   "measured": len(measured), "joined": len(joined),
                   "cells_max": int(max(s["measured_cells"] for s in joined)),
                   "cells_median": float(np.median([s["measured_cells"] for s in joined])),
                   "threshold_cells": args.threshold_cells,
                   "measured_over_threshold": measured_over,
                   "predicted_over_threshold": predicted_over,
                   "fits": fits_summary, "selected": selected,
                   "samples": joined}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=1))
        return 0

    model = json.loads(args.calibration.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))["assets"]
    if args.limit:
        import random
        assets = random.Random(args.seed).sample(assets, min(args.limit, len(assets)))
    samples = run([{"id": a["id"], "path": a["path"]} for a in assets],
                  args.resolution, args.workers)
    ok = [s for s in samples if s["status"] == "loaded"]
    predicted = [predict_one(model, s) for s in ok]
    over = [s["id"] for s, c in zip(ok, predicted) if c > args.threshold_cells]
    summary = {"mode": "predict", "resolution": args.resolution,
               "selected": model["selected"], "threshold_cells": args.threshold_cells,
               "sampled": len(samples), "loaded": len(ok),
               "sample_fraction_over": len(over) / len(ok) if ok else None,
               "sample_cells_median": float(np.median(predicted)) if ok else None,
               "sample_cells_p90": float(np.percentile(predicted, 90)) if ok else None,
               "sample_cells_p99": float(np.percentile(predicted, 99)) if ok else None,
               "sample_cells_max": float(np.max(predicted)) if ok else None,
               "rows": [{k2: s[k2] for k2 in ("id", "faces", "area")} | {"cells_predicted": c}
                        for s, c in zip(ok, predicted)],
               "failed": [s for s in samples if s["status"] != "loaded"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k2: v for k2, v in summary.items()
                      if k2 not in ("rows", "failed")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())