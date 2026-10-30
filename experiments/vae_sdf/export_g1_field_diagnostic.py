"""Export field and support diagnostics for G1 indices 0 and 7."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sparc3d.models.autoencoders.config import load_sparconv_vae

from .g1_model import G1VAE, _rows
from .g1_targets import ROUTE, target_rows as select_target_rows
from .train_g1 import load_asset, sha256, write_json


def export(index: int, asset: dict, model: G1VAE, destination: Path) -> dict:
    with torch.inference_mode():
        gt_cells, gt_s, gt_delta, _, mean, logvar = model(
            asset["coords"], asset["features"], asset["cells"], 128)
        pred_cells, pred_s, _, _, _, _ = model(
            asset["coords"], asset["features"], asset["cells"], 128, predicted_support=True)
        target_cells = asset["cells"]
        # _rows maps each target cell to its exact predicted-field row.
        target_to_pred = _rows(target_cells, gt_cells, 128)
        target_s = asset["field"]
        pred_s_on_target = gt_s[target_to_pred]
        pred_delta_on_target = gt_delta[target_to_pred]
        target_keys = target_cells[:, 0].long() * 16384 + target_cells[:, 1].long() * 128 + target_cells[:, 2].long()
        pred_keys = pred_cells[:, 0].long() * 16384 + pred_cells[:, 1].long() * 128 + pred_cells[:, 2].long()
        target_has_prediction = torch.isin(target_keys, pred_keys)
        prediction_has_target = torch.isin(pred_keys, target_keys)

        payload = {
            "index": index, "resolution": 128,
            "target_cells": target_cells.cpu().numpy().astype(np.int16).tolist(),
            "target_s": target_s.cpu().numpy().astype(np.float32).tolist(),
            "prediction_s_on_target": pred_s_on_target.cpu().numpy().astype(np.float32).tolist(),
            "target_has_prediction": target_has_prediction.cpu().numpy().astype(np.uint8).tolist(),
            "prediction_cells": pred_cells.cpu().numpy().astype(np.int16).tolist(),
            "prediction_s": pred_s.cpu().numpy().astype(np.float32).tolist(),
            "prediction_has_target": prediction_has_target.cpu().numpy().astype(np.uint8).tolist(),
        }
        with gzip.open(destination / f"index{index:02d}_slice.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
        np.savez_compressed(
            destination / f"index{index:02d}_fields.npz",
            target_cells=target_cells.cpu().numpy().astype(np.int16),
            target_s=target_s.cpu().numpy().astype(np.float32),
            prediction_s_on_target=pred_s_on_target.cpu().numpy().astype(np.float32),
            target_has_prediction=target_has_prediction.cpu().numpy().astype(np.uint8),
            prediction_cells=pred_cells.cpu().numpy().astype(np.int16),
            prediction_s=pred_s.cpu().numpy().astype(np.float32),
            prediction_has_target=prediction_has_target.cpu().numpy().astype(np.uint8),
            target_delta=asset["delta"].cpu().numpy().astype(np.float32),
            prediction_delta_on_target=pred_delta_on_target.cpu().numpy().astype(np.float32),
            latent_mean=mean.cpu().numpy().astype(np.float32),
            latent_logvar=logvar.cpu().numpy().astype(np.float32),
        )
    overlap = int(target_has_prediction.sum())
    return {
        "index": index, "asset_id": asset["id"], "target_sha256": asset["target_sha256"],
        "route_id": ROUTE, "target_cells": int(len(target_cells)),
        "predicted_cells": int(len(pred_cells)), "overlap_cells": overlap,
        "precision": float(prediction_has_target.sum() / max(len(pred_cells), 1)),
        "recall": float(overlap / max(len(target_cells), 1)),
        "slice_file": f"index{index:02d}_slice.json.gz",
        "field_npz": f"index{index:02d}_fields.npz",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paired-summary", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data/hi3d/g1").resolve()):
        parser.error("Choose a new diagnostic directory under /data/hi3d/g1")
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    rows = select_target_rows(args.paired_summary, manifest, four_route=True)
    state = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    config = state["config"]
    if (config.get("schema") != "g1_four_route_v1" or config.get("status") != "completed"
            or config.get("target_route") != ROUTE or config.get("manifest_sha256") != manifest_hash
            or config.get("paired_summary_sha256") != sha256(args.paired_summary)
            or config.get("pretrained_weights_sha256") != sha256(args.weights)
            or int(state["steps"]) != 1000):
        raise ValueError("Expected the completed 1000-step flexicubes_smooth G1 checkpoint")
    model = G1VAE(load_sparconv_vae(args.weights, device="cuda:0").decoder).to("cuda:0")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    args.output.mkdir(parents=True)
    report = {
        "schema": "g1_field_diagnostic_v1", "status": "running",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": 1000, "manifest_sha256": manifest_hash,
        "paired_summary_sha256": sha256(args.paired_summary), "route_id": ROUTE,
        "grid_resolution": 128, "sampling_offset": 0.5,
        "coordinates": "fine-grid cells; phase 0.5; field in fine-voxel units",
        "support_classes": {"target_has_prediction=0": "target-only cell",
                             "target_has_prediction=1": "intersection cell",
                             "prediction_has_target=0": "prediction-only cell"},
        "assets": [],
    }
    write_json(args.output / "report.json", report)
    for index in (0, 7):
        asset = load_asset(index, manifest, rows, manifest_hash, "cuda:0")
        report["assets"].append(export(index, asset, model, args.output))
        write_json(args.output / "report.json", report)
    report["status"] = "completed"
    write_json(args.output / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
