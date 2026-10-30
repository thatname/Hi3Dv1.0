"""Read-only G1 field diagnostics on fixed GT support; prints JSON to stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparc3d.models.autoencoders.config import load_sparconv_vae

from .g1_model import G1VAE, _rows
from .g1_split import check_split
from .train_g1 import load_asset, sha256


def weighted_mean(value: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor):
    selected = weight * mask
    denom = selected.sum()
    return float((value * selected).sum() / denom) if bool(denom > 0) else None


def diagnose(asset: dict, model: G1VAE) -> dict:
    with torch.inference_mode():
        cells, signed, delta, _, _, _ = model(
            asset["coords"], asset["features"], asset["cells"], 128)
        rows = _rows(asset["cells"], cells, 128)
        target = asset["field"][rows]
        deform = asset["delta"][rows]
        weight = asset["weight"][rows]
        absolute = target.abs()
        sign_wrong = ((signed < 0) != (target < 0)).float()
        field_error = (signed - target).abs()
        delta_error = (delta - deform).abs().mean(-1)
        bins = {}
        for label, mask in (
            ("abs_s_le_0.5", absolute <= .5),
            ("abs_s_le_1", absolute <= 1),
            ("abs_s_le_2", absolute <= 2),
            ("abs_s_gt_2", absolute > 2),
        ):
            bins[label] = {
                "unique_weighted_points": float((weight * mask).sum()),
                "field_mae": weighted_mean(field_error, weight, mask),
                "sign_error_rate": weighted_mean(sign_wrong, weight, mask & (absolute > 1e-5)),
                "delta_mae": weighted_mean(delta_error, weight, mask),
            }
        target_cross = (target < 0).any(-1) & (target >= 0).any(-1)
        predicted_cross = (signed < 0).any(-1) & (signed >= 0).any(-1)
        overlap = int((target_cross & predicted_cross).sum())
        pred_count = int(predicted_cross.sum())
        target_count = int(target_cross.sum())
        return {
            "index": asset["index"], "id": asset["id"],
            "gt_support_cells": len(cells),
            "crossing_cells": {
                "target": target_count, "predicted": pred_count, "overlap": overlap,
                "precision": overlap / pred_count if pred_count else 0,
                "recall": overlap / target_count if target_count else 0,
            },
            "field_min_max": [float(signed.min()), float(signed.max())],
            "target_field_min_max": [float(target.min()), float(target.max())],
            "bins": bins,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a4-summary", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    manifest_hash = sha256(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    rows = json.loads(args.a4_summary.read_text())["assets"]
    split_hash = check_split(args.split, manifest, manifest_hash, "val", args.indices)
    state = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    config = state["config"]
    if (config.get("status") != "completed"
            or config.get("schema") != "g1_a4_overfit_v2"
            or config.get("split_sha256") != split_hash
            or config.get("manifest_sha256") != manifest_hash
            or config.get("a4_summary_sha256") != sha256(args.a4_summary)
            or config.get("pretrained_weights_sha256") != sha256(args.weights)
            or state["steps"] != config["steps"]):
        raise ValueError("G1 checkpoint, split, or target provenance mismatch")
    model = G1VAE(load_sparconv_vae(args.weights, device="cuda:0").decoder).to("cuda:0")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    print(json.dumps({
        "schema": "g1_field_diagnostic_v1", "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint), "step": config["steps"],
        "split_sha256": split_hash, "support": "GT fine cells and GT coarse latent coordinates",
        "assets": [diagnose(load_asset(index, manifest, rows, manifest_hash, "cuda:0"), model)
                   for index in args.indices],
    }, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
