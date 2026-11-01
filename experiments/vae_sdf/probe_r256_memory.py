"""Per-asset R256 training memory probe: one forward/backward per asset.

Writes one row per asset with its finest-level cell count and the peak CUDA
allocation of a single training step, so an R256 budget can be derived from the
cell distribution instead of guessed. With --cycles the same assets repeat, which
separates "this asset is big" from "memory accumulates across steps".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.vae_sdf.g1_model import G1VAE, _rows
from experiments.vae_sdf.g1_targets import target_rows as select_target_rows
from experiments.vae_sdf.train_g1 import (cell_validity, grad_status, load_asset,
                                          upload)

from sparc3d.models.autoencoders.config import load_sparconv_vae


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--paired-summary", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--asset-device", choices=("cuda", "cpu"), default="cpu")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--encoder-scale", default="E0",
                   choices=("E0", "E1", "E2", "E3"))
    p.add_argument("--keep-locals", action="store_true",
                   help="Mimic the trainer loop: keep the previous step's tensors bound "
                        "and skip the per-step cache flush")
    p.add_argument("--surface-weight", type=float, default=4.0)
    p.add_argument("--surface-sigma", type=float, default=1.0)
    p.add_argument("--truncation", type=float, default=4.0)
    p.add_argument("--validity-weight", type=float, default=0.1)
    args = p.parse_args()

    device = "cuda:0"
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    rows = select_target_rows(args.paired_summary, manifest, four_route=True)
    pretrained = load_sparconv_vae(args.weights, device=device)
    model = G1VAE(pretrained.decoder, args.encoder_scale).to(device).train()
    groups = {
        "encoder": model.encoder,
        "decoder": model.decoder,
    }
    counts = {name: sum(p.numel() for p in mod.parameters()) / 1e6
              for name, mod in groups.items()}
    counts["total"] = sum(p.numel() for p in model.parameters()) / 1e6
    print(json.dumps({"encoder_scale": args.encoder_scale, "params_M": counts}),
          flush=True)
    for param in model.decoder.backbone.conv_out.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    assets = {i: load_asset(i, manifest, rows, manifest_hash, device, args.resolution,
                            args.asset_device)
              for i in args.indices}

    results = []
    step = 0
    for cycle in range(args.cycles):
        for index in args.indices:
            if not args.keep_locals:
                torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            before = torch.cuda.memory_allocated(0)
            try:
                asset = upload(assets[index], device)
                optimizer.zero_grad(set_to_none=True)
                cells, signed, delta, occupancy, mean, logvar = model(
                    asset["coords"], asset["features"], asset["cells"], args.resolution)
                target_rows = _rows(asset["cells"], cells, args.resolution)
                weights = asset["weight"][target_rows]
                target_field = asset["field"][target_rows]
                one_sided = target_field.sign() * torch.minimum(
                    target_field.sign() * signed, signed.new_tensor(args.truncation))
                saturated = target_field.abs() >= args.truncation - 1e-3
                scored = torch.where(saturated, one_sided, signed)
                surface = 1 + args.surface_weight * torch.exp(
                    -target_field.abs() / args.surface_sigma)
                field = F.smooth_l1_loss(scored, target_field, reduction="none")
                deform = F.smooth_l1_loss(delta, asset["delta"][target_rows],
                                          reduction="none").mean(-1)
                loss = ((field * weights * surface).sum() / (weights * surface).sum()
                        + (deform * weights).sum() / weights.sum()
                        + torch.stack(occupancy).mean()
                        + args.validity_weight * cell_validity(delta))
                loss.backward()
                groups = {name: grad_status(params) for name, params in (
                    ("encoder", model.encoder.parameters()),
                    ("decoder_backbone", model.decoder.backbone.parameters()),
                    ("continuous_head", model.decoder.head.parameters()))}
                peak = torch.cuda.max_memory_allocated(0)
                row = {"step": step, "cycle": cycle, "index": index,
                       "fine_cells": int(len(cells)),
                       "allocated_before": int(before), "step_peak_bytes": int(peak),
                       "bytes_per_cell": float(peak / max(len(cells), 1)),
                       "loss": float(loss.detach()),
                       "gradients_ok": all(g["present"] and g["finite"]
                                           for g in groups.values())}
                if not args.keep_locals:
                    del asset, cells, signed, delta, occupancy, mean, logvar
            except torch.OutOfMemoryError as error:
                row = {"step": step, "cycle": cycle, "index": index, "status": "oom",
                       "error": str(error)[:200]}
                torch.cuda.empty_cache()
            results.append(row)
            print(json.dumps(row), flush=True)
            step += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"resolution": args.resolution,
                                       "encoder_scale": args.encoder_scale,
                                       "asset_device": args.asset_device,
                                       "rows": results}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())