"""G1 A4 paired-field overfit smoke on the fixed balanced-50 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from sparc3d.models.autoencoders.config import load_sparconv_vae
from vae3d.geometry import validity_loss
from .g0b_field_adapter import HI3D_FROM_VAE, VAE_FROM_HI3D, pack_target
from .g1_model import G1VAE, _rows
from .g1_split import check_split


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, model, optimizer, step: int, report: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "steps": step, "config": report,
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(), "python": random.getstate()},
    }, temporary)
    temporary.replace(path)


def load_asset(index: int, manifest: dict, rows: list, manifest_hash: str, device: str):
    asset, row = manifest["assets"][index], rows[index]
    if row["index"] != index or row["id"] != asset["id"] or row["status"] != "success":
        raise ValueError("A4 summary and fixed manifest disagree")
    cache = Path(row["cache"])
    config = json.loads((cache / "resolved_config.json").read_text())
    if (config["asset_id"] != asset["id"] or config["manifest_sha256"] != manifest_hash
            or config["resolution"] != 128 or config["sampling_offset"] != 0.5):
        raise ValueError("A4 target provenance differs from G1 protocol")
    with np.load(cache / "target.npz", allow_pickle=False) as target:
        coords = np.asarray(target["coords"], np.int32)
        features = np.asarray(target["features"], np.float32)
        cells = np.asarray(target["cells"], np.int32)
        indices = np.asarray(target["cell_indices"], np.int64)
        resolution = int(target["resolution"])
        phase = float(target["sampling_offset"])
    if resolution != 128 or phase != 0.5 or np.max(np.abs(features[:, 1:])) > .50001:
        raise ValueError("A4 target resolution, phase, or voxel delta is invalid")
    pack_target(coords, features, cells, indices, resolution)
    hi_indices = indices[:, HI3D_FROM_VAE]
    counts = np.bincount(hi_indices.reshape(-1), minlength=len(coords))
    weight = (1 / counts[hi_indices]).astype(np.float32)
    return {
        "index": index, "id": asset["id"], "cache": str(cache),
        "target_sha256": sha256(cache / "target.npz"), "a0_status": row["a0_status"],
        "a0_concerns": row["a0_concerns"],
        "inverted_crossing_cells": row["audit"]["grid_quality"]["inverted_crossing_cells"],
        "coords": torch.as_tensor(coords, device=device),
        "features": torch.as_tensor(features, device=device),
        "cells": torch.as_tensor(cells, device=device),
        "field": torch.as_tensor(features[hi_indices, 0], device=device),
        "delta": torch.as_tensor(features[hi_indices, 1:], device=device),
        "weight": torch.as_tensor(weight, device=device),
    }


def grad_status(parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return {"present": False, "finite": False, "norm": None}
    finite = all(bool(torch.isfinite(g).all()) for g in gradients)
    squared = sum(float(g.detach().float().square().sum()) for g in gradients)
    return {"present": True, "finite": finite, "norm": math.sqrt(squared)}


def cell_validity(delta: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """The vae3d six-tetrahedron soft validity loss on predicted cells."""
    ordered = delta[:, VAE_FROM_HI3D]
    total = ordered.new_zeros(())
    for start in range(0, len(ordered), chunk):
        block = ordered[start:start + chunk]
        value = checkpoint(lambda x: validity_loss(None, x), block, use_reentrant=False)
        total = total + value * (len(block) / len(ordered))
    return total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--a4-summary", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path, help="Exact model/optimizer continuation from a G1 checkpoint")
    p.add_argument("--save-every", type=int, default=0,
                   help="Keep a separately named checkpoint every N updates; 0 saves only at the end")
    p.add_argument("--indices", type=int, nargs="+", default=[48, 5])
    p.add_argument("--split", type=Path, help="Fixed asset-level split; indices must match its train role")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--surface-weight", type=float, default=4.0)
    p.add_argument("--surface-sigma", type=float, default=1.0)
    p.add_argument("--truncation", type=float, default=4.0)
    p.add_argument("--validity-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20261030)
    args = p.parse_args()
    if (args.steps < 1 or args.save_every < 0 or args.lr <= 0 or not args.indices
            or len(set(args.indices)) != len(args.indices)
            or args.surface_weight < 0 or args.surface_sigma <= 0
            or args.truncation < 0 or args.validity_weight < 0):
        p.error("Invalid steps, learning rate, or indices")
    if args.output.exists():
        p.error("Output exists; choose a new run directory to retain provenance")
    if not torch.cuda.is_available():
        p.error("G1 Hi3D/spconv smoke requires CUDA")
    device = "cuda:0"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    rows = json.loads(args.a4_summary.read_text())["assets"]
    if len(manifest["assets"]) != 50 or len(rows) != 50:
        raise ValueError("G1 requires the fixed balanced-50 A4 batch")
    split_hash = check_split(args.split, manifest, manifest_hash, "train", args.indices) if args.split else None
    assets = [load_asset(i, manifest, rows, manifest_hash, device) for i in args.indices]
    pretrained = load_sparconv_vae(args.weights, device=device)
    model = G1VAE(pretrained.decoder).to(device).train()
    # The released sign/distance head is retained only for exact backbone loading.
    for param in model.decoder.backbone.conv_out.parameters():
        param.requires_grad_(False)
    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([p for _, p in named_trainable], lr=args.lr)
    prior_trace = []
    start_step = 0
    rng_restored = False
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        old = state["config"]
        expected = {
            "schema": "g1_a4_overfit_v2", "indices": args.indices,
            "manifest_sha256": manifest_hash, "a4_summary_sha256": sha256(args.a4_summary),
            "pretrained_weights_sha256": sha256(args.weights), "lr": args.lr,
            "surface_weight": args.surface_weight, "surface_sigma": args.surface_sigma,
            "truncation": args.truncation, "validity_weight": args.validity_weight,
            "beta": 0, "teacher_forced_support": True,
            "split_sha256": split_hash,
        }
        mismatch = [key for key, value in expected.items() if old.get(key) != value]
        if mismatch or old.get("status") != "completed":
            raise ValueError(f"G1 checkpoint configuration mismatch: {mismatch}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["steps"])
        prior_trace = list(old.get("trace", []))
        if len(prior_trace) != start_step:
            raise ValueError("G1 checkpoint step and trace length differ")
        if "rng" in state:
            torch.set_rng_state(state["rng"]["torch"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
            np.random.set_state(state["rng"]["numpy"])
            random.setstate(state["rng"]["python"])
            rng_restored = True
    args.output.mkdir(parents=True)
    report = {
        "schema": "g1_a4_overfit_v2", "status": "running", "beta": 0,
        "teacher_forced_support": True, "indices": args.indices,
        "steps": start_step + args.steps, "additional_steps": args.steps,
        "start_step": start_step, "resume_from": str(args.resume) if args.resume else None,
        "rng_restored": rng_restored, "save_every": args.save_every,
        "split": str(args.split) if args.split else None,
        "split_sha256": split_hash,
        "lr": args.lr, "seed": args.seed, "manifest_sha256": manifest_hash,
        "surface_weight": args.surface_weight, "surface_sigma": args.surface_sigma,
        "truncation": args.truncation, "validity_weight": args.validity_weight,
        "a4_summary_sha256": sha256(args.a4_summary),
        "pretrained_weights": str(args.weights), "pretrained_weights_sha256": sha256(args.weights),
        "pretrained_loading": "strict: all released Hi3D decoder parameters; old output head frozen/unused; new head random",
        "encoder_initialization": "random; adapted vae3d Encoder structure, factor 8/latent 32",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {name: sha256(Path(__file__).parent / name) for name in
                        ("train_g1.py", "g1_model.py", "g0b_field_adapter.py")},
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "assets": [{k: v for k, v in a.items() if not isinstance(v, torch.Tensor)} for a in assets],
        "trace": prior_trace,
    }
    write_json(args.output / "resolved_config.json", report)
    started = time.perf_counter()
    try:
        for step in range(start_step, start_step + args.steps):
            asset = assets[step % len(assets)]
            optimizer.zero_grad(set_to_none=True)
            cells, signed, delta, occupancy, mean, logvar = model(
                asset["coords"], asset["features"], asset["cells"], 128)
            target_rows = _rows(asset["cells"], cells, 128)
            weights = asset["weight"][target_rows]
            target_field = asset["field"][target_rows]
            if args.truncation > 0:
                saturated = target_field.abs() >= args.truncation - 1e-3
                one_sided = target_field.sign() * torch.minimum(
                    target_field.sign() * signed, signed.new_tensor(args.truncation))
                scored_field = torch.where(saturated, one_sided, signed)
            else:
                scored_field = signed
            surface = 1 + args.surface_weight * torch.exp(-target_field.abs() / args.surface_sigma)
            field = F.smooth_l1_loss(scored_field, target_field, reduction="none")
            deform = F.smooth_l1_loss(delta, asset["delta"][target_rows], reduction="none").mean(-1)
            field_loss = (field * weights * surface).sum() / (weights * surface).sum()
            delta_loss = (deform * weights).sum() / weights.sum()
            support_loss = torch.stack(occupancy).mean()
            validity = cell_validity(delta) if args.validity_weight else delta.sum() * 0
            loss = field_loss + delta_loss + support_loss + args.validity_weight * validity
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite G1 loss")
            loss.backward()
            groups = {
                "encoder": grad_status(model.encoder.parameters()),
                "decoder_backbone": grad_status(
                    p for n, p in model.decoder.backbone.named_parameters() if "cls_head" not in n and "conv_out" not in n),
                "support_heads": grad_status(
                    p for n, p in model.decoder.backbone.named_parameters() if "cls_head" in n),
                "continuous_head": grad_status(model.decoder.head.parameters()),
            }
            if not all(x["present"] and x["finite"] and x["norm"] > 0 for x in groups.values()):
                raise FloatingPointError(f"Missing, zero, or nonfinite G1 gradients: {groups}")
            optimizer.step()
            row = {"step": step + 1, "index": asset["index"],
                   "loss": float(loss.detach()), "field": float(field_loss.detach()),
                   "delta": float(delta_loss.detach()), "support": float(support_loss.detach()),
                   "validity": float(validity.detach()),
                   "kl_diagnostic": float(0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean().detach()),
                   "latent_sites": len(mean), "fine_cells": len(cells), "gradients": groups,
                   "elapsed_seconds": time.perf_counter() - started}
            report["trace"].append(row)
            write_json(args.output / "trace.json", report["trace"])
            print(json.dumps({k: v for k, v in row.items() if k != "gradients"}), flush=True)
            if args.save_every and (step + 1) % args.save_every == 0:
                snapshot_report = dict(report, status="completed", steps=step + 1)
                save_checkpoint(args.output / f"checkpoint_step_{step + 1:06d}.pt",
                                model, optimizer, step + 1, snapshot_report)
        report["status"] = "completed"
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        report["elapsed_seconds"] = time.perf_counter() - started
        save_checkpoint(args.output / "checkpoint.pt", model, optimizer,
                        start_step + args.steps, report)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        write_json(args.output / "resolved_config.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
