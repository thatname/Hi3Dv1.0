"""Data-parallel multi-GPU trainer for the staged VAE experiments.

One asset per rank per step: rank r handles a different training asset, gradients
are averaged across ranks, so the effective batch becomes ``world_size`` assets
instead of one. Everything else (losses, optimiser, schedule, per-asset update
count) is identical to the single-GPU trainer in train_g1, which stays untouched.

The learning rate is deliberately NOT rescaled: runs record their world size and
effective batch so a larger-batch run is never confused with a single-GPU one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sparc3d.models.autoencoders.config import load_sparconv_vae

from .g1_model import G1VAE, _rows
from .g1_split import check_split
from .g1_targets import ROUTE, target_rows as select_target_rows
from .train_g1 import cell_validity, grad_status, load_asset, sha256, write_json

GPU_WHITELIST = (1, 2, 3, 7)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def worker(rank: int, world: int, gpus: list[int], args, port: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[rank])
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                                         rank=rank, world_size=world)
    device = "cuda:0"
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    rows = select_target_rows(args.paired_summary, manifest, four_route=True)
    assets = {i: load_asset(i, manifest, rows, manifest_hash, device, args.resolution)
              for i in args.indices}
    pretrained = load_sparconv_vae(args.weights, device=device)
    model = G1VAE(pretrained.decoder).to(device).train()
    for param in model.decoder.backbone.conv_out.parameters():
        param.requires_grad_(False)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0],
                                                      find_unused_parameters=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    start_step = 0
    prior_trace = []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        old = state["config"]
        if (old.get("schema") != "g1_four_route_v1" or old.get("status") != "completed"
                or old.get("manifest_sha256") != manifest_hash
                or old.get("paired_summary_sha256") != sha256(args.paired_summary)
                or old.get("pretrained_weights_sha256") != sha256(args.weights)
                or old.get("indices") != args.indices or int(state["steps"]) != old.get("steps")):
            raise ValueError("Resume checkpoint does not match this run")
        model.module.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["steps"])
        prior_trace = list(old.get("trace", []))
        rng = state.get("rng", {}).get(str(rank))
        if rng:
            torch.set_rng_state(rng["torch"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "g1_four_route_v1", "status": "running", "beta": 0,
        "stage": args.stage, "target_route": ROUTE,
        "teacher_forced_support": True, "indices": args.indices,
        "steps": start_step + args.steps, "additional_steps": args.steps,
        "start_step": start_step, "resolution": args.resolution,
        "world_size": world, "gpus": gpus, "effective_batch": world,
        "lr": args.lr, "lr_scaled_for_batch": False, "seed": args.seed,
        "manifest_sha256": manifest_hash,
        "surface_weight": args.surface_weight, "surface_sigma": args.surface_sigma,
        "truncation": args.truncation, "validity_weight": args.validity_weight,
        "paired_summary_sha256": sha256(args.paired_summary),
        "pretrained_weights": str(args.weights),
        "pretrained_weights_sha256": sha256(args.weights),
        "trace": prior_trace,
    }
    if rank == 0:
        write_json(args.output / "resolved_config.json", report)
    torch.distributed.barrier()
    started = time.perf_counter()
    for step in range(start_step, start_step + args.steps):
        index = args.indices[(step * world + rank) % len(args.indices)]
        asset = assets[index]
        optimizer.zero_grad(set_to_none=True)
        cells, signed, delta, occupancy, mean, logvar = model(
            asset["coords"], asset["features"], asset["cells"], args.resolution)
        target_rows = _rows(asset["cells"], cells, args.resolution)
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
            raise FloatingPointError(f"Nonfinite loss on rank {rank} at step {step + 1}")
        loss.backward()
        groups = {
            "encoder": grad_status(model.module.encoder.parameters()),
            "decoder_backbone": grad_status(
                p for n, p in model.module.decoder.backbone.named_parameters()
                if "cls_head" not in n and "conv_out" not in n),
            "support_heads": grad_status(
                p for n, p in model.module.decoder.backbone.named_parameters() if "cls_head" in n),
            "continuous_head": grad_status(model.module.decoder.head.parameters()),
        }
        if not all(x["present"] and x["finite"] and x["norm"] > 0 for x in groups.values()):
            raise FloatingPointError(f"Bad gradients on rank {rank}: {groups}")
        optimizer.step()
        row = {"step": step + 1, "rank": rank, "index": index,
               "loss": float(loss.detach()), "field": float(field_loss.detach()),
               "delta": float(delta_loss.detach()), "support": float(support_loss.detach()),
               "validity": float(validity.detach()),
               "kl_diagnostic": float(0.5 * (mean.square() + logvar.exp() - 1 - logvar)
                                      .mean().detach()),
               "latent_sites": len(mean), "fine_cells": len(cells),
               "elapsed_seconds": time.perf_counter() - started}
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, row)
        if rank == 0:
            report["trace"].extend(gathered)
            write_json(args.output / "trace.json", report["trace"])
            if (step + 1) % 20 == 0 or step + 1 == start_step + args.steps:
                print(json.dumps({"step": step + 1,
                                  "mean_loss": float(np.mean([g["loss"] for g in gathered])),
                                  "indices": [g["index"] for g in gathered]}), flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            save_snapshot(args, model, optimizer, step + 1, report, world, device,
                          f"checkpoint_step_{step + 1:06d}.pt")
    report["status"] = "completed"
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
    save_snapshot(args, model, optimizer, start_step + args.steps, report, world, device,
                  "checkpoint.pt")
    if rank == 0:
        write_json(args.output / "resolved_config.json", report)
    torch.distributed.destroy_process_group()


def save_snapshot(args, model, optimizer, step, report, world, device, name) -> None:
    local = {"torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all(),
             "numpy": np.random.get_state(),
             "python": random.getstate()}
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, local)
    if torch.distributed.get_rank() == 0:
        payload = {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(),
                   "steps": step, "config": dict(report, steps=step),
                   "rng": {str(rank): state for rank, state in enumerate(gathered)}}
        temporary = args.output / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(args.output / name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--paired-summary", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", type=Path)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=[7])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--surface-weight", type=float, default=4.0)
    p.add_argument("--surface-sigma", type=float, default=1.0)
    p.add_argument("--truncation", type=float, default=4.0)
    p.add_argument("--validity-weight", type=float, default=0.1)
    p.add_argument("--resolution", type=int, choices=(128, 256), default=128)
    p.add_argument("--stage", choices=("G1", "G2"), default="G2")
    p.add_argument("--resume", type=Path)
    p.add_argument("--seed", type=int, default=20261030)
    args = p.parse_args()
    if not args.indices or args.steps < 1 or len(set(args.indices)) != len(args.indices):
        p.error("Invalid indices or steps")
    if any(gpu not in GPU_WHITELIST for gpu in args.gpus) or len(set(args.gpus)) != len(args.gpus):
        p.error(f"--gpus must be distinct values from {GPU_WHITELIST}")
    if args.output.exists():
        p.error("Output exists; choose a new run directory to retain provenance")
    manifest = json.loads(args.manifest.read_text())
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if args.split:
        check_split(args.split, manifest, manifest_hash, "train", args.indices)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.multiprocessing.spawn(worker, args=(len(args.gpus), args.gpus, args, free_port()),
                                nprocs=len(args.gpus), join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())