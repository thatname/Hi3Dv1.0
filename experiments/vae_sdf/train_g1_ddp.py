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
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sparc3d.models.autoencoders.config import load_sparconv_vae

from .g1_model import G1VAE, _rows
from .g1_split import check_split, check_split_subset
from .g1_targets import ROUTE, target_rows as select_target_rows
from .streaming_targets import LazyTargets
from .train_g1 import (cell_validity, grad_status, load_asset, sha256, upload,
                       write_json)

GPU_WHITELIST = (0, 1, 2, 3, 4, 5, 6, 7)
SCHEMA = "g1_four_route_v1"
CODE_FILES = ("train_g1_ddp.py", "train_g1.py", "g1_model.py", "g0b_field_adapter.py")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def checkpoint_payload(args, model, optimizer, step, report, world, device) -> dict:
    """Gather every rank's RNG so a resumed run replays the same asset order."""
    local = {"torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all(),
             "numpy": np.random.get_state(),
             "python": random.getstate()}
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, local)
    return {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(),
            "steps": step, "config": dict(report, steps=step),
            "rng": {str(rank): state for rank, state in enumerate(gathered)}}


def save_snapshot(args, model, optimizer, step, report, world, device, name) -> None:
    payload = checkpoint_payload(args, model, optimizer, step, report, world, device)
    if torch.distributed.get_rank() == 0:
        temporary = args.output / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(args.output / name)


def rank_problems(loss, groups) -> list[str]:
    problems = []
    if not bool(torch.isfinite(loss)):
        problems.append("nonfinite loss")
    bad = {name: group for name, group in groups.items()
           if not (group["present"] and group["finite"] and group["norm"] > 0)}
    if bad:
        problems.append(f"bad gradients: {bad}")
    return problems


def abort_if_any_rank_fails(problems: list[str], step: int, world: int, device: str) -> None:
    """All ranks stop together: a one-sided raise would strand the others in the
    next collective until the NCCL watchdog kills the job."""
    flag = torch.tensor([0.0 if problems else 1.0], device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    if float(flag.item()) == 1.0:
        return
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, problems)
    report = {f"rank{rank}": items for rank, items in enumerate(gathered) if items}
    raise FloatingPointError(f"Step {step} aborted on {max(report)}: {report}")


def worker(rank: int, world: int, gpus: list[int], args, port: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[rank])
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                                         rank=rank, world_size=world)
    device = "cuda:0"
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    rows = None if args.targets_root else select_target_rows(args.paired_summary, manifest,
                                                           four_route=True)
    summary_hash = sha256(args.paired_summary) if args.paired_summary else None
    indices = args.indices
    split_hash = (None if not args.split else
                  (check_split_subset if args.targets_root else check_split)
                  (args.split, manifest, manifest_hash, "train", indices))
    if args.targets_root:
        # the pool keeps growing: freeze the finished set at launch so both arms
        # replay the same assets in the same order
        wanted = indices
        if not wanted and args.split:
            wanted = json.loads(args.split.read_text(encoding="utf-8"))["train_indices"]
        assets = LazyTargets(args.targets_root, manifest, manifest_hash, args.resolution,
                             device, args.asset_device, wanted,
                             max_resident=args.stream_resident, seed=args.seed)
        indices = assets.indices
    else:
        assets = {i: load_asset(i, manifest, rows, manifest_hash, device, args.resolution,
                                args.asset_device)
                  for i in indices}
    pretrained = load_sparconv_vae(args.weights, device=device)
    model = G1VAE(pretrained.decoder, args.encoder_scale).to(device).train()
    for param in model.decoder.backbone.conv_out.parameters():
        param.requires_grad_(False)
    # Some steps reach the row-level loss with a head that contributed nothing, so
    # unused-parameter detection stays on even though it costs a graph traversal.
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0],
                                                      find_unused_parameters=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr)
    start_step = 0
    prior_trace = []
    rng_restored = False
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        old = state["config"]
        expected = {
            "schema": SCHEMA, "stage": args.stage, "encoder_scale": args.encoder_scale, "indices": indices,
            "manifest_sha256": manifest_hash, "paired_summary_sha256": summary_hash,
            "pretrained_weights_sha256": sha256(args.weights), "lr": args.lr,
            "surface_weight": args.surface_weight, "surface_sigma": args.surface_sigma,
            "truncation": args.truncation, "validity_weight": args.validity_weight,
            "resolution": args.resolution, "beta": 0, "teacher_forced_support": True,
            "split_sha256": split_hash, "world_size": world,
            "asset_device": args.asset_device,
            "lr_scaled_for_batch": False,
        }
        mismatch = [key for key, value in expected.items() if old.get(key) != value]
        if mismatch:
            raise ValueError(f"DDP checkpoint configuration mismatch: {mismatch}; the asset "
                             f"order depends on world_size, so resuming needs the same ranks")
        if old.get("status") != "completed":
            raise ValueError("DDP checkpoint is not a completed run")
        model.module.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["steps"])
        prior_trace = list(old.get("trace", []))
        if len(prior_trace) != start_step * world:
            raise ValueError("DDP checkpoint step and trace length differ")
        saved = state.get("rng", {}).get(str(rank))
        if saved:
            torch.set_rng_state(saved["torch"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda"]])
            np.random.set_state(saved["numpy"])
            random.setstate(saved["python"])
            rng_restored = True
    # Only rank 0 persists the config, so the restored-rank list must be gathered.
    rng_flags = [None] * world
    torch.distributed.all_gather_object(rng_flags, rng_restored)
    restored_ranks = [rank for rank, flag in enumerate(rng_flags) if flag]
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": SCHEMA, "status": "running", "beta": 0,
        "stage": args.stage, "target_route": ROUTE, "teacher_forced_support": True,
        "indices": indices, "steps": start_step + args.steps,
        "additional_steps": args.steps, "start_step": start_step,
        "resume_from": str(args.resume) if args.resume else None,
        "rng_restored_ranks": sorted(restored_ranks), "save_every": args.save_every,
        "split": str(args.split) if args.split else None, "split_sha256": split_hash,
        "resolution": args.resolution, "asset_device": args.asset_device,
        "world_size": world, "gpus": gpus, "effective_batch": world,
        "ladder": "one asset per rank per step; gradients averaged over ranks",
        "lr": args.lr, "lr_scaled_for_batch": False, "seed": args.seed,
        "manifest_sha256": manifest_hash, "surface_weight": args.surface_weight,
        "surface_sigma": args.surface_sigma, "truncation": args.truncation,
        "validity_weight": args.validity_weight,
        "paired_summary_sha256": summary_hash,
        "target_summary": str(args.paired_summary) if args.paired_summary else None,
        "pretrained_weights": str(args.weights),
        "pretrained_weights_sha256": sha256(args.weights),
        "pretrained_loading": "strict: all released Hi3D decoder parameters; old output head "
                              "frozen/unused; new head random",
        "encoder_initialization": "random; adapted vae3d Encoder structure, factor 8/latent 32",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {name: sha256(Path(__file__).parent / name) for name in CODE_FILES},
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "assets": ([] if args.targets_root else
                   [{k: v for k, v in assets[i].items() if not isinstance(v, torch.Tensor)}
                    for i in indices]),
        "target_stream": assets.describe() if args.targets_root else None,
        "trace": prior_trace,
    }
    if rank == 0:
        write_json(args.output / "resolved_config.json", report)
    torch.distributed.barrier()
    started = time.perf_counter()
    try:
        for step in range(start_step, start_step + args.steps):
            index = indices[(step * world + rank) % len(indices)]
            asset = upload(assets[index], device)
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
            deform = F.smooth_l1_loss(delta, asset["delta"][target_rows],
                                      reduction="none").mean(-1)
            field_loss = (field * weights * surface).sum() / (weights * surface).sum()
            delta_loss = (deform * weights).sum() / weights.sum()
            support_loss = torch.stack(occupancy).mean()
            validity = cell_validity(delta) if args.validity_weight else delta.sum() * 0
            loss = field_loss + delta_loss + support_loss + args.validity_weight * validity
            loss.backward()
            groups = {
                "encoder": grad_status(model.module.encoder.parameters()),
                "decoder_backbone": grad_status(
                    p for n, p in model.module.decoder.backbone.named_parameters()
                    if "cls_head" not in n and "conv_out" not in n),
                "support_heads": grad_status(
                    p for n, p in model.module.decoder.backbone.named_parameters()
                    if "cls_head" in n),
                "continuous_head": grad_status(model.module.decoder.head.parameters()),
            }
            abort_if_any_rank_fails(rank_problems(loss, groups), step + 1, world, device)
            optimizer.step()
            row = {"step": step + 1, "rank": rank, "index": index,
                   "loss": float(loss.detach()), "field": float(field_loss.detach()),
                   "delta": float(delta_loss.detach()), "support": float(support_loss.detach()),
                   "validity": float(validity.detach()),
                   "kl_diagnostic": float(0.5 * (mean.square() + logvar.exp() - 1 - logvar)
                                          .mean().detach()),
                   "latent_sites": len(mean), "fine_cells": len(cells), "gradients": groups,
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
                snapshot = dict(report, status="completed", steps=step + 1)
                save_snapshot(args, model, optimizer, step + 1, snapshot, world, device,
                              f"checkpoint_step_{step + 1:06d}.pt")
        report["status"] = "completed"
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        report["elapsed_seconds"] = time.perf_counter() - started
        peaks = [None] * world
        torch.distributed.all_gather_object(peaks,
                                            torch.cuda.max_memory_allocated(0))
        report["peak_cuda_allocated_bytes_per_rank"] = [int(value) for value in peaks]
        save_snapshot(args, model, optimizer, start_step + args.steps, report, world, device,
                      "checkpoint.pt")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        if rank == 0:
            write_json(args.output / "resolved_config.json", report)
        raise
    finally:
        if rank == 0:
            write_json(args.output / "resolved_config.json", report)
        torch.distributed.destroy_process_group()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--paired-summary", type=Path)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", type=Path)
    p.add_argument("--indices", type=int, nargs="+")
    p.add_argument("--targets-root", type=Path,
                   help="Stream finished A4 targets instead of a fixed paired summary")
    p.add_argument("--stream-resident", type=int, default=1,
                   help="How many streamed targets stay resident per rank")
    p.add_argument("--gpus", type=int, nargs="+", default=[7])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--surface-weight", type=float, default=4.0)
    p.add_argument("--surface-sigma", type=float, default=1.0)
    p.add_argument("--truncation", type=float, default=4.0)
    p.add_argument("--validity-weight", type=float, default=0.1)
    p.add_argument("--resolution", type=int, choices=(128, 256), default=128)
    p.add_argument("--asset-device", choices=("cuda", "cpu"), default="cuda",
                   help="Where each rank keeps the cached target tensors between steps")
    p.add_argument("--encoder-scale", default="E0",
                   choices=("E0", "E1", "E2", "E3", "E4"),
                   help="Encoder width ladder; the latent width stays 32 so the"
                        " released decoder weights still load")
    p.add_argument("--stage", choices=("G1", "G2"), default="G2")
    p.add_argument("--resume", type=Path)
    p.add_argument("--seed", type=int, default=20261030)
    args = p.parse_args()
    if ((not args.indices and not args.targets_root) or args.steps < 1
            or args.save_every < 0 or args.lr <= 0
            or len(set(args.indices or [])) != len(args.indices or [])
            or args.surface_weight < 0 or args.surface_sigma <= 0
            or args.truncation < 0 or args.validity_weight < 0):
        p.error("Invalid indices, steps, learning rate, or loss weights")
    if any(gpu not in GPU_WHITELIST for gpu in args.gpus) or len(set(args.gpus)) != len(args.gpus):
        p.error(f"--gpus must be distinct values from {GPU_WHITELIST}")
    args.indices = args.indices or []
    if args.indices and len(args.indices) < len(args.gpus):
        p.error("Every rank needs its own asset: --indices must be at least as long as --gpus")
    if args.output.exists():
        p.error("Output exists; choose a new run directory to retain provenance")
    if not torch.cuda.is_available():
        p.error("DDP Hi3D/spconv training requires CUDA")
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if (args.paired_summary is not None and args.output.resolve()
            .is_relative_to(args.paired_summary.resolve().parent.parent)):
        p.error("Four-route output must be outside the G0 preparation directory")
    if args.split and not args.targets_root:
        check_split(args.split, manifest, manifest_hash, "train", args.indices)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.multiprocessing.spawn(worker, args=(len(args.gpus), args.gpus, args, free_port()),
                                nprocs=len(args.gpus), join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())