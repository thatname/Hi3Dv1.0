"""Evaluate one G1 checkpoint with GT and predicted child support on raw meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from sparc3d.models.autoencoders.config import load_sparconv_vae
from vae3d.data import load_mesh
from vae3d.geometry import extract_mesh

from .evaluate_initial_targets import mesh_metrics, sample_surface, scene_for
from .g0b_field_adapter import VAE_FROM_HI3D
from .g1_model import G1VAE, _keys, _rows
from .train_g1 import load_asset, sha256, write_json
from .g1_split import check_split


def _raw_mesh(cells, signed, delta, center, scale, resolution):
    import trimesh

    if len(cells) == 0:
        return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), np.int64),
                               process=False)
    packed = torch.cat((torch.zeros((len(cells), 1), device=cells.device, dtype=torch.int32),
                        cells.int()), 1)
    vertices, faces = extract_mesh(
        packed, signed[:, VAE_FROM_HI3D], delta[:, VAE_FROM_HI3D],
        resolution, sampling_offset=.5)
    return trimesh.Trimesh(vertices=vertices.cpu().numpy() * scale + center,
                           faces=faces.cpu().numpy(), process=False)


def _support_stats(predicted, expected, resolution):
    pred_keys = _keys(predicted, resolution)
    target_keys = _keys(expected, resolution)
    overlap = int(torch.isin(pred_keys, target_keys).sum()) if len(pred_keys) else 0
    return {"predicted_cells": len(predicted), "target_cells": len(expected),
            "overlap_cells": overlap,
            "precision": overlap / len(predicted) if len(predicted) else 0.0,
            "recall": overlap / len(expected) if len(expected) else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a4-summary", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[48, 5])
    parser.add_argument("--split", type=Path, help="Fixed asset-level split")
    parser.add_argument("--role", choices=("train", "val"), default="val")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--max-predicted-sites", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.samples < 1 or args.max_predicted_sites < 1 or args.output.exists():
        parser.error("Invalid samples/site limit or output already exists")
    if not torch.cuda.is_available():
        parser.error("G1 checkpoint evaluation needs CUDA for the Hi3D backbone")
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    rows = json.loads(args.a4_summary.read_text())["assets"]
    split_hash = check_split(args.split, manifest, manifest_hash, args.role, args.indices) if args.split else None
    state = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    config = state["config"]
    if (config.get("schema") != "g1_a4_overfit_v2"
            or config.get("status") != "completed"
            or config.get("manifest_sha256") != manifest_hash
            or config.get("a4_summary_sha256") != sha256(args.a4_summary)
            or config.get("pretrained_weights_sha256") != sha256(args.weights)
            or (split_hash is not None and config.get("split_sha256") != split_hash)
            or int(state["steps"]) != config.get("steps")):
        raise ValueError("G1 checkpoint or target provenance mismatch")
    model = G1VAE(load_sparconv_vae(args.weights, device="cuda:0").decoder).to("cuda:0")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    del state
    args.output.mkdir(parents=True)
    report = {
        "schema": "g1_reconstruction_eval_v1", "status": "running",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": config["steps"], "training_run": str(args.checkpoint.parent),
        "manifest_sha256": manifest_hash, "a4_summary_sha256": sha256(args.a4_summary),
        "weights_sha256": sha256(args.weights), "samples": args.samples, "seed": args.seed,
        "split": str(args.split) if args.split else None,
        "split_sha256": split_hash, "split_role": args.role if args.split else None,
        "max_predicted_sites": args.max_predicted_sites,
        "support_condition": "GT coarse latent coordinates; compare GT versus predicted child support",
        "mesh_protocol": "vae3d classic MC, sampling_offset=0.5; original coordinates; no cleanup or renormalization",
        "assets": [],
    }
    write_json(args.output / "report.json", report)
    for index in args.indices:
        started = time.perf_counter()
        asset = load_asset(index, manifest, rows, manifest_hash, "cuda:0")
        source_path = Path(manifest["assets"][index]["path"])
        if sha256(source_path) != manifest["assets"][index]["sha256"]:
            raise ValueError(f"Source hash changed at index {index}")
        normalized, center, scale = load_mesh(
            str(source_path), normalization_extent=(128 - 2) / 128)
        import trimesh
        source = trimesh.Trimesh(vertices=np.asarray(normalized.vertices) * scale + center,
                                 faces=np.asarray(normalized.faces), process=False)
        target = trimesh.load(Path(asset["cache"]) / "raw_mesh.ply", process=False)
        ref_points, ref_normals = sample_surface(source, args.samples, args.seed + index)
        ref_scene = scene_for(source)
        target_points, target_normals = sample_surface(target, args.samples, args.seed + index)
        target_scene = scene_for(target)
        record = {key: asset[key] for key in ("index", "id", "a0_status", "a0_concerns",
                                                  "inverted_crossing_cells", "target_sha256")}
        record["target_vs_source"] = mesh_metrics(
            target, source, samples=args.samples, seed=args.seed + index,
            threshold_fractions=(.005, .01), reference_points=ref_points,
            reference_normals=ref_normals, reference_scene=ref_scene)
        with torch.inference_mode():
            for name, predicted in (("gt_support", False), ("predicted_support", True)):
                try:
                    cells, signed, delta, _, mean, logvar = model(
                        asset["coords"], asset["features"], asset["cells"], 128,
                        predicted_support=predicted, max_sites=args.max_predicted_sites)
                    support = _support_stats(cells, asset["cells"], 128)
                    entry = {"status": "success", "support": support,
                             "latent_sites": len(mean),
                             "kl_diagnostic": float(0.5 * (
                                 mean.square() + logvar.exp() - 1 - logvar).mean())}
                    if not predicted:
                        target_rows = _rows(asset["cells"], cells, 128)
                        weight = asset["weight"][target_rows]
                        field_error = (signed - asset["field"][target_rows]).abs()
                        delta_error = (delta - asset["delta"][target_rows]).abs().mean(-1)
                        entry["field_mae_unique_weighted"] = float((field_error * weight).sum() / weight.sum())
                        entry["delta_mae_unique_weighted"] = float((delta_error * weight).sum() / weight.sum())
                    mesh = _raw_mesh(cells, signed, delta, center, scale, 128)
                    entry["mesh_file"] = f"{index:02d}_{name}_raw.ply"
                    mesh.export(args.output / entry["mesh_file"])
                    entry["vs_source"] = mesh_metrics(
                        mesh, source, samples=args.samples, seed=args.seed + index,
                        threshold_fractions=(.005, .01), reference_points=ref_points,
                        reference_normals=ref_normals, reference_scene=ref_scene)
                    entry["vs_target"] = mesh_metrics(
                        mesh, target, samples=args.samples, seed=args.seed + index,
                        threshold_fractions=(.005, .01), reference_points=target_points,
                        reference_normals=target_normals, reference_scene=target_scene)
                    record[name] = entry
                except (RuntimeError, ValueError) as error:
                    record[name] = {"status": "failed", "error": repr(error)}
        record["seconds"] = time.perf_counter() - started
        report["assets"].append(record)
        write_json(args.output / "report.json", report)
        print(json.dumps({"index": index, "seconds": record["seconds"],
                          "gt": record["gt_support"]["status"],
                          "predicted": record["predicted_support"]["status"]}), flush=True)
    report["status"] = "completed"
    write_json(args.output / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
