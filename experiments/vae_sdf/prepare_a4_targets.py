"""Fit one fixed G0 A3 cache with pinned vae3d consistent-depth rendering (A4)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from .preflight import VAE3D_COMMIT, sha256
from .prepare_a3_targets import _extract


SCHEMA = "g0_a4_render_target_v1"


def run(args: argparse.Namespace) -> dict:
    import nvdiffrast
    import numpy as np
    import torch
    import trimesh
    from vae3d.data import load_mesh
    from vae3d.center_points import deduplicate_fitted
    from vae3d.refine import grid_quality
    from vae3d.render_refine import fit
    from .evaluate_initial_targets import mesh_metrics, sample_surface, scene_for

    if (args.resolution != 128 or args.steps < 1 or args.image_resolution < 32
            or args.metric_samples < 1 or args.seed < 0):
        raise ValueError("Invalid A4 resolution, steps, render size, metric count, or seed")
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise ValueError("Pinned vae3d render fitting requires cuda:0")
    manifest_bytes = args.manifest.read_bytes()
    a3_bytes = args.a3_summary.read_bytes()
    a0_bytes = args.a0_audit.read_bytes()
    assets = json.loads(manifest_bytes)["assets"]
    a3_rows = json.loads(a3_bytes)["assets"]
    a0_rows = json.loads(a0_bytes)["assets"]
    if not (len(assets) == len(a3_rows) == len(a0_rows) == 50
            and 0 <= args.index < 50):
        raise ValueError("Expected fixed balanced50, A3 summary, and A0 audit")
    asset, a3_row, a0_row = (seq[args.index] for seq in (assets, a3_rows, a0_rows))
    if (a3_row["index"] != args.index or a3_row["id"] != asset["id"]
            or a3_row["status"] != "success"
            or a3_row["source_sha256"] != asset["sha256"]
            or a0_row["index"] != args.index or a0_row["id"] != asset["id"]
            or a3_row["a0_status"] != a0_row["status"]
            or a3_row["a0_concerns"] != a0_row.get("concerns", [])):
        raise ValueError("A3/A0 records differ from fixed manifest")
    source = Path(asset["path"])
    if source.stat().st_size != asset["bytes"] or sha256(source) != asset["sha256"]:
        raise ValueError("Source mesh differs from fixed manifest")
    head = subprocess.check_output(
        ["git", "-C", str(args.vae3d_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != VAE3D_COMMIT:
        raise ValueError(f"Wrong vae3d commit: {head}")
    raster_head = subprocess.check_output(
        ["git", "-C", str(args.nvdiffrast_root), "rev-parse", "HEAD"], text=True
    ).strip()
    a3_cache = Path(a3_row["cache"])
    a3_config = json.loads((a3_cache / "resolved_config.json").read_text())
    if (a3_config["stage"] != "A3" or a3_config["asset_id"] != asset["id"]
            or a3_config["source_sha256"] != asset["sha256"]
            or a3_config["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
            or a3_config["a0_audit_sha256"] != hashlib.sha256(a0_bytes).hexdigest()
            or a3_config["vae3d_commit"] != head
            or a3_config["resolution"] != args.resolution
            or a3_config["steps"] != 30 or a3_config["fit_samples"] != 8192
            or a3_config["sampling_offset"] != 0.5):
        raise ValueError("A3 cache differs from fixed A4 input protocol")
    with np.load(a3_cache / "target.npz", allow_pickle=False) as data:
        points = np.array(data["coords"], np.int32)
        original = np.array(data["features"], np.float32)
        cells = np.array(data["cells"], np.int32)
        indices = np.array(data["cell_indices"], np.int32)
    if (original.shape != (len(points), 4) or indices.shape != (len(cells), 8)
            or not np.isfinite(original).all()
            or np.max(np.abs(original[:, 1:])) > 0.50001):
        raise ValueError("Invalid A3 paired target")
    normalized, center, scale = load_mesh(
        str(source), normalization_extent=(args.resolution - 2) / args.resolution
    )
    a2_config = json.loads((Path(a3_config["a2_cache"]) / "resolved_config.json").read_text())
    if (a2_config["asset_id"] != asset["id"]
            or not np.allclose(center, a2_config["normalization_center"], rtol=0, atol=1e-8)
            or not np.isclose(scale, a2_config["normalization_scale"], rtol=0, atol=1e-8)):
        raise ValueError("A3 and A4 normalization differ")
    packed = {"coords": cells, "features": np.concatenate(
        (original[indices, 0], original[indices, 1:].reshape(-1, 24)), axis=1),
        "vertices": np.asarray(normalized.vertices, np.float32),
        "faces": np.asarray(normalized.faces),
        "resolution": np.int32(args.resolution), "sampling_offset": np.float32(0.5)}
    module_dir = Path(__file__).parent
    config = {
        "schema": SCHEMA, "stage": "A4", "asset_id": asset["id"],
        "manifest_index": args.index, "source": str(source),
        "source_sha256": asset["sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "a3_summary_sha256": hashlib.sha256(a3_bytes).hexdigest(),
        "a0_audit_sha256": hashlib.sha256(a0_bytes).hexdigest(),
        "a0_status": a0_row["status"], "a0_concerns": a0_row.get("concerns", []),
        "topology_candidate": asset["topology_candidate"],
        "body_count": a0_row.get("body_count"),
        "a3_cache": str(a3_cache), "a3_target_sha256": sha256(a3_cache / "target.npz"),
        "vae3d_commit": head, "nvdiffrast_commit": raster_head,
        "nvdiffrast_version": nvdiffrast.__version__,
        "torch_version": torch.__version__,
        "resolution": args.resolution, "sampling_offset": 0.5,
        "steps": args.steps, "image_resolution": args.image_resolution,
        "views": 32, "held_out_views": 12, "seed": args.seed,
        "metric_samples": args.metric_samples, "device": args.device,
        "render_backend": "cuda", "allow_sign_changes": True,
        "freeze_boundary": False, "max_deformation": 0.5,
        "target_thickness_pixels": 1.0, "depth_weight": 10.0,
        "supersample": 1, "extractor": "vae3d classic MC for A3 and A4",
        "code_sha256": {name: sha256(module_dir / name) for name in (
            "prepare_a4_targets.py", "prepare_a3_targets.py",
            "evaluate_initial_targets.py")},
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    destination = args.output_root / asset["id"] / f"R{args.resolution}" / f"A4-{key[:12]}"
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(f"A4 cache already exists or incomplete: {destination}")

    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    fitted, trace, render_audit = fit(
        packed, steps=args.steps, image_resolution=args.image_resolution,
        views=32, seed=args.seed, allow_sign_changes=True,
        supersample=1, direct_fixed_signs=False,
        max_deformation=0.5, target_thickness_pixels=1.0,
        depth_weight=10.0, return_surface=False,
        render_backend="cuda", freeze_boundary=False,
    )
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - started
    quality = grid_quality(fitted)
    unique = deduplicate_fitted(fitted)
    if (not np.array_equal(unique["coords"], points)
            or not np.array_equal(unique["cells"], cells)
            or not np.array_equal(unique["cell_indices"], indices)):
        raise ValueError("A4 support differs from A3")
    final = np.asarray(unique["features"], np.float32)
    if (final.shape != original.shape or not np.isfinite(final).all()
            or np.max(np.abs(final[:, 1:])) > 0.50001):
        raise ValueError("A4 paired field/delta invalid")
    a3_mesh = _extract(packed, args.resolution, center, scale, args.device)
    a4_mesh = _extract(fitted, args.resolution, center, scale, args.device)
    reference = trimesh.Trimesh(
        vertices=np.asarray(normalized.vertices) * scale + center,
        faces=np.asarray(normalized.faces), process=False,
    )
    ref_points, ref_normals = sample_surface(reference, args.metric_samples, args.seed)
    ref_scene = scene_for(reference)
    metrics = {stage: mesh_metrics(
        output, reference, samples=args.metric_samples, seed=args.seed,
        threshold_fractions=(0.005, 0.01), reference_points=ref_points,
        reference_normals=ref_normals, reference_scene=ref_scene,
    ) for stage, output in (("A3_classic", a3_mesh), ("A4", a4_mesh))}
    audit = {
        "stage": "A4", "cache_key": key, "fit_seconds": fit_seconds,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "support_cells": len(cells), "support_points": len(points),
        "sign_changes_from_a3": int(np.count_nonzero(
            (final[:, 0] < 0) != (original[:, 0] < 0))),
        "changed_field_points": int(np.count_nonzero(final[:, 0] != original[:, 0])),
        "nonzero_delta_points": int(np.count_nonzero(
            np.linalg.norm(final[:, 1:], axis=1) > 1e-7)),
        "max_abs_delta_voxels": float(np.max(np.abs(final[:, 1:]))),
        "grid_quality": quality, "render": render_audit,
        "a0_status": a0_row["status"], "a0_concerns": a0_row.get("concerns", []),
        "a3_raw_faces": len(a3_mesh.faces), "a4_raw_faces": len(a4_mesh.faces),
        "raw_mesh_coordinate_frame": "source", "raw_mesh_postprocessing": "none",
    }
    partial.mkdir(parents=True)
    np.savez_compressed(
        partial / "target.npz", coords=points, features=final,
        cells=cells, cell_indices=indices,
        resolution=np.int32(args.resolution), sampling_offset=np.float32(0.5),
        stage=np.array("A4"), field_layout=np.array("center_points_sdf_delta_v1"),
        field_units=np.array("fine_voxel"), delta_units=np.array("fine_voxel"),
    )
    a3_mesh.export(partial / "a3_classic_raw_mesh.ply")
    a4_mesh.export(partial / "raw_mesh.ply")
    for name, obj in (("resolved_config.json", config), ("audit.json", audit),
                      ("trace.json", trace), ("metrics.json", metrics)):
        (partial / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    partial.rename(destination)
    return {"cache": str(destination), "audit": audit, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a3-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--nvdiffrast-root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--image-resolution", type=int, default=4096)
    parser.add_argument("--metric-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
