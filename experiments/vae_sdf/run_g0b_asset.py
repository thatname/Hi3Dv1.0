"""Verify one A4 cache through Hi3D packing, classic MC, and FlexiCubes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from .g0b_field_adapter import (
    HI3D_FROM_VAE, fixed_coordinate_error, make_point, pack_target, point_cell_fields,
)
from .preflight import VAE3D_COMMIT, sha256


SCHEMA = "g0b_field_interface_v1"


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    import trimesh
    from vae3d.data import load_mesh
    from vae3d.geometry import extract_mesh
    from sparc3d.sparcubes.sparse_cube import sparse_cube2verts
    from sparc3d.sparcubes.extract import point_to_mesh
    from .evaluate_initial_targets import mesh_metrics, sample_surface, scene_for

    if args.metric_samples < 1 or not 0 <= args.index < 50:
        raise ValueError("Invalid metric count or index")
    if not torch.cuda.is_available() or args.device != "cuda:0":
        raise ValueError("G0b FlexiCubes run requires cuda:0")
    manifest_bytes = args.manifest.read_bytes()
    summary_bytes = args.a4_summary.read_bytes()
    assets = json.loads(manifest_bytes)["assets"]
    rows = json.loads(summary_bytes)["assets"]
    if len(assets) != 50 or len(rows) != 50:
        raise ValueError("Expected fixed 50-asset G0b input")
    asset, row = assets[args.index], rows[args.index]
    if (row["index"] != args.index or row["id"] != asset["id"]
            or row["status"] != "success"
            or row["source_sha256"] != asset["sha256"]):
        raise ValueError("A4 result differs from fixed manifest")
    source = Path(asset["path"])
    if source.stat().st_size != asset["bytes"] or sha256(source) != asset["sha256"]:
        raise ValueError("Source mesh differs from fixed manifest")
    vae3d_commit = subprocess.check_output(
        ["git", "-C", str(args.vae3d_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if vae3d_commit != VAE3D_COMMIT:
        raise ValueError(f"Wrong vae3d commit: {vae3d_commit}")
    hi3d_commit = subprocess.check_output(
        ["git", "-C", str(args.hi3d_root), "rev-parse", "HEAD"], text=True
    ).strip()
    a4_cache = Path(row["cache"])
    a4_config = json.loads((a4_cache / "resolved_config.json").read_text())
    if (a4_config["stage"] != "A4" or a4_config["asset_id"] != asset["id"]
            or a4_config["source_sha256"] != asset["sha256"]
            or a4_config["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
            or a4_config["resolution"] != 128 or a4_config["sampling_offset"] != 0.5):
        raise ValueError("A4 cache differs from G0b protocol")
    with np.load(a4_cache / "target.npz", allow_pickle=False) as data:
        coords = np.array(data["coords"], np.int32)
        features = np.array(data["features"], np.float32)
        cells = np.array(data["cells"], np.int32)
        indices = np.array(data["cell_indices"], np.int32)
        resolution = int(data["resolution"])
        phase = float(data["sampling_offset"])
    if (resolution != 128 or phase != 0.5 or np.max(np.abs(features[:, 1:])) > 0.50001):
        raise ValueError("A4 target has wrong resolution, phase, or delta units")
    normalized, center, scale = load_mesh(
        str(source), normalization_extent=(resolution - 2) / resolution
    )
    a3_config = json.loads((Path(a4_config["a3_cache"]) / "resolved_config.json").read_text())
    a2_config = json.loads((Path(a3_config["a2_cache"]) / "resolved_config.json").read_text())
    if (not np.allclose(center, a2_config["normalization_center"], rtol=0, atol=1e-8)
            or not np.isclose(scale, a2_config["normalization_scale"], rtol=0, atol=1e-8)):
        raise ValueError("A4/source normalization differs")
    module_dir = Path(__file__).parent
    hi3d_files = (
        "sparc3d/sparcubes/sparse_cube.py", "sparc3d/sparcubes/extract.py",
        "sparc3d/structures/point.py", "sparc3d/ops/flexicubes/flexicubes.py",
        "sparc3d/ops/flexicubes/tables.py",
    )
    config = {
        "schema": SCHEMA, "asset_id": asset["id"], "manifest_index": args.index,
        "source": str(source), "source_sha256": asset["sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "a4_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "a4_cache": str(a4_cache), "a4_target_sha256": sha256(a4_cache / "target.npz"),
        "a0_status": row["a0_status"], "a0_concerns": row["a0_concerns"],
        "a4_inverted_crossing_cells": row["audit"]["grid_quality"]["inverted_crossing_cells"],
        "vae3d_commit": vae3d_commit, "hi3d_commit": hi3d_commit,
        "hi3d_source_sha256": {name: sha256(args.hi3d_root / name) for name in hi3d_files},
        "code_sha256": {name: sha256(module_dir / name) for name in (
            "g0b_field_adapter.py", "run_g0b_asset.py", "evaluate_initial_targets.py")},
        "resolution": resolution, "sampling_offset": phase,
        "corner_order_hi3d_from_vae": HI3D_FROM_VAE.tolist(),
        "point_deform_raw_scale": 2.0,
        "matrix_diagonal": 1.0 / resolution,
        "matrix_translation": 0.5 / resolution - 0.5,
        "extractors": ["vae3d classic MC", "Hi3D FlexiCubes"],
        "metric_samples": args.metric_samples, "metric_seed": args.seed,
        "device": args.device, "torch_version": torch.__version__,
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    destination = args.output_root / asset["id"] / "R128" / f"G0b-{key[:12]}"
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(f"G0b cache already exists or incomplete: {destination}")

    started = time.perf_counter()
    hi3d_cells, hi3d_feat, hi3d_indices = pack_target(
        coords, features, cells, indices, resolution)
    point = make_point(hi3d_cells, hi3d_feat, resolution, args.device)
    field, delta = point_cell_fields(point)
    expected_s = torch.as_tensor(features[indices, 0], device=args.device)
    expected_d = torch.as_tensor(features[indices, 1:], device=args.device)
    field_error = float((field - expected_s).abs().max().item())
    delta_error = float((delta - expected_d).abs().max().item())
    phase_error = fixed_coordinate_error(point, coords, features)
    hi3d_corner_values, _ = point.get_vert_feat(point.feat)
    unique_coords, unique_values, hi3d_cubes = sparse_cube2verts(
        point.grid_coord, hi3d_corner_values)
    expected_coords = torch.as_tensor(coords, device=args.device)
    expected_values = torch.as_tensor(features, device=args.device)
    if not torch.equal(unique_coords, expected_coords):
        raise ValueError("Hi3D shared-vertex coordinates differ from A4 unique points")
    shared_error = float((unique_values - expected_values).abs().max().item())
    if max(field_error, delta_error, shared_error, phase_error) > 1e-6:
        raise ValueError("Hi3D field packing/coordinate conversion exceeded 1e-6")
    batch_coords = torch.cat((torch.zeros((len(cells), 1), device=args.device,
                                          dtype=torch.int32), point.grid_coord), dim=1)
    with torch.no_grad():
        direct_v, direct_f = extract_mesh(
            batch_coords, expected_s, expected_d, resolution, sampling_offset=phase)
        adapter_v, adapter_f = extract_mesh(
            batch_coords, field, delta, resolution, sampling_offset=phase)
        flexi = point_to_mesh(point)
    if (not torch.equal(direct_f, adapter_f)
            or direct_v.shape != adapter_v.shape
            or not torch.allclose(direct_v, adapter_v, rtol=0, atol=1e-7)):
        raise ValueError("Same-MC extraction changed after Hi3D packing")
    if not flexi.success:
        raise ValueError("Hi3D FlexiCubes produced an empty surface")
    mc_error = float((direct_v - adapter_v).abs().max().item())
    mc_raw = trimesh.Trimesh(
        vertices=adapter_v.cpu().numpy() * scale + center,
        faces=adapter_f.cpu().numpy(), process=False)
    flexi_raw = trimesh.Trimesh(
        vertices=flexi.vertices.cpu().numpy() * scale + center,
        faces=flexi.faces.cpu().numpy(), process=False)
    source_raw = trimesh.Trimesh(
        vertices=np.asarray(normalized.vertices) * scale + center,
        faces=np.asarray(normalized.faces), process=False)
    if (not np.isfinite(mc_raw.vertices).all()
            or not np.isfinite(flexi_raw.vertices).all()):
        raise ValueError("Nonfinite extracted vertices")
    ref_points, ref_normals = sample_surface(source_raw, args.metric_samples, args.seed)
    ref_scene = scene_for(source_raw)
    mc_points, mc_normals = sample_surface(mc_raw, args.metric_samples, args.seed)
    mc_scene = scene_for(mc_raw)
    metrics = {
        "MC_vs_source": mesh_metrics(
            mc_raw, source_raw, samples=args.metric_samples, seed=args.seed,
            threshold_fractions=(0.005, 0.01), reference_points=ref_points,
            reference_normals=ref_normals, reference_scene=ref_scene),
        "FlexiCubes_vs_source": mesh_metrics(
            flexi_raw, source_raw, samples=args.metric_samples, seed=args.seed,
            threshold_fractions=(0.005, 0.01), reference_points=ref_points,
            reference_normals=ref_normals, reference_scene=ref_scene),
        "FlexiCubes_vs_MC": mesh_metrics(
            flexi_raw, mc_raw, samples=args.metric_samples, seed=args.seed,
            threshold_fractions=(0.005, 0.01), reference_points=mc_points,
            reference_normals=mc_normals, reference_scene=mc_scene),
    }
    audit = {
        "asset_id": asset["id"], "a0_status": row["a0_status"],
        "a0_concerns": row["a0_concerns"],
        "a4_inverted_crossing_cells": row["audit"]["grid_quality"]["inverted_crossing_cells"],
        "center_points": len(coords), "cells": len(cells),
        "hi3d_shared_vertices": len(unique_coords),
        "hi3d_shared_cubes": len(hi3d_cubes),
        "field_error": field_error, "delta_error": delta_error,
        "shared_point_error": shared_error, "coordinate_error": phase_error,
        "same_mc_vertex_error": mc_error,
        "same_mc_faces_exact": True,
        "mc_raw_faces": len(mc_raw.faces), "flexicubes_raw_faces": len(flexi_raw.faces),
        "seconds": time.perf_counter() - started,
        "raw_mesh_coordinate_frame": "source", "raw_mesh_postprocessing": "none",
    }
    partial.mkdir(parents=True)
    mc_raw.export(partial / "mc_raw_mesh.ply")
    flexi_raw.export(partial / "flexicubes_raw_mesh.ply")
    for name, value in (("resolved_config.json", config), ("audit.json", audit),
                        ("metrics.json", metrics)):
        (partial / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    partial.rename(destination)
    return {"cache": str(destination), "audit": audit, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a4-summary", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--hi3d-root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metric-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
