"""Fit one A2 cache with pinned vae3d joint geometry refinement (A3)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from .preflight import VAE3D_COMMIT, sha256


SCHEMA = "g0_a3_geometry_target_v1"


def _extract(packed, resolution: int, center, scale, device):
    import numpy as np
    import torch
    import trimesh
    from vae3d.geometry import extract_mesh

    cells = np.asarray(packed["coords"], dtype=np.int32)
    features = np.asarray(packed["features"], dtype=np.float32)
    coordinates = np.column_stack((np.zeros(len(cells), np.int32), cells))
    with torch.no_grad():
        vertices, faces = extract_mesh(
            torch.as_tensor(coordinates, device=device),
            torch.as_tensor(features[:, :8], device=device),
            torch.as_tensor(features[:, 8:].reshape(-1, 8, 3), device=device),
            resolution, sampling_offset=0.5,
        )
    if len(faces) == 0:
        raise ValueError("Classic MC produced no faces")
    output = trimesh.Trimesh(
        vertices=vertices.cpu().numpy() * scale + center,
        faces=faces.cpu().numpy(), process=False,
    )
    if not np.isfinite(output.vertices).all():
        raise ValueError("Nonfinite extracted vertices")
    return output


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    import trimesh
    import kaolin
    from vae3d.data import load_mesh
    from vae3d.refine import refine, grid_quality
    from vae3d.center_points import deduplicate_fitted
    from .evaluate_initial_targets import mesh_metrics, sample_surface, scene_for

    if args.resolution < 4 or args.steps < 1 or args.fit_samples < 1 or args.metric_samples < 1:
        raise ValueError("Invalid resolution, steps, or sample count")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise ValueError("A3 requires a CUDA device with Kaolin gradients")
    device = torch.device(args.device)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    summary_bytes = args.a1a2_summary.read_bytes()
    summary = json.loads(summary_bytes)
    a0_bytes = args.a0_audit.read_bytes()
    a0 = json.loads(a0_bytes)
    assets = manifest["assets"]
    if (len(assets) != 50 or len(summary["assets"]) != 50
            or len(a0["assets"]) != 50 or not 0 <= args.index < 50):
        raise ValueError("Expected fixed balanced50, A1/A2, and A0 audit")
    asset = assets[args.index]
    row = summary["assets"][args.index]
    a0_row = a0["assets"][args.index]
    if (row["index"] != args.index or row["id"] != asset["id"]
            or row["status"] != "success" or row["source_sha256"] != asset["sha256"]
            or a0_row["index"] != args.index or a0_row["id"] != asset["id"]):
        raise ValueError("A1/A2 or A0 records differ from the fixed asset")
    source = Path(asset["path"])
    if source.stat().st_size != asset["bytes"] or sha256(source) != asset["sha256"]:
        raise ValueError("Source mesh differs from fixed manifest")
    head = subprocess.check_output(
        ["git", "-C", str(args.vae3d_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != VAE3D_COMMIT:
        raise ValueError(f"Wrong vae3d commit: {head}")
    a2_cache = Path(row["caches"]["A2"])
    a2_config = json.loads((a2_cache / "resolved_config.json").read_text())
    if (a2_config["stage"] != "A2" or a2_config["asset_id"] != asset["id"]
            or a2_config["source_sha256"] != asset["sha256"]
            or a2_config["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
            or a2_config["resolution"] != args.resolution
            or a2_config["support_stages"] != ["A1", "A2"]):
        raise ValueError("A2 cache does not match fixed batch protocol")
    with np.load(a2_cache / "target.npz", allow_pickle=False) as data:
        points = np.array(data["coords"], dtype=np.int32)
        initial = np.array(data["features"], dtype=np.float32)
        cells = np.array(data["cells"], dtype=np.int32)
        indices = np.array(data["cell_indices"], dtype=np.int32)
    if (initial.shape != (len(points), 4) or indices.shape != (len(cells), 8)
            or not np.array_equal(points[indices], cells[:, None, :] + np.asarray(
                ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                 (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))))):
        raise ValueError("A2 shared support is inconsistent")
    if np.any(initial[:, 1:] != 0):
        raise ValueError("A2 initial delta is not zero")

    mesh, center, scale = load_mesh(
        str(source), normalization_extent=(args.resolution - 2) / args.resolution
    )
    if (not np.allclose(center, a2_config["normalization_center"], rtol=0, atol=1e-8)
            or not np.isclose(scale, a2_config["normalization_scale"], rtol=0, atol=1e-8)):
        raise ValueError("A2 and A3 normalization differ")
    packed = {"coords": cells, "features": np.concatenate(
        (initial[indices, 0], np.zeros((len(cells), 24), np.float32)), axis=1),
        "vertices": np.asarray(mesh.vertices, np.float32),
        "faces": np.asarray(mesh.faces),
        "resolution": np.int32(args.resolution), "sampling_offset": np.float32(0.5)}

    module_dir = Path(__file__).parent
    config = {"schema": SCHEMA, "stage": "A3", "asset_id": asset["id"],
              "manifest_index": args.index, "source": str(source),
              "source_sha256": asset["sha256"],
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "a1a2_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
              "a0_audit_sha256": hashlib.sha256(a0_bytes).hexdigest(),
              "a0_status": a0_row["status"], "a0_concerns": a0_row.get("concerns", []),
              "topology_candidate": asset["topology_candidate"],
              "body_count": a0_row.get("body_count"),
              "a2_cache": str(a2_cache),
              "a2_target_sha256": sha256(a2_cache / "target.npz"),
              "vae3d_commit": head, "resolution": args.resolution,
              "sampling_offset": 0.5, "steps": args.steps,
              "fit_samples": args.fit_samples, "metric_samples": args.metric_samples,
              "seed": args.seed, "device": str(args.device), "max_deformation": 0.5,
              "mode": "joint", "extractor": "vae3d classic MC for A2 and A3",
              "torch_version": torch.__version__, "kaolin_version": kaolin.__version__,
              "code_sha256": {name: sha256(module_dir / name) for name in
                              ("prepare_a3_targets.py", "evaluate_initial_targets.py")}}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    destination = args.output_root / asset["id"] / f"R{args.resolution}" / f"A3-{key[:12]}"
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(f"A3 cache already exists or is incomplete: {destination}")

    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        fitted, trace = refine(
            packed, device=str(args.device), steps=args.steps,
            samples=args.fit_samples, max_deformation=0.5, mode="joint",
        )
    torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - started
    quality = grid_quality(fitted)
    unique = deduplicate_fitted(fitted)
    if (not np.array_equal(unique["coords"], points)
            or not np.array_equal(unique["cells"], cells)
            or not np.array_equal(unique["cell_indices"], indices)):
        raise ValueError("A3 support differs from A2 common support")
    final = np.asarray(unique["features"], np.float32)
    if (final.shape != initial.shape or not np.isfinite(final).all()
            or not np.array_equal(final[:, 0] < 0, initial[:, 0] < 0)
            or np.max(np.abs(final[:, 1:])) > 0.50001):
        raise ValueError("A3 field/delta failed sign, finite, or deformation checks")
    a2_classic = _extract(packed, args.resolution, center, scale, device)
    a3_raw = _extract(fitted, args.resolution, center, scale, device)
    reference = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices) * scale + center,
        faces=np.asarray(mesh.faces), process=False,
    )
    ref_points, ref_normals = sample_surface(reference, args.metric_samples, args.seed)
    ref_scene = scene_for(reference)
    metrics = {stage: mesh_metrics(
        output, reference, samples=args.metric_samples, seed=args.seed,
        threshold_fractions=(0.005, 0.01), reference_points=ref_points,
        reference_normals=ref_normals, reference_scene=ref_scene,
    ) for stage, output in (("A2_classic", a2_classic), ("A3", a3_raw))}
    audit = {"stage": "A3", "cache_key": key, "fit_seconds": fit_seconds,
             "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
             "support_cells": len(cells), "support_points": len(points),
             "sign_changes": int(np.count_nonzero((final[:, 0] < 0) != (initial[:, 0] < 0))),
             "changed_field_points": int(np.count_nonzero(final[:, 0] != initial[:, 0])),
             "nonzero_delta_points": int(np.count_nonzero(np.linalg.norm(final[:, 1:], axis=1) > 1e-7)),
             "max_abs_delta_voxels": float(np.max(np.abs(final[:, 1:]))),
             "grid_quality": quality, "a0_status": a0_row["status"],
             "a0_concerns": a0_row.get("concerns", []),
             "a2_classic_faces": len(a2_classic.faces), "a3_raw_faces": len(a3_raw.faces),
             "raw_mesh_coordinate_frame": "source",
             "raw_mesh_postprocessing": "none"}
    partial.mkdir(parents=True)
    np.savez_compressed(partial / "target.npz", coords=points, features=final,
                        cells=cells, cell_indices=indices,
                        resolution=np.int32(args.resolution), sampling_offset=np.float32(0.5),
                        stage=np.array("A3"), field_layout=np.array("center_points_sdf_delta_v1"),
                        field_units=np.array("fine_voxel"), delta_units=np.array("fine_voxel"))
    a2_classic.export(partial / "a2_classic_raw_mesh.ply")
    a3_raw.export(partial / "raw_mesh.ply")
    for name, obj in (("resolved_config.json", config), ("audit.json", audit),
                      ("trace.json", trace), ("metrics.json", metrics)):
        (partial / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    partial.rename(destination)
    return {"cache": str(destination), "audit": audit, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fit-samples", type=int, default=8192)
    parser.add_argument("--metric-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
