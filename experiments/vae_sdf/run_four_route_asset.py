"""Generate one independent A3/A4 route target from a fixed A2 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from .preflight import VAE3D_COMMIT, sha256


SCHEMA = "g0_four_route_r128_v1"
GPU_WHITELIST = (1, 2, 3, 7)


def physical_gpu() -> int:
    """Physical GPU this process was mapped onto; it never enters the cache key."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if value.strip() not in {str(card) for card in GPU_WHITELIST}:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES must name exactly one of {GPU_WHITELIST}, got {value!r}")
    return int(value.strip())
ROUTES = {
    "mc_flat": ("mc", "flat"), "mc_smooth": ("mc", "smooth"),
    "flexicubes_flat": ("flexicubes", "flat"),
    "flexicubes_smooth": ("flexicubes", "smooth"),
}


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _extract_mesh(sample, extractor, center, scale):
    import numpy as np
    import torch
    import trimesh
    from .four_route_a3 import RouteSurface

    with torch.no_grad():
        surface = RouteSurface(sample, extractor, "cuda:0")
        vertices, faces = surface.extract(surface.base.initial, surface.base.initial_delta)
    if not len(faces) or not torch.isfinite(vertices).all():
        raise RuntimeError(f"{extractor} extraction returned empty/nonfinite mesh")
    return trimesh.Trimesh(vertices=vertices.cpu().numpy() * scale + center,
                           faces=faces.cpu().numpy(), process=False)


def _deduplicate(packed, points, cells, indices, stage):
    import numpy as np
    from vae3d.center_points import deduplicate_fitted

    unique = deduplicate_fitted(packed)
    if (not np.array_equal(unique["coords"], points)
            or not np.array_equal(unique["cells"], cells)
            or not np.array_equal(unique["cell_indices"], indices)):
        raise RuntimeError(f"{stage} changed fixed A2 support")
    features = np.asarray(unique["features"], np.float32)
    if (features.shape != (len(points), 4) or not np.isfinite(features).all()
            or np.max(np.abs(features[:, 1:])) > .50001):
        raise RuntimeError(f"{stage} paired s/delta is invalid")
    return features


def _packed(features, cells, indices, normalized, resolution, phase):
    import numpy as np

    return {"coords": cells, "features": np.concatenate((
        features[indices, 0], features[indices, 1:].reshape(-1, 24)), axis=1),
        "vertices": np.asarray(normalized.vertices, np.float32),
        "faces": np.asarray(normalized.faces),
        "resolution": np.int32(resolution), "sampling_offset": np.float32(phase)}


def run(args):
    import numpy as np
    import torch
    import trimesh
    from vae3d.data import load_mesh
    from vae3d.refine import grid_quality
    from .four_route_a3 import fit_a3
    from .four_route_a4 import fit_a4
    from .evaluate_initial_targets import mesh_metrics, sample_surface, scene_for

    if args.device != "cuda:0" or physical_gpu() not in GPU_WHITELIST:
        raise ValueError(
            f"Four-route run needs one whitelisted physical GPU {GPU_WHITELIST} "
            "mapped to cuda:0 via CUDA_VISIBLE_DEVICES")
    if args.route not in ROUTES or args.resolution != 128 or args.a3_steps < 1 or args.a4_steps < 1:
        raise ValueError("Unsupported route or fit budget")
    if args.image_resolution < 32 or args.metric_samples < 1 or args.seed < 0:
        raise ValueError("Invalid render/evaluation configuration")
    extractor, normal_mode = ROUTES[args.route]
    manifest_bytes = args.manifest.read_bytes()
    summary_bytes = args.a1a2_summary.read_bytes()
    a0_bytes = args.a0_audit.read_bytes()
    assets = json.loads(manifest_bytes)["assets"]
    rows = json.loads(summary_bytes)["assets"]
    audits = json.loads(a0_bytes)["assets"]
    if (not assets or len(rows) != len(assets) or len(audits) != len(assets)
            or not 0 <= args.index < len(assets)):
        raise ValueError("Manifest, A1/A2 summary, and A0 audits must list the same assets")
    asset, row, a0 = assets[args.index], rows[args.index], audits[args.index]
    if (row["index"] != args.index or row["id"] != asset["id"]
            or row["status"] != "success" or row["source_sha256"] != asset["sha256"]
            or a0["id"] != asset["id"]):
        raise ValueError("Asset differs across fixed inputs")
    source = Path(asset["path"])
    if source.stat().st_size != asset["bytes"] or sha256(source) != asset["sha256"]:
        raise ValueError("Source mesh hash changed")
    a2_cache = Path(row["caches"]["A2"])
    a2_config = json.loads((a2_cache / "resolved_config.json").read_text())
    if (a2_config["asset_id"] != asset["id"] or a2_config["stage"] != "A2"
            or a2_config["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()):
        raise ValueError("A2 cache identity mismatch")
    with np.load(a2_cache / "target.npz", allow_pickle=False) as data:
        points = np.asarray(data["coords"], np.int32)
        initial = np.asarray(data["features"], np.float32)
        cells = np.asarray(data["cells"], np.int32)
        indices = np.asarray(data["cell_indices"], np.int32)
        resolution = int(data["resolution"])
        phase = float(data["sampling_offset"])
    if (resolution != args.resolution or phase != .5
            or initial.shape != (len(points), 4) or np.any(initial[:, 1:] != 0)):
        raise ValueError("Expected historical R128 phase=.5 A2 cache")
    normalized, center, scale = load_mesh(str(source), normalization_extent=(resolution - 2) / resolution)
    if (not np.allclose(center, a2_config["normalization_center"], rtol=0, atol=1e-8)
            or not np.isclose(scale, a2_config["normalization_scale"], rtol=0, atol=1e-8)):
        raise ValueError("A2/source normalization mismatch")
    reference = trimesh.Trimesh(vertices=np.asarray(normalized.vertices) * scale + center,
                                faces=np.asarray(normalized.faces), process=False)
    reference_points, reference_normals = sample_surface(reference, args.metric_samples, args.seed)
    reference_scene = scene_for(reference)
    module = Path(__file__).parent
    code_files = ("run_four_route_asset.py", "four_route_a3.py", "four_route_a4.py",
                  "four_route_normals.py", "g0b_field_adapter.py")
    config = {
        "schema": SCHEMA, "route_id": args.route, "extractor": extractor,
        "normal_mode": normal_mode, "index": args.index, "asset_id": asset["id"],
        "source": str(source), "source_sha256": asset["sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "a1a2_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "a0_audit_sha256": hashlib.sha256(a0_bytes).hexdigest(),
        "a0_status": a0["status"], "a0_concerns": a0.get("concerns", []),
        "a2_cache": str(a2_cache), "a2_target_sha256": sha256(a2_cache / "target.npz"),
        "resolution": resolution, "sampling_offset": phase,
        "normalization_center": np.asarray(center).tolist(), "normalization_scale": float(scale),
        "support_points": len(points), "support_cells": len(cells),
        "support_sha256": hashlib.sha256(points.tobytes() + cells.tobytes() + indices.tobytes()).hexdigest(),
        "a3_steps": args.a3_steps, "a3_samples": args.a3_samples,
        "a4_steps": args.a4_steps, "image_resolution": args.image_resolution,
        "views": 32, "held_out_views": 12, "seed": args.seed,
        "metric_samples": args.metric_samples, "delta_limit": .5,
        "allow_sign_changes_a4": True, "target_thickness_pixels": 1., "depth_weight": 10.,
        # The physical card is recorded in audit.json, never in this hashed config:
        # the same protocol must keep one cache identity whichever GPU ran it.
        "torch_version": torch.__version__,
        "hi3d_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "vae3d_commit": subprocess.check_output(["git", "-C", str(args.vae3d_root), "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {name: sha256(module / name) for name in code_files},
    }
    if config["vae3d_commit"] != VAE3D_COMMIT:
        raise ValueError("Four-route run requires the pinned vae3d source")
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    destination = args.output_root / asset["id"] / f"R{resolution}" / f"{args.route}-{key[:12]}"
    if destination.exists():
        if not args.resume or json.loads((destination / "resolved_config.json").read_text()) != config:
            raise FileExistsError("Existing route cache differs or --resume omitted")
    else:
        destination.mkdir(parents=True)
        _write_json(destination / "resolved_config.json", config)
    packed_a2 = _packed(initial, cells, indices, normalized, resolution, phase)
    output = {"cache": str(destination), "route_id": args.route, "index": args.index,
              "asset_id": asset["id"], "stages": {}}
    for stage in ("a3", "a4"):
        stage_dir = destination / stage
        if stage_dir.exists():
            if not args.resume:
                raise FileExistsError(stage_dir)
            with np.load(stage_dir / "target.npz", allow_pickle=False) as data:
                unique_features = np.asarray(data["features"], np.float32)
            if unique_features.shape != initial.shape or not np.isfinite(unique_features).all():
                raise ValueError(f"{stage} resume target invalid")
            output["stages"][stage] = {"status": "reused", "cache": str(stage_dir)}
            input_unique = unique_features
            continue
        partial = destination / f"{stage}.partial"
        if partial.exists():
            raise FileExistsError(f"Incomplete stage needs inspection: {partial}")
        input_packed = packed_a2 if stage == "a3" else _packed(unique_features, cells, indices, normalized, resolution, phase)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if stage == "a3":
            fitted, trace = fit_a3(input_packed, extractor, normal_mode, steps=args.a3_steps,
                                   samples=args.a3_samples, device=args.device, seed=args.seed)
            render_audit = None
        else:
            fitted, trace, render_audit = fit_a4(input_packed, extractor, normal_mode,
                steps=args.a4_steps, image_resolution=args.image_resolution,
                views=32, seed=args.seed, device=args.device)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        unique_features = _deduplicate(fitted, points, cells, indices, stage)
        if stage == "a3" and not np.array_equal(unique_features[:, 0] < 0, initial[:, 0] < 0):
            raise RuntimeError("A3 changed fixed A2 signs")
        raw_mesh = _extract_mesh(fitted, extractor, center, scale)
        metrics = mesh_metrics(raw_mesh, reference, samples=args.metric_samples, seed=args.seed,
            threshold_fractions=(.005, .01), reference_points=reference_points,
            reference_normals=reference_normals, reference_scene=reference_scene)
        audit = {"stage": stage.upper(), "route_id": args.route,
                 "physical_gpu": physical_gpu(), "logical_device": args.device,
                 "seconds": seconds, "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                 "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                 "grid_quality": grid_quality(fitted), "faces": len(raw_mesh.faces),
                 "max_abs_delta": float(np.abs(unique_features[:, 1:]).max()),
                 "sign_changes_from_input": int(np.count_nonzero((unique_features[:, 0] < 0) !=
                     ((initial if stage == "a3" else input_unique)[:, 0] < 0))),
                 "render": render_audit}
        partial.mkdir()
        np.savez_compressed(partial / "target.npz", coords=points, features=unique_features,
            cells=cells, cell_indices=indices, resolution=np.int32(resolution),
            sampling_offset=np.float32(phase), stage=np.array(stage.upper()),
            route_id=np.array(args.route), field_units=np.array("fine_voxel"),
            delta_units=np.array("fine_voxel"))
        raw_mesh.export(partial / "raw_mesh.ply")
        _write_json(partial / "trace.json", trace)
        _write_json(partial / "audit.json", audit)
        _write_json(partial / "metrics.json", metrics)
        partial.rename(stage_dir)
        output["stages"][stage] = {"status": "success", "cache": str(stage_dir),
                                   "audit": audit, "metrics": metrics}
        input_unique = unique_features
        print(json.dumps({"stage": stage, "route": args.route, "index": args.index,
                          "seconds": seconds, "faces": len(raw_mesh.faces),
                          "inverted_crossing_cells": audit["grid_quality"]["inverted_crossing_cells"]}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--route", choices=tuple(ROUTES), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--a3-steps", type=int, default=30)
    parser.add_argument("--a3-samples", type=int, default=8192)
    parser.add_argument("--a4-steps", type=int, default=200)
    parser.add_argument("--image-resolution", type=int, default=4096)
    parser.add_argument("--metric-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
