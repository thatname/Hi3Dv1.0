"""Export separate G0 A1/A2 initial targets, shared support, and raw MC meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from .preflight import VAE3D_COMMIT, sha256


SCHEMA = "g0_initial_target_v1"
CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))


def candidate_cells(field: np.ndarray) -> tuple[np.ndarray, int]:
    import numpy as np
    from scipy.ndimage import binary_dilation

    r = field.shape[0]
    low = np.full((r - 1,) * 3, np.inf, np.float32)
    high = np.full((r - 1,) * 3, -np.inf, np.float32)
    near = np.zeros((r - 1,) * 3, bool)
    for x, y, z in CORNERS:
        values = field[x:x + r - 1, y:y + r - 1, z:z + r - 1]
        np.minimum(low, values, out=low)
        np.maximum(high, values, out=high)
        near |= np.abs(values) < 3 / r
    crossing = (low < 0) & (high >= 0)
    if not crossing.any():
        raise ValueError("Initial field has no crossing cells")
    return binary_dilation(near | crossing, iterations=2), int(crossing.sum())


def complete_support(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    corners_array = np.asarray(CORNERS, dtype=np.int32)
    r = mask.shape[0] + 1
    initial = np.argwhere(mask).astype(np.int32)
    points = np.zeros((r,) * 3, bool)
    for off in CORNERS:
        p = initial + off
        points[tuple(p.T)] = True
    complete = np.ones((r - 1,) * 3, bool)
    for x, y, z in CORNERS:
        complete &= points[x:x + r - 1, y:y + r - 1, z:z + r - 1]
    cells = np.argwhere(complete).astype(np.int32)
    if not len(cells):
        raise ValueError("No complete support cells")
    corners = cells[:, None, :] + corners_array[None, :, :]
    flat = (corners[..., 0].astype(np.int64) * r + corners[..., 1]) * r + corners[..., 2]
    unique, inverse = np.unique(flat.reshape(-1), return_inverse=True)
    coords = np.column_stack(np.unravel_index(unique, (r, r, r))).astype(np.int32)
    indices = inverse.reshape(-1, 8).astype(np.int32)
    if not np.array_equal(coords[indices], corners):
        raise ValueError("Shared center-point indices are inconsistent")
    return cells, coords, indices


def raw_mesh(field: np.ndarray, resolution: int):
    import trimesh
    from skimage.measure import marching_cubes

    vertices, faces, _, _ = marching_cubes(
        field, level=0, spacing=(1 / resolution,) * 3, allow_degenerate=False
    )
    vertices += 0.5 / resolution - 0.5
    return trimesh.Trimesh(vertices, faces, process=False)


def run(args: argparse.Namespace) -> dict:
    import numpy as np

    from .initial_fields import query_initial_fields
    if args.resolution < 4:
        raise ValueError("resolution must be at least 4")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    assets = manifest["assets"]
    if not 0 <= args.index < len(assets):
        raise IndexError("index outside manifest")
    asset = assets[args.index]
    source = Path(asset["path"])
    if source.stat().st_size != asset["bytes"] or sha256(source) != asset["sha256"]:
        raise ValueError("Source mesh differs from the fixed manifest")
    head = subprocess.check_output(["git", "-C", str(args.vae3d_root), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != VAE3D_COMMIT:
        raise ValueError(f"vae3d must be {VAE3D_COMMIT}, found {head}")
    from vae3d.data import load_mesh

    started = time.perf_counter()
    mesh, center, scale = load_mesh(
        str(source), normalization_extent=(args.resolution - 2) / args.resolution
    )
    load_seconds = time.perf_counter() - started
    stages = ["A1", "A2"] if args.stage == "both" else [args.stage]
    started = time.perf_counter()
    result = query_initial_fields(mesh, args.resolution,
                                  raycast_backend=args.raycast_backend,
                                  distance_backend=args.distance_backend,
                                  shell="A2" in stages)
    query_seconds = time.perf_counter() - started
    reference = None
    if args.verify_reference and "A2" in stages:
        from vae3d.back_shell import back_shell_grid

        started = time.perf_counter()
        expected = back_shell_grid(mesh, args.resolution,
                                   raycast_backend=args.raycast_backend,
                                   distance_backend=args.distance_backend,
                                   sampling_offset=0.5)
        error = float(np.max(np.abs(expected - result["A2"])))
        reference = {"max_abs_error": error, "seconds": time.perf_counter() - started,
                     "equal": bool(np.array_equal(expected, result["A2"]))}
        if error > 1e-6:
            raise ValueError(f"A2 differs from pinned vae3d back_shell_grid: {error}")

    masks = {}
    crossing_counts = {}
    for stage in stages:
        masks[stage], crossing_counts[stage] = candidate_cells(result[stage])
    support_mask = np.logical_or.reduce([masks[stage] for stage in stages])
    cells, coords, indices = complete_support(support_mask)
    if len(coords) > args.max_sites:
        raise ValueError(f"Support has {len(coords)} points, exceeds max-sites {args.max_sites}")

    module_dir = Path(__file__).parent
    code_hashes = {name: sha256(module_dir / name)
                   for name in ("initial_fields.py", "prepare_initial_targets.py")}
    shared = {"schema": SCHEMA, "asset_id": asset["id"], "source": str(source),
              "source_sha256": asset["sha256"], "source_bytes": asset["bytes"],
              "manifest": str(args.manifest),
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "manifest_index": args.index, "vae3d_commit": head,
              "code_sha256": code_hashes, "resolution": args.resolution,
              "sampling_offset": 0.5, "raycast_backend": args.raycast_backend,
              "distance_backend": args.distance_backend,
              "normalization_center": center.tolist(), "normalization_scale": scale,
              "support_rule": "union of stage near/crossing cells, dilate 2, complete cells",
              "support_stages": stages, "max_sites": args.max_sites}
    output = {}
    for stage in stages:
        config = dict(shared, stage=stage)
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        destination = args.output_root / asset["id"] / f"R{args.resolution}" / f"{stage}-{key[:12]}"
        if destination.exists():
            raise FileExistsError(f"Cache already exists: {destination}")
        stage_field = result[stage]
        features = np.zeros((len(coords), 4), np.float32)
        features[:, 0] = np.clip(stage_field[tuple(coords.T)] * args.resolution, -4, 4)
        if not np.array_equal(coords[indices],
                              cells[:, None, :] + np.asarray(CORNERS)[None, :, :]):
            raise ValueError("Cell/point connectivity changed")
        surface = raw_mesh(stage_field, args.resolution)
        surface.vertices = np.asarray(surface.vertices) * scale + center
        destination.mkdir(parents=True)
        np.save(destination / "field.npy", stage_field)
        np.savez_compressed(destination / "target.npz", coords=coords, features=features,
                            cells=cells, cell_indices=indices,
                            resolution=np.int32(args.resolution), sampling_offset=np.float32(0.5),
                            stage=np.array(stage), field_layout=np.array("center_points_sdf_delta_v1"),
                            field_units=np.array("fine_voxel"), delta_units=np.array("fine_voxel"))
        surface.export(destination / "raw_mesh.ply")
        audit = {"stage": stage, "cache_key": key, "field_min": float(stage_field.min()),
                 "field_max": float(stage_field.max()),
                 "negative_samples": int((stage_field < 0).sum()),
                 "crossing_cells": crossing_counts[stage], "support_cells": len(cells),
                 "support_points": len(coords), "raw_mesh_vertices": len(surface.vertices),
                 "raw_mesh_faces": len(surface.faces), "load_seconds": load_seconds,
                 "query_seconds": query_seconds,
                 "reference_comparison": reference if stage == "A2" else None,
                 "query_audit": result[f"audit_{stage}"],
                 "raw_mesh_extractor": "skimage.measure.marching_cubes (Lewiner)",
                 "raw_mesh_coordinate_frame": "source",
                 "raw_mesh_postprocessing": "none"}
        (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        (destination / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
        output[stage] = {"cache": str(destination), "audit": audit}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("A1", "A2", "both"), default="both")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--raycast-backend", choices=("cpu", "dxr", "vulkan"), default="cpu")
    parser.add_argument("--distance-backend", choices=("cpu", "cuda", "rtx"), default="cpu")
    parser.add_argument("--max-sites", type=int, default=2_000_000)
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
