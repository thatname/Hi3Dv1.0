"""Round-trip and spot-check a saved unverified A0 candidate cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def check(path: Path) -> dict:
    import numpy as np
    import open3d as o3d
    import trimesh
    from vae3d.center_points import complete_fitting_support
    from vae3d.data import load_mesh
    from .a0_signed_distance import scene_for
    from .prepare_initial_targets import candidate_cells

    config = json.loads((path / "resolved_config.json").read_text())
    audit = json.loads((path / "audit.json").read_text())
    if (config["stage"] != "A0"
            or config["sign_status"] not in {"candidate_unverified", "review_approved"}):
        raise ValueError("A0 cache has no valid sign-review label")
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if key != audit["cache_key"]:
        raise ValueError("Cache identity differs from resolved config")
    if hashlib.sha256(Path(config["a0_audit"]).read_bytes()).hexdigest() != config["a0_audit_sha256"]:
        raise ValueError("A0 audit changed after cache creation")
    field = np.load(path / "field.npy", allow_pickle=False)
    with np.load(path / "target.npz", allow_pickle=False) as values:
        target = {name: values[name] for name in values.files}
    n = config["resolution"]
    coords, features = target["coords"], target["features"]
    cells, indices = target["cells"], target["cell_indices"]
    if (field.shape != (n, n, n) or features.shape != (len(coords), 4)
            or str(target["field_layout"]) != "center_points_sdf_delta_v1"
            or str(target["field_units"]) != "fine_voxel"
            or str(target["delta_units"]) != "fine_voxel"
            or float(target["sampling_offset"]) != 0.5):
        raise ValueError("Invalid A0 target shape, layout or units")
    if (np.any(features[:, 1:] != 0) or
            not np.allclose(features[:, 0], np.clip(field[tuple(coords.T)] * n, -4, 4),
                            atol=1e-6)):
        raise ValueError("A0 point features differ from dense signed distance")
    mask, _ = candidate_cells(field)
    expected = complete_fitting_support(np.argwhere(mask).astype(np.int32), n)
    if not np.array_equal(cells, expected):
        raise ValueError("A0 support differs from pinned vae3d completion")
    corners = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                          (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))
    if not np.array_equal(coords[indices], cells[:, None, :] + corners[None, :, :]):
        raise ValueError("A0 shared point indices are inconsistent")
    surface = trimesh.load(path / "raw_mesh.ply", process=False)
    if len(surface.vertices) != audit["raw_mesh_vertices"] or len(surface.faces) != audit["raw_mesh_faces"]:
        raise ValueError("A0 raw mesh differs from audit")
    mesh, _, _ = load_mesh(
        config["source"], normalization_extent=(n - 2) / n
    )
    scene = scene_for(mesh)
    flat = np.linspace(0, field.size - 1, min(512, field.size), dtype=np.int64)
    ijk = np.column_stack(np.unravel_index(flat, field.shape)).astype(np.float32)
    points = (ijk + 0.5) / n - 0.5
    tensor = o3d.core.Tensor(points)
    unsigned = scene.compute_distance(tensor).numpy()
    occupancy = scene.compute_occupancy(tensor, nsamples=config["nsamples"]).numpy()
    sampled = field.reshape(-1)[flat]
    max_error = float(np.max(np.abs(np.abs(sampled) - unsigned)))
    clear = np.abs(sampled) > 2 / n
    disagreement = int(np.count_nonzero(((sampled < 0) != (occupancy > 0.5)) & clear))
    if max_error > 1e-4 or disagreement:
        raise ValueError(f"A0 distance/occupancy spot check failed: {max_error}, {disagreement}")
    return {"resolution": n, "support_cells": len(cells), "support_points": len(coords),
            "raw_mesh_faces": len(surface.faces), "distance_max_abs_error": max_error,
            "clear_sign_disagreements": disagreement, "sign_status": config["sign_status"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.cache)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
