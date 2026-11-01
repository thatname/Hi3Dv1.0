"""Lossless A4 center-point to Hi3D per-cell field bridge for G0b.

This bypasses the network and Point.activate. The existing Hi3D Point layout
denormalizes deform by 0.5, so its packed channel carries 2*delta. The +0.5
sampling phase belongs in the fixed coordinate matrix, never in delta.
"""

from __future__ import annotations

import numpy as np


VAE_CORNERS = np.array(
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), dtype=np.int32)
HI3D_CORNERS = np.array(
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
     (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)), dtype=np.int32)
HI3D_FROM_VAE = np.array(
    [next(j for j, src in enumerate(VAE_CORNERS) if np.array_equal(src, dst))
     for dst in HI3D_CORNERS], dtype=np.int64)
VAE_FROM_HI3D = np.argsort(HI3D_FROM_VAE)


def rebuild_cell_indices(coords, cells, resolution: int) -> np.ndarray:
    """Recover the unique-point index of all eight cell corners.

    Cell corners are exactly grid points, so the mapping is a lookup and not an
    approximation: the packed cache can drop its 8-wide index array (a third of
    the file) and regenerate it whenever a target is loaded.
    """

    coords = np.asarray(coords, np.int64)
    cells = np.asarray(cells, np.int64)
    if (coords.ndim != 2 or coords.shape[1] != 3 or cells.ndim != 2
            or cells.shape[1] != 3 or len(coords) == 0 or len(cells) == 0):
        raise ValueError("Invalid coordinates or cells for index rebuild")
    if ((coords < 0).any() or (coords >= resolution).any()
            or (cells < 0).any() or (cells >= resolution - 1).any()):
        raise ValueError("Coordinates or cells outside the grid")

    def encode(points: np.ndarray) -> np.ndarray:
        return (points[:, 0] * resolution + points[:, 1]) * resolution + points[:, 2]

    keys = encode(coords)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Duplicate coordinates cannot be rebuilt into an index")
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    out = np.empty((len(cells), len(VAE_CORNERS)), np.int32)
    for corner, offset in enumerate(VAE_CORNERS):
        position = np.searchsorted(sorted_keys, encode(cells + offset))
        if position.max() >= len(sorted_keys):
            raise ValueError("A cell corner is missing from the support points")
        out[:, corner] = order[position]
    return out


def pack_target(coords, features, cells, cell_indices, resolution: int):
    """Return Hi3D cell coordinates/features and its original unique-point indices."""
    coords = np.asarray(coords, np.int32)
    features = np.asarray(features, np.float32)
    cells = np.asarray(cells, np.int32)
    if cell_indices is None:
        cell_indices = rebuild_cell_indices(coords, cells, resolution)
    cell_indices = np.asarray(cell_indices, np.int64)
    if (resolution < 2 or coords.ndim != 2 or coords.shape[1] != 3
            or cells.ndim != 2 or cells.shape[1] != 3
            or features.shape != (len(coords), 4)
            or cell_indices.shape != (len(cells), 8)
            or not np.isfinite(features).all()
            or len(coords) == 0 or len(cells) == 0):
        raise ValueError("Invalid unique center-point field or support")
    if ((coords < 0).any() or (coords >= resolution).any()
            or (cells < 0).any() or (cells >= resolution - 1).any()
            or (cell_indices < 0).any() or (cell_indices >= len(coords)).any()
            or not np.array_equal(coords[cell_indices], cells[:, None] + VAE_CORNERS)):
        raise ValueError("A4 cell_indices or coordinates are inconsistent")
    hi3d_indices = cell_indices[:, HI3D_FROM_VAE]
    corners = features[hi3d_indices]
    # Point.get_vert_feat() multiplies deform by 0.5. Bypass activate/tanh.
    hi3d_feat = np.concatenate(
        (corners[:, :, 0], (2.0 * corners[:, :, 1:]).reshape(-1, 24)), axis=1
    ).astype(np.float32)
    if not np.isfinite(hi3d_feat).all():
        raise ValueError("Nonfinite Hi3D packed features")
    return cells, hi3d_feat, hi3d_indices


def make_point(cells, hi3d_feat, resolution: int, device: str):
    """Build a Point using exact fine-grid coordinates and preactivated fields."""
    import torch
    from sparc3d.structures.point import Point

    grid = torch.as_tensor(cells, device=device, dtype=torch.int32)
    feat = torch.as_tensor(hi3d_feat, device=device, dtype=torch.float32)
    matrix = torch.eye(4, device=device, dtype=torch.float32)
    matrix[:3, :3] /= resolution
    matrix[:3, 3] = 0.5 / resolution - 0.5
    point = Point({
        "grid_coord": grid, "feat": feat, "sdf": None, "deform": None,
        "resolution": resolution, "matrix": matrix,
        "offset": torch.tensor([len(grid)], device=device, dtype=torch.int64),
    })
    point.initialization()
    if "sdf_sign" in point:
        raise ValueError("G0b direct field must not use a separate sign override")
    return point


def point_cell_fields(point):
    """Read Hi3D's denormalized corner values in vae3d corner order."""
    from sparc3d.structures.point import Point

    vert_feat, layouts = point.get_vert_feat(point.feat)
    sdf = Point.get_attr_from_feat(vert_feat, layouts, "sdf").squeeze(-1)
    delta = Point.get_attr_from_feat(vert_feat, layouts, "deform")
    return sdf[:, VAE_FROM_HI3D], delta[:, VAE_FROM_HI3D]


def fixed_coordinate_error(point, coords, features) -> float:
    """Maximum normalized-position error versus (ijk+0.5+delta)/R-0.5."""
    import torch
    from sparc3d.structures.point import Point

    grid = torch.as_tensor(coords, device=point.feat.device, dtype=torch.float32)
    delta = torch.as_tensor(features[:, 1:], device=point.feat.device, dtype=torch.float32)
    actual = Point.grid_to_coord(grid + delta, point.matrix)
    expected = (grid + 0.5 + delta) / point.resolution - 0.5
    return float((actual - expected).abs().max().item())
