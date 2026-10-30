"""Differentiable flat and fan-aware smooth corner normals for route fitting."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def corner_fans_cpu(faces, vertex_count: int):
    """Reference grouping across manifold, consistently wound edges."""
    import torch

    array = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces)
    array = np.asarray(array, np.int64)
    if array.ndim != 2 or array.shape[1] != 3 or (array < 0).any() or (array >= vertex_count).any():
        raise ValueError("Invalid face index buffer")
    parent = np.arange(array.size, dtype=np.int64)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb

    incidence = defaultdict(list)
    for face_index, tri in enumerate(array):
        for u, v in ((0, 1), (1, 2), (2, 0)):
            a, b = int(tri[u]), int(tri[v])
            if a == b:
                continue
            incidence[(min(a, b), max(a, b))].append((face_index, u, v, a < b))
    for entries in incidence.values():
        if len(entries) != 2 or entries[0][3] == entries[1][3]:
            continue
        (fa, ua, va, _), (fb, ub, vb, _) = entries
        # The opposite directed edge is fb[vb] -> fb[ub].
        union(3 * fa + ua, 3 * fb + vb)
        union(3 * fa + va, 3 * fb + ub)
    labels = np.fromiter((root(i) for i in range(array.size)), dtype=np.int64, count=array.size)
    _, group = np.unique(labels, return_inverse=True)
    return torch.as_tensor(group, dtype=torch.long, device=faces.device if isinstance(faces, torch.Tensor) else None).reshape(-1, 3)


def corner_fans_gpu(faces, vertex_count: int):
    """GPU connected components of face corners, matching corner_fans_cpu."""
    import torch

    if not isinstance(faces, torch.Tensor) or not faces.is_cuda:
        raise ValueError("GPU corner fans require CUDA faces")
    if faces.ndim != 2 or faces.shape[1] != 3 or vertex_count < 1:
        raise ValueError("Invalid face index buffer")
    tri = faces.long()
    if bool(((tri < 0) | (tri >= vertex_count)).any()):
        raise ValueError("Face index out of range")
    corners = tri.numel()
    if not corners:
        return torch.empty_like(tri)
    u = tri.reshape(-1)
    v = tri[:, [1, 2, 0]].reshape(-1)
    valid = u != v
    occurrence = torch.arange(corners, device=faces.device)[valid]
    u, v = u[valid], v[valid]
    key = torch.minimum(u, v) * vertex_count + torch.maximum(u, v)
    order = torch.argsort(key)
    sorted_key = key[order]
    n = len(sorted_key)
    if n < 2:
        return torch.arange(corners, device=faces.device).reshape_as(tri)
    adjacent = sorted_key[:-1] == sorted_key[1:]
    starts = adjacent.clone()
    starts[1:] &= ~adjacent[:-1]
    starts[:-1] &= ~adjacent[1:]
    pair_at = torch.nonzero(starts).flatten()
    first, second = order[pair_at], order[pair_at + 1]
    opposite = (u[first] < v[first]) != (u[second] < v[second])
    first, second = occurrence[first[opposite]], occurrence[second[opposite]]
    first_v = first // 3 * 3 + (first % 3 + 1) % 3
    second_v = second // 3 * 3 + (second % 3 + 1) % 3
    edge_a = torch.cat((first, first_v))
    edge_b = torch.cat((second_v, second))
    labels = torch.arange(corners, device=faces.device)
    if not edge_a.numel():
        return labels.reshape_as(tri)
    for _ in range(256):
        next_labels = labels.clone()
        minimum = torch.minimum(labels[edge_a], labels[edge_b])
        next_labels.scatter_reduce_(0, edge_a, minimum, reduce="amin", include_self=True)
        next_labels.scatter_reduce_(0, edge_b, minimum, reduce="amin", include_self=True)
        next_labels = torch.minimum(next_labels, next_labels[next_labels])
        if torch.equal(next_labels, labels):
            return labels.reshape_as(tri)
        labels = next_labels
    raise RuntimeError("GPU corner fan labels did not converge")


def corner_fans(faces, vertex_count: int):
    """Select GPU grouping for CUDA faces, CPU reference otherwise."""
    import torch

    if isinstance(faces, torch.Tensor) and faces.is_cuda:
        return corner_fans_gpu(faces, vertex_count)
    return corner_fans_cpu(faces, vertex_count)


def corner_normals(vertices, faces, mode: str, fans=None):
    """Return [F,3,3] normals with gradient to vertices where defined."""
    import torch
    import torch.nn.functional as F

    if mode not in {"flat", "smooth"}:
        raise ValueError(f"Unknown normal mode: {mode}")
    tri = vertices[faces.long()]
    cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    face_normal = F.normalize(cross, dim=-1, eps=1e-8)
    if mode == "flat":
        return face_normal[:, None, :].expand(-1, 3, -1)
    if fans is None:
        fans = corner_fans(faces, len(vertices))
    if fans.shape != faces.shape:
        raise ValueError("Corner fan map differs from mesh faces")
    edge_a = tri[:, [1, 2, 0]] - tri
    edge_b = tri[:, [2, 0, 1]] - tri
    sin = torch.linalg.vector_norm(torch.cross(edge_a, edge_b, dim=-1), dim=-1)
    cos = (edge_a * edge_b).sum(-1)
    # atan2(0, 0) has an undefined backward. Degenerate triangles and zero
    # length corners are common in the fixed 50-mesh set; exclude them before
    # atan2 so torch.where cannot pass a NaN gradient through a dead branch.
    valid = (sin > 1e-12) & (torch.linalg.vector_norm(edge_a, dim=-1) > 1e-10) & (
        torch.linalg.vector_norm(edge_b, dim=-1) > 1e-10)
    safe_sin = torch.where(valid, sin, torch.ones_like(sin))
    safe_cos = torch.where(valid, cos, torch.zeros_like(cos))
    angle = torch.where(valid, torch.atan2(safe_sin, safe_cos), torch.zeros_like(sin))
    weighted = face_normal[:, None, :] * angle[..., None]
    group_count = int(fans.max().item()) + 1 if fans.numel() else 0
    sums = vertices.new_zeros(group_count, 3)
    sums.index_add_(0, fans.reshape(-1), weighted.reshape(-1, 3))
    return F.normalize(sums[fans], dim=-1, eps=1e-8)
