"""Separate A1 UE50 votes and A2 vote-gated shell on the same query grid.

The A2 equations mirror vae3d.back_shell.back_shell_grid at the pinned
reference revision. Its Queries and direction sampler are imported directly;
the local implementation exposes the A1 field before the shell mutation.
"""

from __future__ import annotations

import time

import numpy as np


def query_initial_fields(mesh, resolution: int, *, raycast_backend: str = "cpu",
                         distance_backend: str = "cpu", shell: bool = True) -> dict:
    from vae3d.back_shell import Queries
    from vae3d.ue_sign import directions

    if resolution < 4:
        raise ValueError("resolution must be at least 4")
    if raycast_backend not in {"cpu", "dxr", "vulkan"}:
        raise ValueError("Unsupported raycast backend")
    if distance_backend not in {"cpu", "cuda", "rtx"}:
        raise ValueError("Unsupported distance backend")
    if distance_backend == "rtx" and raycast_backend == "cpu":
        raise ValueError("RTX distance requires DXR/Vulkan rays")
    if np.abs(mesh.vertices).max() >= 0.5:
        raise ValueError("Mesh must lie inside normalized SDF box")

    n = resolution
    sampling_offset = 0.5
    cap = 4 * np.sqrt(3) / n
    radius = np.sqrt(3) / n
    pullback = 1e-4 * cap
    length = np.sqrt(3) + pullback
    samples = directions()
    counts = np.empty((n, n, n), np.uint8)
    udf = np.empty((n, n, n), np.float32)
    backside = np.empty((n, n, n), bool)
    timings = dict(distance_seconds=0.0, raycast_seconds=0.0)
    queries = gate_queries = None
    started = time.perf_counter()
    try:
        queries = Queries(mesh, raycast_backend, distance_backend)
        timings["scene_seconds"] = time.perf_counter() - started
        adapter = queries.adapter
        yz = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
        for x in range(n):
            points = (np.column_stack([np.full(len(yz), x), yz]).astype(np.float32)
                      + sampling_offset) / n - 0.5
            start = time.perf_counter()
            distance, back = queries.nearest(points, cap)
            timings["distance_seconds"] += time.perf_counter() - start
            udf[x] = np.minimum(distance, cap).reshape(n, n)
            backside[x] = back.reshape(n, n)
            start = time.perf_counter()
            counts[x] = queries.counts(points, length, samples, pullback).reshape(n, n)
            timings["raycast_seconds"] += time.perf_counter() - start
            if x % 32 == 0:
                print(f"G0 UE50 queries: {x}/{n}", flush=True)
        queries.close()
        queries = None

        a1 = np.where(counts > 49, -udf, udf).astype(np.float32)
        result = {"A1": a1, "counts": counts, "udf": udf, "backside": backside}
        common = dict(raycast_backend=raycast_backend, distance_backend=distance_backend,
                      adapter=adapter, directions=98, seed=0, backface_fraction=0.5,
                      trace_length_voxels=float(length * n), ray_extent="domain",
                      pullback_voxels=float(pullback * n),
                      distance_cap_voxels=float(cap * n),
                      queried_samples=n**3, sampling_offset=sampling_offset)
        result["audit_A1"] = dict(method="ue50_votes_before_shell", **common,
                                   negative_samples=int((a1 < 0).sum()), **timings)
        if not shell:
            return result

        from skimage.measure import marching_cubes
        import trimesh

        start = time.perf_counter()
        vote_field = 0.25 - counts.astype(np.float32) / 98
        if vote_field.min() < 0 < vote_field.max():
            vertices, faces, _, _ = marching_cubes(
                vote_field, level=0, spacing=(1 / n,) * 3, allow_degenerate=False
            )
            gate = trimesh.Trimesh(vertices + sampling_offset / n - 0.5,
                                   faces, process=False)
            gate_queries = Queries(gate, raycast_backend, distance_backend)
        a2 = a1.copy()
        ids = np.argwhere(backside & (udf < radius + 2 / n))
        added = 0
        for begin in range(0, len(ids), 65536):
            ijk = ids[begin:begin + 65536]
            idx = tuple(ijk.T)
            points = (ijk.astype(np.float32) + sampling_offset) / n - 0.5
            gd = (gate_queries.nearest(points, cap)[0] if gate_queries is not None
                  else np.full(len(points), cap, np.float32))
            gd = np.minimum(gd, cap)
            gd = np.where(counts[idx] > 24.5, -gd, gd)
            shell_values = np.maximum(np.maximum(-udf[idx], udf[idx] - radius), gd)
            old = a2[idx]
            new = np.minimum(old, shell_values)
            added += int(((new < 0) & (old >= 0)).sum())
            a2[idx] = new
        timings["gate_seconds"] = time.perf_counter() - start
        result["A2"] = a2
        result["audit_A2"] = dict(method="ue50_back_shell25", **common,
                                   shell_lower_fraction=0.25,
                                   shell_radius_voxels=float(np.sqrt(3)),
                                   negative_samples=int((a2 < 0).sum()),
                                   added_shell_samples=added,
                                   front_sample_changes=0, **timings)
        return result
    finally:
        if queries is not None:
            queries.close()
        if gate_queries is not None:
            gate_queries.close()
