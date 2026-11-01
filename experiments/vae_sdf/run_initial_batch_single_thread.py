"""Run the A1/A2 batch with small thread pools.

`vae3d.back_shell.cpu_scene` builds the open3d raycasting scene with a hardcoded
`nthreads=8`, and no environment variable reaches it. That caps the batch at
about sixteen processes: beyond that the ray threads of different assets fight
for the same cores and each asset slows down faster than the extra concurrency
helps. The scene is rebuilt here with a configurable thread count instead; the
raycast result does not depend on how many threads run it.

The batch plan pins the sha256 of `run_initial_batch.py`, so this behaviour lives
in a wrapper rather than in that file: editing it would invalidate an output root
that is already in progress.
"""

from __future__ import annotations

import os

import open3d as o3d
import torch

import vae3d.back_shell as back_shell

from .run_initial_batch import main

torch.set_num_threads(int(os.environ.get("VAE3D_TORCH_THREADS", "1")))

_RAY_THREADS = int(os.environ.get("VAE3D_RAY_THREADS", "1"))


def _cpu_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene(nthreads=_RAY_THREADS)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                  o3d.utility.Vector3iVector(mesh.faces))))
    return scene


back_shell.cpu_scene = _cpu_scene

if __name__ == "__main__":
    raise SystemExit(main())