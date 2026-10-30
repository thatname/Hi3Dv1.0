"""Compare CUDA and CPU corner fans, normals, gradients, and runtime on GPU 7."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def canonical(labels):
    import numpy as np

    flat = labels.reshape(-1).cpu().numpy()
    first = {}
    return np.fromiter((first.setdefault(int(x), i) for i, x in enumerate(flat)),
                       dtype=np.int64, count=len(flat))


def run_case(name, vertices, faces):
    import numpy as np
    import torch
    from .four_route_normals import corner_fans_cpu, corner_fans_gpu, corner_normals

    fcpu = torch.as_tensor(faces, dtype=torch.long)
    fgpu = fcpu.to("cuda:0")
    v = torch.as_tensor(vertices, dtype=torch.float32, device="cuda:0")
    started = time.perf_counter()
    reference = corner_fans_cpu(fcpu, len(v))
    cpu_seconds = time.perf_counter() - started
    torch.cuda.synchronize()
    started = time.perf_counter()
    gpu = corner_fans_gpu(fgpu, len(v))
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - started
    if not np.array_equal(canonical(reference), canonical(gpu)):
        raise AssertionError(f"{name}: CUDA fan partition differs from CPU")
    va = v.detach().clone().requires_grad_(True)
    vb = v.detach().clone().requires_grad_(True)
    na = corner_normals(va, fgpu, "smooth", fans=reference.to("cuda:0"))
    nb = corner_normals(vb, fgpu, "smooth", fans=gpu)
    normal_error = float((na - nb).abs().max())
    weight = torch.arange(na.numel(), device="cuda:0").reshape_as(na) % 7
    (na * weight).sum().backward()
    (nb * weight).sum().backward()
    grad_error = float((va.grad - vb.grad).abs().max())
    grad_scale = max(float(va.grad.abs().max()), 1.)
    grad_relative_error = grad_error / grad_scale
    if (not torch.isfinite(vb.grad).all() or normal_error > 2e-5
            or grad_relative_error > 1e-5):
        raise AssertionError(f"{name}: CUDA normals or gradient differ: {normal_error}, "
                             f"{grad_error}, relative {grad_relative_error}")
    return {"case": name, "faces": len(faces), "corners": fcpu.numel(),
            "cpu_seconds": cpu_seconds, "gpu_seconds": gpu_seconds,
            "speedup": cpu_seconds / gpu_seconds, "normal_max_error": normal_error,
            "gradient_max_error": grad_error, "gradient_relative_error": grad_relative_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise ValueError("This check is pinned to physical GPU 7")
    import numpy as np
    import trimesh

    cases = [
        ("tetra", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
         np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])),
        ("nonmanifold_edge", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                        [0, -1, 0], [0, 0, 1]]),
         np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])),
        ("same_winding", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]]),
         np.array([[0, 1, 2], [0, 1, 3]])),
        ("degenerate", np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]]),
         np.array([[0, 1, 2], [0, 1, 3]])),
    ]
    mesh = trimesh.load(args.mesh, process=False)
    cases.append(("fixed_mesh", np.asarray(mesh.vertices), np.asarray(mesh.faces)))
    output = [run_case(name, vertices, faces) for name, vertices, faces in cases]
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
