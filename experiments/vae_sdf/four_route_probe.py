"""GPU preflight for differentiable MC/FlexiCubes field extraction.

This is a gate for the four-route experiment, not an A3/A4 fitting runner.
It reads an existing A4 target and writes no target cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(summary_path: Path, index: int, device: str = "cuda:0",
        finite_difference: bool = False) -> dict:
    import numpy as np
    import torch
    from vae3d.render_refine import DynamicSurface
    from sparc3d.ops.flexicubes import FlexiCubes
    from .g0b_field_adapter import HI3D_FROM_VAE
    from .four_route_normals import corner_normals

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the four-route gradient gate")
    row = json.loads(summary_path.read_text())["assets"][index]
    if row["status"] != "success":
        raise ValueError("Selected A4 asset has no successful cache")
    cache = Path(row["cache"])
    with np.load(cache / "target.npz", allow_pickle=False) as data:
        points = np.asarray(data["coords"], np.int32)
        features = np.asarray(data["features"], np.float32)
        cells = np.asarray(data["cells"], np.int32)
        indices = np.asarray(data["cell_indices"], np.int32)
        resolution = int(data["resolution"])
        phase = float(data["sampling_offset"])
    packed = {
        "coords": cells,
        "features": np.concatenate((features[indices, 0], features[indices, 1:].reshape(-1, 24)), axis=1),
        "resolution": np.int32(resolution),
        "sampling_offset": np.float32(phase),
    }
    surface = DynamicSurface(packed, device=device)
    if not torch.equal(surface.corners[:, 1:], torch.as_tensor(points, device=device)):
        raise ValueError("Unique point order differs from cached A4 field")
    if not torch.equal(surface.indices, torch.as_tensor(indices, device=device)):
        raise ValueError("Cell index order differs from cached A4 field")
    flexi = FlexiCubes(device=device)
    cell_coord = torch.as_tensor(cells, device=device, dtype=torch.int32)
    cube_order = torch.argsort((cell_coord[:, 0] * resolution + cell_coord[:, 1]) * resolution + cell_coord[:, 2])
    cubes = surface.indices[:, HI3D_FROM_VAE][cube_order]
    cell_coord = cell_coord[cube_order]
    result = {"asset_index": index, "asset_id": row["id"], "resolution": resolution,
              "sampling_offset": phase, "input_cache": str(cache), "routes": {}}
    for extractor in ("mc", "flexicubes"):
        for mode in ("flat", "smooth"):
            signed = torch.nn.Parameter(surface.initial.clone())
            delta = torch.nn.Parameter(surface.initial_delta.clone())
            def extract(s, d):
                if extractor == "mc":
                    return surface.extract(s, d)
                xyz = (surface.corners[:, 1:].float() + phase + d) / resolution - 0.5
                return flexi(xyz, s, cubes, resolution, cell_coord)

            def objective(s, d):
                v, f = extract(s, d)
                w = torch.tensor([0.7, 1.1, 1.3], device=device)
                positions = ((v * w).square().sum(-1)).mean()
                normals = corner_normals(v, f, mode)
                normal_term = ((normals * w).sum(-1).square()).mean()
                return positions + 0.1 * normal_term, positions, normal_term, v, f

            loss, position_loss, normal_loss, vertices, faces = objective(signed, delta)
            if vertices.numel() == 0 or faces.numel() == 0 or not torch.isfinite(vertices).all():
                raise RuntimeError(f"{extractor} returned an empty or nonfinite mesh")
            gs, gd = torch.autograd.grad(loss, (signed, delta), allow_unused=True)
            if gs is None or gd is None or not torch.isfinite(gs).all() or not torch.isfinite(gd).all():
                raise RuntimeError(f"{extractor}/{mode} has disconnected or nonfinite s/delta gradients")
            nz_s = int(torch.count_nonzero(gs).item())
            nz_d = int(torch.count_nonzero(gd).item())
            if not nz_s or not nz_d:
                raise RuntimeError(f"{extractor}/{mode} has zero s or delta gradient")
            result["routes"][f"{extractor}_{mode}"] = {
                "vertices": len(vertices), "faces": len(faces),
                "loss": float(loss.detach()), "position_loss": float(position_loss.detach()),
                "normal_loss": float(normal_loss.detach()),
                "nonzero_s_gradients": nz_s, "nonzero_delta_gradients": nz_d,
                "max_abs_s_gradient": float(gs.abs().max()),
                "max_abs_delta_gradient": float(gd.abs().max()),
            }
            if finite_difference:
                # The position term is smooth under fixed topology. The normal
                # term contains normalization and can be poorly conditioned at
                # nearly degenerate triangles, so probe extraction with position.
                probe_position = objective(signed, delta)[1]
                pgs, pgd = torch.autograd.grad(probe_position, (signed, delta))
                directions = {
                    "s": pgs * (signed.detach().abs() > .05),
                    "delta": pgd,
                }
                checks = {}
                for name, gradient in (("s", pgs), ("delta", pgd)):
                    direction = directions[name]
                    # A dense unit direction changes each float32 field value by
                    # far less than one ulp. Concentrate the probe on the largest
                    # gradient entries so the central difference is measurable.
                    flat = direction.flatten()
                    count = min(64, flat.numel())
                    keep = torch.topk(flat.abs(), count).indices
                    sparse = torch.zeros_like(flat)
                    sparse[keep] = flat[keep]
                    direction = sparse.reshape_as(direction)
                    norm = torch.linalg.vector_norm(direction)
                    if float(norm) < 1e-12:
                        raise RuntimeError(f"{extractor}/{mode} has no stable {name} direction")
                    direction = direction / norm
                    epsilon = .1
                    with torch.no_grad():
                        if name == "s":
                            plus = objective(signed.detach() + epsilon * direction, delta.detach())
                            minus = objective(signed.detach() - epsilon * direction, delta.detach())
                        else:
                            plus = objective(signed.detach(), delta.detach() + epsilon * direction)
                            minus = objective(signed.detach(), delta.detach() - epsilon * direction)
                    if not torch.equal(plus[4], faces) or not torch.equal(minus[4], faces):
                        raise RuntimeError(f"{extractor}/{mode} changed topology during {name} finite difference")
                    numeric = float((plus[1] - minus[1]) / (2 * epsilon))
                    analytic = float((gradient * direction).sum())
                    relative = abs(numeric - analytic) / max(abs(analytic), abs(numeric), 1e-7)
                    if numeric * analytic <= 0 or relative > .1:
                        raise RuntimeError(f"{extractor}/{mode} {name} finite difference disagrees: "
                                           f"analytic={analytic}, numeric={numeric}, relative={relative}")
                    checks[name] = {"term": "position_loss", "analytic": analytic, "numeric": numeric,
                                    "relative_error": relative, "epsilon": epsilon}
                result["routes"][f"{extractor}_{mode}"]["finite_difference"] = checks
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a4-summary", type=Path, required=True)
    parser.add_argument("--index", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--finite-difference", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.a4_summary, args.index, args.device, args.finite_difference),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
