"""Independent A3 geometry fit with a selected extractor and normal model."""

from __future__ import annotations

import numpy as np

from .g0b_field_adapter import HI3D_FROM_VAE
from .four_route_normals import corner_normals


class RouteSurface:
    def __init__(self, sample, extractor: str, device="cuda:0"):
        import torch
        from vae3d.render_refine import DynamicSurface
        from sparc3d.ops.flexicubes import FlexiCubes

        if extractor not in {"mc", "flexicubes"}:
            raise ValueError("Unknown extractor")
        self.base = DynamicSurface(sample, device=device)
        self.extractor = extractor
        self.flexi = FlexiCubes(device=device) if extractor == "flexicubes" else None
        self.resolution = int(sample["resolution"])
        self.phase = float(sample.get("sampling_offset", 0.0))
        cells = torch.as_tensor(sample["coords"], device=device, dtype=torch.int32)
        order = torch.argsort((cells[:, 0] * self.resolution + cells[:, 1]) * self.resolution + cells[:, 2])
        self.cubes = self.base.indices[:, HI3D_FROM_VAE][order]
        self.cell_coord = cells[order]

    def extract(self, signed, delta):
        if self.extractor == "mc":
            return self.base.extract(signed, delta)
        xyz = (self.base.corners[:, 1:].float() + self.phase + delta) / self.resolution - .5
        return self.flexi(xyz, signed, self.cubes, self.resolution, self.cell_coord)


def smooth_surface_loss(pv, pf, tv, tf, samples, target_weights, tail_weight, target_normals):
    import torch
    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    from vae3d.geometry import sample_surface
    from vae3d.normal_targets import interpolate_normals

    p, _, pface = sample_surface(pv, pf, samples, return_face=True)
    t, _, tface = sample_surface(tv, tf, samples, target_weights, return_face=True)
    pd, pi, _ = point_to_mesh_distance(p[None].contiguous(), tv[tf][None].contiguous())
    td, ti, _ = point_to_mesh_distance(t[None].contiguous(), pv[pf][None].contiguous())
    pred_normals = corner_normals(pv, pf, "smooth")
    pn = interpolate_normals(p, pv[pf[pface]], pred_normals[pface])
    tn = interpolate_normals(t, tv[tf[tface]], target_normals[tface])
    target_nearest = interpolate_normals(p, tv[tf[pi[0]]], target_normals[pi[0]])
    pred_nearest = interpolate_normals(t, pv[pf[ti[0]]], pred_normals[ti[0]])
    forward = 1 - (pn * target_nearest).sum(-1).abs().clamp(max=1)
    backward = 1 - (tn * pred_nearest).sum(-1).abs().clamp(max=1)
    normal = (forward.mean() + backward.mean()) / 2
    distance = (pd.mean() + td.mean()) / 2
    if tail_weight:
        k = max(1, samples // 10)
        distance = distance + tail_weight * (pd.flatten().topk(k).values.mean() + td.flatten().topk(k).values.mean()) / 2
    return distance, normal


def fit_a3(sample, extractor: str, normal_mode: str, *, steps=30, samples=8192,
           device="cuda:0", max_deformation=.5, seed=42):
    import torch
    import torch.nn.functional as F
    import trimesh
    from vae3d.geometry import CORNERS, EDGES, shared_corners, surface_loss
    from vae3d.refine import (cell_determinants, local_valid_step, projection_vectors,
                              scaled_sdf, surface_plan, valid_step)
    from vae3d.quality import feature_faces

    if extractor not in {"mc", "flexicubes"} or normal_mode not in {"flat", "smooth"}:
        raise ValueError("Invalid route")
    if steps < 1 or samples < 1:
        raise ValueError("Invalid fit budget")
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if extractor == "mc" and normal_mode == "flat":
        from vae3d.refine import refine
        # Preserve the pinned historical route exactly. The other three routes
        # use the independent differentiable extractor/normal implementation.
        result, trace = refine(sample, device=device, steps=steps, samples=samples,
                               max_deformation=max_deformation, mode="joint")
        result["route_extractor"] = np.array(extractor)
        result["route_normal_mode"] = np.array(normal_mode)
        return result, trace
    resolution = int(sample["resolution"])
    cells = torch.as_tensor(sample["coords"], device=device, dtype=torch.int32)
    batch_cells = torch.cat([torch.zeros(len(cells), 1, device=device, dtype=torch.int32), cells], 1)
    initial_signed = torch.as_tensor(sample["features"][:, :8], device=device)
    unique, indices, a, b, t, faces = surface_plan(batch_cells, initial_signed)
    if not len(faces):
        raise ValueError("No initial A2 surface")
    route_surface = RouteSurface(sample, extractor, device)
    if not torch.equal(unique, route_surface.base.corners) or not torch.equal(indices, route_surface.base.indices):
        raise ValueError("A3 surface corner packing differs")
    base = (unique[:, 1:].float() + float(sample.get("sampling_offset", 0.0))) / resolution - .5
    source_v = torch.as_tensor(sample["vertices"], device=device, dtype=torch.float32)
    source_f = torch.as_tensor(sample["faces"], device=device, dtype=torch.long)
    target_weights = torch.as_tensor(feature_faces(trimesh.Trimesh(
        vertices=sample["vertices"], faces=sample["faces"], process=False)).astype(np.float32), device=device)
    target_weights = 1 + 3 * target_weights
    target_normals = corner_normals(source_v, source_f, "smooth") if normal_mode == "smooth" else None
    initial_vertices = base[a] * (1 - t[:, None]) + base[b] * t[:, None]
    projection = projection_vectors(initial_vertices, source_v[source_f][None].contiguous(), 4096)
    desired = projection * resolution
    offset = torch.zeros_like(base)
    count = torch.zeros(len(base), 1, device=device)
    for endpoint in (a, b):
        offset.index_add_(0, endpoint, desired)
        count.index_add_(0, endpoint, torch.ones(len(endpoint), 1, device=device))
    offset /= count.clamp_min(1)
    offset.clamp_(-max_deformation, max_deformation)
    offset, initial_scale = valid_step(torch.zeros_like(offset), offset, indices, .05)
    displacement = torch.nn.Parameter(offset)
    _, mean, _ = shared_corners(batch_cells, initial_signed[..., None])
    fixed_sdf = mean[:, 0]
    log_scale = torch.nn.Parameter(torch.zeros_like(fixed_sdf))
    grid_edges = indices[:, torch.as_tensor(EDGES, device=device)].reshape(-1, 2).sort(1).values
    grid_edges = torch.unique(grid_edges, dim=0)
    optimizer = torch.optim.Adam([displacement, log_scale], lr=.03)
    trace = [{"initial_projection_scale": float(initial_scale)}]
    for step in range(steps):
        optimizer.param_groups[0]["lr"] = .03 * (.1 ** (step / max(steps - 1, 1)))
        optimizer.zero_grad(set_to_none=True)
        signed = scaled_sdf(fixed_sdf, log_scale)
        vertices, fitted_faces = route_surface.extract(signed, displacement)
        if not len(fitted_faces):
            raise RuntimeError("A3 surface became empty")
        if normal_mode == "flat":
            distance, normal = surface_loss(vertices, fitted_faces, source_v, source_f,
                                            samples=samples, target_weights=target_weights,
                                            tail_weight=.25)
        else:
            distance, normal = smooth_surface_loss(vertices, fitted_faces, source_v, source_f,
                                                   samples, target_weights, .25, target_normals)
        edge_smooth = (displacement[a] - displacement[b]).square().mean()
        ea, eb = grid_edges.unbind(1)
        coherence = F.smooth_l1_loss(displacement[ea], displacement[eb], beta=.25)
        loss = distance * resolution ** 2 + 5 * normal + .01 * edge_smooth + .05 * coherence + .001 * log_scale.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite A3 loss")
        loss.backward()
        for parameter in (displacement, log_scale):
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Disconnected or nonfinite A3 gradient")
        torch.nn.utils.clip_grad_norm_([displacement, log_scale], 10.)
        previous = displacement.detach().clone()
        optimizer.step()
        with torch.no_grad():
            displacement.clamp_(-max_deformation, max_deformation)
            log_scale.clamp_(-2., 2.)
            accepted, alpha = local_valid_step(previous, displacement, indices, .05)
            displacement.copy_(accepted)
            if alpha < 1:
                optimizer.state[displacement]["exp_avg"].mul_(alpha)
        if step % 10 == 0 or step == steps - 1:
            trace.append({"step": step + 1, "loss": float(loss.detach()),
                          "distance": float(distance.detach()), "normal": float(normal.detach()),
                          "faces": len(fitted_faces), "accepted_step_scale": float(alpha)})
    result = dict(sample)
    result["features"] = torch.cat([scaled_sdf(fixed_sdf, log_scale)[indices],
                                    displacement[indices].reshape(-1, 24)], 1).detach().cpu().numpy()
    result["route_extractor"] = np.array(extractor)
    result["route_normal_mode"] = np.array(normal_mode)
    result["cell_volume_guard"] = np.bool_(True)
    return result, trace
