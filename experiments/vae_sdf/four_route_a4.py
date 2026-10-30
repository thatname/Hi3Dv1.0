"""A4 render fit using the selected extractor and flat/smooth normal loss."""

from __future__ import annotations

import numpy as np

from .four_route_a3 import RouteSurface
from .four_route_normals import corner_normals


class RouteRenderer:
    def __init__(self, resolution: int, normal_mode: str, backend="cuda"):
        import nvdiffrast.torch as dr

        if normal_mode not in {"flat", "smooth"}:
            raise ValueError("Invalid normal mode")
        if backend != "cuda":
            raise ValueError("Four-route renderer currently requires CUDA rasterization")
        self.dr = dr
        self.context = dr.RasterizeCudaContext()
        self.resolution = resolution
        self.normal_mode = normal_mode

    def __call__(self, vertices, faces, camera, *, corner_override=None):
        import torch
        import torch.nn.functional as F

        dr = self.dr
        view = vertices @ camera.T
        clip = torch.cat((view[:, :2] / .65, -view[:, 2:3], torch.ones_like(view[:, :1])), 1)[None].contiguous()
        triangles = faces.int().contiguous()
        raster, _ = dr.rasterize(self.context, clip, triangles, (self.resolution, self.resolution))
        mask = dr.antialias((raster[..., 3:4] > 0).float(), raster, clip, triangles)
        depth = dr.interpolate(view[:, 2:3][None].contiguous(), raster, triangles)[0]
        normals = corner_override if corner_override is not None else corner_normals(vertices, faces, self.normal_mode)
        if normals.shape != (len(faces), 3, 3):
            raise ValueError("Corner normal buffer differs from faces")
        attributes = normals.reshape(1, -1, 3).contiguous()
        normal_faces = torch.arange(len(faces) * 3, device=vertices.device, dtype=torch.int32).reshape(-1, 3)
        normal = dr.interpolate(attributes, raster, normal_faces)[0]
        if self.normal_mode == "smooth":
            normal = F.normalize(normal, dim=-1, eps=1e-12)
        return mask, depth, normal


def fit_a4(sample, extractor: str, normal_mode: str, *, steps=200,
           image_resolution=4096, views=32, seed=42, max_deformation=.5,
           depth_weight=10., target_thickness_pixels=1., device="cuda:0"):
    import torch
    import torch.nn.functional as F
    from vae3d.refine import cell_determinants
    from vae3d.render_refine import cameras, image_loss, inward_render_prisms

    if steps < 1 or image_resolution < 32 or views < 2 or device != "cuda:0":
        raise ValueError("Invalid A4 configuration")
    torch.manual_seed(seed)
    surface = RouteSurface(sample, extractor, device)
    base = surface.base
    signed = torch.nn.Parameter(base.initial.clone())
    raw_delta = torch.nn.Parameter(base.initial_delta.clone())
    def delta():
        return raw_delta.clamp(-max_deformation, max_deformation)

    renderer = RouteRenderer(image_resolution, normal_mode)
    original_v = torch.as_tensor(sample["vertices"], device=device, dtype=torch.float32)
    original_f = torch.as_tensor(sample["faces"], device=device, dtype=torch.long)
    thickness = 1.3 * target_thickness_pixels / image_resolution
    if thickness > 0:
        target_v, target_f = inward_render_prisms(original_v, original_f, thickness)
    else:
        target_v, target_f = original_v, original_f
    target_corner = corner_normals(target_v, target_f, "flat")
    if normal_mode == "smooth":
        source_corner = corner_normals(original_v, original_f, "smooth")
        if thickness > 0:
            target_corner = target_corner.clone()
            target_corner[0::8] = source_corner
        else:
            target_corner = source_corner
    train = cameras(views, device=device)
    held = cameras(12, phase=.731, device=device)
    with torch.no_grad():
        tri = original_v[original_f]
        area_vector = .5 * torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
        all_cameras = torch.cat((train, held))
        projected_area = (area_vector @ all_cameras[:, 2, :].T).abs().sum(0)
        weights = (projected_area / (.05 * projected_area.max().clamp_min(1e-12))).clamp(.02, 1.)
        train_weights, held_weights = weights[:views], weights[views:]
        targets = [tuple(x.cpu() for x in renderer(target_v, target_f, c, corner_override=target_corner))
                   for c in train] if image_resolution <= 1024 else None
        held_targets = [tuple(x.cpu() for x in renderer(target_v, target_f, c, corner_override=target_corner))
                        for c in held] if image_resolution <= 1024 else None
    def target(cache, camera, index):
        if cache is not None:
            return tuple(x.to(device) for x in cache[index])
        with torch.no_grad():
            return renderer(target_v, target_f, camera, corner_override=target_corner)
    def evaluate(vertices, faces):
        with torch.no_grad():
            corners = corner_normals(vertices, faces, normal_mode)
            scores = [held_weights[j] * torch.stack(image_loss(
                renderer(vertices, faces, camera, corner_override=corners),
                target(held_targets, camera, j)))
                for j, camera in enumerate(held)]
            return dict(zip(("silhouette", "depth", "normal"),
                            torch.stack(scores).mean(0).cpu().tolist()))

    before = evaluate(*surface.extract(signed, delta()))
    optimizer = torch.optim.Adam([{"params": [signed], "lr": .08},
                                  {"params": [raw_delta], "lr": .03}])
    trace = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        view_indices = torch.randperm(views, device=device)[:2].tolist()
        terms = []
        if normal_mode == "smooth":
            pv, pf = surface.extract(signed, delta())
            if not len(pf):
                raise RuntimeError("A4 surface became empty")
            prediction_corners = corner_normals(pv, pf, normal_mode)
            objectives = []
        for j in view_indices:
            if normal_mode == "flat":
                pv, pf = surface.extract(signed, delta())
                if not len(pf):
                    raise RuntimeError("A4 surface became empty")
            losses = train_weights[j] * torch.stack(image_loss(
                renderer(pv, pf, train[j], corner_override=prediction_corners)
                if normal_mode == "smooth" else renderer(pv, pf, train[j]),
                target(targets, train[j], j)))
            objective = losses @ losses.new_tensor([1., depth_weight, .1]) / len(view_indices)
            if not torch.isfinite(objective):
                raise RuntimeError("Nonfinite A4 image loss")
            if normal_mode == "smooth":
                objectives.append(objective)
            else:
                objective.backward()
            terms.append(losses.detach())
        if normal_mode == "smooth":
            torch.stack(objectives).sum().backward()
        validity = F.relu(.05 - cell_determinants(delta(), base.indices)).mean()
        regularizer = 10 * validity + .001 * delta().square().mean()
        regularizer.backward()
        for parameter in (signed, raw_delta):
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Disconnected or nonfinite A4 s/delta gradient")
        mean_terms = torch.stack(terms).mean(0)
        total = mean_terms @ mean_terms.new_tensor([1., depth_weight, .1]) + regularizer.detach()
        torch.nn.utils.clip_grad_norm_([signed, raw_delta], 10.)
        optimizer.step()
        with torch.no_grad():
            signed.clamp_(-4, 4)
            raw_delta.clamp_(-max_deformation, max_deformation)
        if step % 10 == 0 or step == steps - 1:
            trace.append({"step": step + 1, "loss": float(total),
                          "silhouette": float(mean_terms[0]), "depth": float(mean_terms[1]),
                          "normal": float(mean_terms[2]), "faces": len(pf),
                          "changed_signs": int(((signed < 0) != (base.initial < 0)).sum())})
    with torch.no_grad():
        final_v, final_f = surface.extract(signed, delta())
        after = evaluate(final_v, final_f)
        result = dict(sample)
        result["features"] = torch.cat((signed[base.indices], delta()[base.indices].reshape(-1, 24)), 1).cpu().numpy()
        result["route_extractor"] = np.array(extractor)
        result["route_normal_mode"] = np.array(normal_mode)
        result["render_signs_fixed"] = np.bool_(False)
        result["cell_volume_guard"] = np.bool_(False)
        audit = {"before": before, "after": after, "train_views": views,
                 "held_out_views": 12, "image_resolution": image_resolution,
                 "target_thickness_pixels": target_thickness_pixels,
                 "depth_weight": depth_weight,
                 "changed_signs": int(((signed < 0) != (base.initial < 0)).sum()),
                 "max_abs_delta": float(delta().abs().max()),
                 "extractor": extractor, "normal_mode": normal_mode,
                 "physical_gpu": 7, "logical_device": device}
    return result, trace, audit
