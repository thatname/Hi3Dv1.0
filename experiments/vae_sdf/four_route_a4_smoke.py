"""Run one render optimization step per route using a fixed historical A3 cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(manifest_path: Path, a3_summary: Path, index: int, image_resolution: int, route: str | None):
    import numpy as np
    import torch
    from vae3d.data import load_mesh
    from vae3d.refine import grid_quality
    from vae3d.render_refine import fit as baseline_fit
    from .four_route_a4 import fit_a4

    assets = json.loads(manifest_path.read_text())["assets"]
    rows = json.loads(a3_summary.read_text())["assets"]
    asset, row = assets[index], rows[index]
    if row["id"] != asset["id"] or row["status"] != "success":
        raise ValueError("A3 summary differs from fixed manifest")
    cache = Path(row["cache"])
    with np.load(cache / "target.npz", allow_pickle=False) as data:
        points = np.asarray(data["coords"], np.int32)
        features = np.asarray(data["features"], np.float32)
        cells = np.asarray(data["cells"], np.int32)
        indices = np.asarray(data["cell_indices"], np.int32)
        resolution = int(data["resolution"])
        phase = float(data["sampling_offset"])
    mesh, _, _ = load_mesh(asset["path"], normalization_extent=(resolution - 2) / resolution)
    packed = {"coords": cells, "features": np.concatenate((
        features[indices, 0], features[indices, 1:].reshape(-1, 24)), axis=1),
        "vertices": np.asarray(mesh.vertices, np.float32), "faces": np.asarray(mesh.faces),
        "resolution": np.int32(resolution), "sampling_offset": np.float32(phase)}
    output = {"index": index, "id": asset["id"], "a3_cache": str(cache),
              "physical_gpu": 7, "logical_device": "cuda:0", "routes": {}}
    for extractor in ("mc", "flexicubes"):
        for mode in ("flat", "smooth"):
            if route is not None and f"{extractor}_{mode}" != route:
                continue
            torch.cuda.reset_peak_memory_stats()
            fitted, trace, audit = fit_a4(packed, extractor, mode, steps=1,
                                          image_resolution=image_resolution, device="cuda:0")
            output["routes"][f"{extractor}_{mode}"] = {
                "trace": trace, "audit": audit, "grid_quality": grid_quality(fitted),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())}
            if extractor == "mc" and mode == "flat":
                baseline, baseline_trace, baseline_audit = baseline_fit(
                    packed, steps=1, image_resolution=image_resolution,
                    views=32, seed=42, allow_sign_changes=True,
                    supersample=1, direct_fixed_signs=False,
                    max_deformation=.5, target_thickness_pixels=1.,
                    depth_weight=10., render_backend="cuda", freeze_boundary=False)
                error = float(np.max(np.abs(fitted["features"] - baseline["features"])))
                output["mc_flat_baseline_check"] = {
                    "max_abs_field_delta_error": error,
                    "loss_error": abs(trace[-1]["loss"] - baseline_trace[-1]["loss"]),
                    "before_error": max(abs(audit["before"][k] - baseline_audit["before"][k])
                                        for k in audit["before"]),
                }
                if error > 1e-5:
                    raise RuntimeError(f"New MC flat A4 differs from pinned vae3d: {error}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a3-summary", type=Path, required=True)
    parser.add_argument("--index", type=int, default=48)
    parser.add_argument("--image-resolution", type=int, default=128)
    parser.add_argument("--route", choices=("mc_flat", "mc_smooth", "flexicubes_flat", "flexicubes_smooth"))
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.a3_summary, args.index, args.image_resolution, args.route), indent=2))


if __name__ == "__main__":
    main()
