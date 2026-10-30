"""Run one A3 optimization step for all four routes on one cached A2 asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(manifest_path: Path, summary_path: Path, index: int, samples: int):
    import numpy as np
    import torch
    from vae3d.data import load_mesh
    from vae3d.refine import grid_quality
    from .four_route_a3 import fit_a3

    manifest = json.loads(manifest_path.read_text())["assets"]
    summary = json.loads(summary_path.read_text())["assets"]
    asset, row = manifest[index], summary[index]
    if row["id"] != asset["id"] or row["status"] != "success":
        raise ValueError("A2 summary differs from fixed manifest")
    cache = Path(row["caches"]["A2"])
    with np.load(cache / "target.npz", allow_pickle=False) as data:
        points = np.asarray(data["coords"], np.int32)
        features = np.asarray(data["features"], np.float32)
        cells = np.asarray(data["cells"], np.int32)
        indices = np.asarray(data["cell_indices"], np.int32)
        resolution = int(data["resolution"])
        phase = float(data["sampling_offset"])
    if phase != .5 or np.any(features[:, 1:] != 0):
        raise ValueError("Expected historical phase=.5 A2 zero-deformation cache")
    mesh, center, scale = load_mesh(asset["path"], normalization_extent=(resolution - 2) / resolution)
    packed = {"coords": cells, "features": np.concatenate(
        (features[indices, 0], features[indices, 1:].reshape(-1, 24)), axis=1),
        "vertices": np.asarray(mesh.vertices, np.float32), "faces": np.asarray(mesh.faces),
        "resolution": np.int32(resolution), "sampling_offset": np.float32(phase)}
    output = {"index": index, "id": asset["id"], "a2_cache": str(cache),
              "physical_gpu": 7, "logical_device": "cuda:0", "routes": {}}
    for extractor in ("mc", "flexicubes"):
        for normal_mode in ("flat", "smooth"):
            torch.cuda.reset_peak_memory_stats()
            fitted, trace = fit_a3(packed, extractor, normal_mode, steps=1,
                                   samples=samples, device="cuda:0", seed=42)
            quality = grid_quality(fitted)
            output["routes"][f"{extractor}_{normal_mode}"] = {
                "trace": trace, "grid_quality": quality,
                "max_abs_delta": float(np.abs(fitted["features"][:, 8:]).max()),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            }
            if extractor == "mc" and normal_mode == "flat":
                output["mc_flat_baseline_check"] = {
                    "implementation": "pinned vae3d.refine.refine(mode=joint)",
                    "same_function_as_historical_A3": True,
                }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--index", type=int, default=48)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.a1a2_summary, args.index, args.samples), indent=2))


if __name__ == "__main__":
    main()
