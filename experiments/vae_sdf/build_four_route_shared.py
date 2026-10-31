"""Extract shared A1/A2 classic MC and FlexiCubes meshes on physical GPU 7."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--metric-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in {"1", "2", "3", "7"}:
        raise ValueError("Shared extraction needs one whitelisted physical GPU (1/2/3/7)")
    import numpy as np
    import torch
    import trimesh
    from vae3d.data import load_mesh
    from .four_route_a3 import RouteSurface
    from .evaluate_initial_targets import mesh_metrics, sample_surface, scene_for

    assets = json.loads(args.manifest.read_text())["assets"]
    rows = json.loads(args.a1a2_summary.read_text())["assets"]
    if not assets or len(assets) != len(rows):
        raise ValueError("Manifest and A1/A2 summary must list the same assets")
    failure_count = 0
    for index in args.indices:
        if not 0 <= index < len(assets):
            raise ValueError(index)
        asset, row = assets[index], rows[index]
        if (row["index"] != index or row["id"] != asset["id"]
                or row["source_sha256"] != asset["sha256"] or row["status"] != "success"):
            raise ValueError(f"A1/A2 mismatch at {index}")
        normalized, center, scale = load_mesh(asset["path"], normalization_extent=126 / 128)
        reference = trimesh.Trimesh(vertices=np.asarray(normalized.vertices) * scale + center,
                                    faces=np.asarray(normalized.faces), process=False)
        reference_points, reference_normals = sample_surface(reference, args.metric_samples, args.seed)
        reference_scene = scene_for(reference)
        for stage in ("A1", "A2"):
            with np.load(Path(row["caches"][stage]) / "target.npz", allow_pickle=False) as data:
                cells = np.asarray(data["cells"], np.int32)
                indices = np.asarray(data["cell_indices"], np.int32)
                features = np.asarray(data["features"], np.float32)
                sample = {"coords": cells, "features": np.concatenate((
                    features[indices, 0], features[indices, 1:].reshape(-1, 24)), 1),
                    "resolution": 128, "sampling_offset": .5}
            for extractor in ("mc", "flexicubes"):
                path = args.output_root / "shared" / asset["id"] / f"R128-{stage.lower()}-{extractor}"
                if all((path / name).is_file() for name in ("raw_mesh.ply", "identity.json", "metrics.json")):
                    print(json.dumps({"index": index, "stage": stage, "extractor": extractor,
                                      "status": "reused"}), flush=True)
                    continue
                if path.exists():
                    raise FileExistsError(f"Incomplete shared extraction: {path}")
                try:
                    with torch.no_grad():
                        surface = RouteSurface(sample, extractor, "cuda:0")
                        signed = surface.base.initial
                        delta = surface.base.initial_delta
                        vertices, faces = surface.extract(signed, delta)
                    if not len(vertices) or not len(faces):
                        raise RuntimeError(f"Empty {index} {stage} {extractor} surface")
                    mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy() * scale + center,
                                           faces=faces.detach().cpu().numpy(), process=False)
                    metrics = mesh_metrics(mesh, reference, samples=args.metric_samples, seed=args.seed,
                        threshold_fractions=(.005, .01), reference_points=reference_points,
                        reference_normals=reference_normals, reference_scene=reference_scene)
                    partial = path.with_name(path.name + ".partial")
                    partial.mkdir(parents=True)
                    mesh.export(partial / "raw_mesh.ply")
                    (partial / "identity.json").write_text(json.dumps({
                        "index": index, "asset_id": asset["id"], "source_sha256": asset["sha256"],
                        "a1a2_cache": row["caches"][stage], "stage": stage,
                        "extractor": extractor, "resolution": 128,
                        "sampling_offset": .5, "faces": len(mesh.faces),
                        "metric_samples": args.metric_samples, "seed": args.seed}, indent=2))
                    (partial / "metrics.json").write_text(json.dumps(metrics, indent=2))
                    partial.rename(path)
                    print(json.dumps({"index": index, "stage": stage, "extractor": extractor,
                                      "status": "success", "faces": len(mesh.faces)}), flush=True)
                except Exception as exc:
                    failure_count += 1
                    failures = args.output_root / "shared-failures.jsonl"
                    with failures.open("a") as stream:
                        stream.write(json.dumps({"index": index, "asset_id": asset["id"],
                            "stage": stage, "extractor": extractor, "error": repr(exc)}) + "\n")
                    print(json.dumps({"index": index, "stage": stage, "extractor": extractor,
                                      "status": "failed", "error": repr(exc)}), flush=True)
    if failure_count:
        raise RuntimeError(f"{failure_count} shared A1/A2 extractions failed; see shared-failures.jsonl")


if __name__ == "__main__":
    main()
