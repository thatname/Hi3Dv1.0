"""Deterministic raw-mesh metrics for the G0 A1/A2 batch baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROTOCOL = "g0_a1a2_surface_metrics_v1"


def sample_surface(mesh, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted points and their face normals, without global RNG state."""
    area = np.asarray(mesh.area_faces, dtype=np.float64)
    total = float(area.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Mesh has no positive-area triangles")
    rng = np.random.default_rng(seed)
    faces = rng.choice(len(area), size=count, p=area / total)
    triangle = np.asarray(mesh.triangles)[faces]
    u, v = rng.random((2, count))
    root = np.sqrt(u)
    points = ((1 - root)[:, None] * triangle[:, 0]
              + (root * (1 - v))[:, None] * triangle[:, 1]
              + (root * v)[:, None] * triangle[:, 2])
    return points.astype(np.float32), np.asarray(mesh.face_normals)[faces].astype(np.float32)


def scene_for(mesh):
    import open3d as o3d

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    if not len(faces):
        raise ValueError("Mesh has no triangles")
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices), o3d.core.Tensor(faces)
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene


def closest(points: np.ndarray, target, scene, batch_size: int = 8192):
    import open3d as o3d

    distances, normals = [], []
    face_normals = np.asarray(target.face_normals, dtype=np.float32)
    for begin in range(0, len(points), batch_size):
        chunk = points[begin:begin + batch_size]
        hit = scene.compute_closest_points(o3d.core.Tensor(chunk))
        ids = hit["primitive_ids"].numpy().astype(np.int64)
        if np.any(ids >= len(face_normals)):
            raise ValueError("Open3D returned an invalid nearest triangle")
        q = hit["points"].numpy()
        distances.append(np.linalg.norm(chunk - q, axis=1))
        normals.append(face_normals[ids])
    return np.concatenate(distances), np.concatenate(normals)


def mesh_metrics(predicted, reference, *, samples: int, seed: int,
                 threshold_fractions: tuple[float, ...],
                 reference_points: np.ndarray | None = None,
                 reference_normals: np.ndarray | None = None,
                 reference_scene=None) -> dict:
    import trimesh

    if samples < 1 or any(t <= 0 for t in threshold_fractions):
        raise ValueError("Samples and thresholds must be positive")
    diagonal = float(np.linalg.norm(reference.bounds[1] - reference.bounds[0]))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Reference has zero bounding-box diagonal")
    out = {"reference_bbox_diagonal": diagonal,
           "vertices": int(len(predicted.vertices)), "faces": int(len(predicted.faces))}
    if not len(predicted.faces) or predicted.area <= 0:
        out.update(empty=True, chamfer_l1=None, normal_consistency=None,
                   fscore={str(t): 0.0 for t in threshold_fractions})
        return out
    if reference_points is None:
        reference_points, reference_normals = sample_surface(reference, samples, seed)
    if reference_scene is None:
        reference_scene = scene_for(reference)
    predicted_points, predicted_normals = sample_surface(predicted, samples, seed + 1)
    predicted_scene = scene_for(predicted)
    pred_distance, nearest_reference_normals = closest(predicted_points, reference, reference_scene)
    ref_distance, nearest_predicted_normals = closest(reference_points, predicted, predicted_scene)
    if not (np.all(np.isfinite(pred_distance)) and np.all(np.isfinite(ref_distance))):
        raise ValueError("Nonfinite surface distances")
    normal = (np.abs(np.einsum("ij,ij->i", predicted_normals, nearest_reference_normals)).mean()
              + np.abs(np.einsum("ij,ij->i", reference_normals, nearest_predicted_normals)).mean()) / 2
    out.update(empty=False,
               chamfer_l1=float((pred_distance.mean() + ref_distance.mean()) / 2),
               mean_pred_to_source=float(pred_distance.mean()),
               mean_source_to_pred=float(ref_distance.mean()),
               normal_consistency=float(normal),
               watertight=bool(predicted.is_watertight),
               fscore={})
    edges, counts = np.unique(np.sort(predicted.edges, axis=1), axis=0, return_counts=True)
    out["boundary_edges"] = int((counts == 1).sum())
    out["nonmanifold_edges"] = int((counts > 2).sum())
    for fraction in threshold_fractions:
        threshold = diagonal * fraction
        precision = float((pred_distance < threshold).mean())
        recall = float((ref_distance < threshold).mean())
        out["fscore"][str(fraction)] = {
            "threshold_source_units": threshold,
            "precision": precision, "recall": recall,
            "fscore": 2 * precision * recall / max(precision + recall, 1e-12),
        }
    return out


def evaluate_pair(a1_path: Path, a2_path: Path, source: Path, *, resolution: int,
                  samples: int = 4096, seed: int = 20261029,
                  threshold_fractions: tuple[float, ...] = (0.005, 0.01)) -> dict:
    import trimesh
    from vae3d.data import load_mesh
    from .check_initial_targets import check

    cache_check = check(a1_path, a2_path)
    normalized, center, scale = load_mesh(
        str(source), normalization_extent=(resolution - 2) / resolution
    )
    reference = trimesh.Trimesh(
        vertices=np.asarray(normalized.vertices) * scale + center,
        faces=np.asarray(normalized.faces), process=False,
    )
    ref_points, ref_normals = sample_surface(reference, samples, seed)
    ref_scene = scene_for(reference)
    result = {"protocol": PROTOCOL, "source": str(source), "samples_per_direction": samples,
              "seed": seed, "threshold_fractions": list(threshold_fractions),
              "reference_faces": int(len(reference.faces)), "cache_check": cache_check,
              "stages": {}}
    for stage, path in (("A1", a1_path), ("A2", a2_path)):
        mesh = trimesh.load(path / "raw_mesh.ply", process=False)
        result["stages"][stage] = mesh_metrics(
            mesh, reference, samples=samples, seed=seed,
            threshold_fractions=threshold_fractions,
            reference_points=ref_points, reference_normals=ref_normals,
            reference_scene=ref_scene,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    print(json.dumps(evaluate_pair(args.a1, args.a2, args.source,
                                   resolution=args.resolution, samples=args.samples,
                                   seed=args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
