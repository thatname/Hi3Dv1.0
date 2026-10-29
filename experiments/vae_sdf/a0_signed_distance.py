"""Audit closed-mesh A0 candidates and export a signed Euclidean field.

Open3D ray parity supplies the sign. Its documented assumptions include a
watertight surface without mesh intersections. Automated checks here are a
screen only: no result is marked as approved ground truth without review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from .preflight import VAE3D_COMMIT, sha256
from .supplement_usable import write_json


AUDIT_SCHEMA = "g0_a0_sign_audit_v1"
CACHE_SCHEMA = "g0_a0_candidate_target_v1"


def read_bytes_json(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def pinned_head(root: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != VAE3D_COMMIT:
        raise ValueError(f"vae3d must be {VAE3D_COMMIT}, found {head}")
    return head


def checked_manifest(manifest: dict, load_audit: dict) -> list[tuple[dict, dict]]:
    assets = manifest["assets"]
    results = load_audit["assets"]
    if len(assets) != 50 or len(results) != 50 or load_audit["counts"] != {"loaded": 50}:
        raise ValueError("Expected the fully loaded balanced50 manifest")
    pairs = []
    for index, (asset, result) in enumerate(zip(assets, results)):
        if (result["index"] != index or result["id"] != asset["id"]
                or result["path"] != asset["path"] or result["status"] != "loaded"
                or (asset["topology_candidate"] == "watertight_candidate")
                != result["watertight_geometry"]):
            raise ValueError(f"Manifest and load audit differ at index {index}")
        pairs.append((asset, result))
    return pairs


def scene_for(mesh):
    import open3d as o3d

    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
    )
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def sign_probes(mesh, scene, resolution: int, seed: int) -> dict:
    import numpy as np
    import open3d as o3d

    rng = np.random.default_rng(seed)
    points = rng.uniform(-0.5, 0.5, (512, 3)).astype(np.float32)
    center = np.asarray(mesh.bounds).mean(0).astype(np.float32)
    outside = np.repeat(center[None, :], 6, axis=0)
    for axis in range(3):
        outside[2 * axis, axis] = -0.5
        outside[2 * axis + 1, axis] = 0.5
    probes = np.vstack([points, outside])
    tensor = o3d.core.Tensor(probes)
    signed1 = scene.compute_signed_distance(tensor, nsamples=1).numpy()
    signed5 = scene.compute_signed_distance(tensor, nsamples=5).numpy()
    unsigned = scene.compute_distance(tensor).numpy()
    clear = (np.abs(signed1) > 2 / resolution) & (np.abs(signed5) > 2 / resolution)
    return {
        "random_seed": seed, "random_points": len(points), "outside_points": len(outside),
        "random_inside_points": int(np.count_nonzero(signed5[:len(points)] < 0)),
        "clear_points": int(clear.sum()),
        "sign_disagreements_clear": int(np.count_nonzero(((signed1 < 0) != (signed5 < 0)) & clear)),
        "outside_nonpositive": int(np.count_nonzero(signed5[-6:] <= 0)),
        "max_unsigned_difference": float(np.max(np.abs(np.abs(signed5) - unsigned))),
        "nonfinite_values": int(np.count_nonzero(~np.isfinite(signed1))
                                + np.count_nonzero(~np.isfinite(signed5))),
    }


def audit_candidates(args: argparse.Namespace) -> dict:
    import numpy as np
    import open3d as o3d
    import trimesh
    from vae3d.data import load_mesh

    manifest, manifest_hash = read_bytes_json(args.manifest)
    load_audit, load_audit_hash = read_bytes_json(args.load_audit)
    pairs = checked_manifest(manifest, load_audit)
    head = pinned_head(args.vae3d_root)
    records = []
    metadata = {"schema": AUDIT_SCHEMA, "complete": False,
                "manifest": str(args.manifest), "manifest_sha256": manifest_hash,
                "load_audit": str(args.load_audit), "load_audit_sha256": load_audit_hash,
                "vae3d_commit": head, "resolution": args.resolution,
                "open3d_version": o3d.__version__,
                "trimesh_version": trimesh.__version__,
                "code_sha256": sha256(Path(__file__)),
                "sign_method": "Open3D ray parity, nsamples=5 for fields",
                "self_intersection_check": "not_performed",
                "manual_review_required": True}

    def report(complete: bool) -> dict:
        return dict(metadata, complete=complete, counts=dict(Counter(x["status"] for x in records)),
                    assets=records)

    for index, (asset, prior) in enumerate(pairs):
        row = {"index": index, "id": asset["id"], "path": asset["path"],
               "topology_candidate": asset["topology_candidate"]}
        started = time.perf_counter()
        try:
            if asset["topology_candidate"] != "watertight_candidate":
                row.update(status="excluded_nonclosed", faces=prior["faces"],
                           vertices=prior["vertices"],
                           reason="A0 sign is not defined as reliable for this open mesh")
            else:
                path = Path(asset["path"])
                if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
                    raise ValueError("Source mesh changed from balanced50 manifest")
                mesh, center, scale = load_mesh(
                    str(path), normalization_extent=(args.resolution - 2) / args.resolution
                )
                watertight = bool(mesh.is_watertight)
                winding = bool(mesh.is_winding_consistent)
                volume = float(mesh.volume)
                is_volume = bool(mesh.is_volume)
                row.update(faces=len(mesh.faces), vertices=len(mesh.vertices),
                           watertight=watertight, winding_consistent=winding,
                           is_volume=is_volume, signed_volume=volume,
                           body_count=int(mesh.body_count),
                           euler_number=int(mesh.euler_number),
                           center=np.asarray(center).tolist(), scale=scale)
                if not watertight:
                    row.update(status="excluded_nonclosed",
                               reason="Watertight status changed after reloading")
                else:
                    row["sign_probes"] = sign_probes(
                        mesh, scene_for(mesh), args.resolution,
                        int(hashlib.sha256(asset["id"].encode()).hexdigest()[:8], 16)
                    )
                    probes = row["sign_probes"]
                    concerns = []
                    if not winding:
                        concerns.append("inconsistent winding")
                    if not is_volume or volume <= 0:
                        concerns.append("nonpositive or invalid oriented volume")
                    if row["body_count"] > 1:
                        concerns.append("multiple bodies; component intersections or nesting not checked")
                    if probes["sign_disagreements_clear"]:
                        concerns.append("ray vote disagreement away from surface")
                    if probes["outside_nonpositive"]:
                        concerns.append("outside probe classified inside")
                    if not probes["random_inside_points"]:
                        concerns.append("random sign probes missed the interior")
                    if probes["nonfinite_values"] or probes["max_unsigned_difference"] > 1e-4:
                        concerns.append("nonfinite or inconsistent distance")
                    row.update(status="needs_review" if concerns else "candidate",
                               concerns=concerns, manual_review_required=True)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = time.perf_counter() - started
        records.append(row)
        write_json(args.report, report(False))
        print(f"A0 audit {index + 1}/{len(pairs)}: {row['status']} {asset['id']} "
              f"{row['seconds']:.2f}s", flush=True)
    final = report(True)
    write_json(args.report, final)
    print(json.dumps(final["counts"]))
    return final


def prepare_candidate(args: argparse.Namespace) -> dict:
    import numpy as np
    import open3d as o3d
    from vae3d.data import load_mesh
    from .prepare_initial_targets import candidate_cells, complete_support, raw_mesh

    manifest, manifest_hash = read_bytes_json(args.manifest)
    audit, audit_hash = read_bytes_json(args.audit_report)
    if (audit.get("schema") != AUDIT_SCHEMA or not audit.get("complete")
            or audit.get("manifest_sha256") != manifest_hash):
        raise ValueError("A0 audit is incomplete or belongs to another manifest")
    if not 0 <= args.index < len(manifest["assets"]):
        raise IndexError("index outside manifest")
    asset = manifest["assets"][args.index]
    row = audit["assets"][args.index]
    if row["id"] != asset["id"] or row["path"] != asset["path"]:
        raise ValueError("A0 audit asset does not match manifest")
    if row["status"] not in {"candidate", "needs_review"}:
        raise ValueError(f"A0 asset is not a closed topology candidate: {row['status']}")
    review_source = None
    sign_status = "candidate_unverified"
    if args.review_file:
        review, review_hash = read_bytes_json(args.review_file)
        if (review.get("schema") != "g0_a0_manual_review_v1"
                or review.get("manifest_sha256") != manifest_hash
                or review.get("a0_audit_sha256") != audit_hash):
            raise ValueError("Manual review does not match this manifest and A0 audit")
        approvals = [entry for entry in review.get("approved", [])
                     if entry.get("index") == args.index]
        if (len(approvals) != 1 or approvals[0].get("id") != asset["id"]
                or approvals[0].get("source_sha256") != asset["sha256"]
                or not approvals[0].get("reviewer") or not approvals[0].get("note")):
            raise ValueError("This asset has no complete, matching manual A0 approval")
        review_source = {"path": str(args.review_file), "sha256": review_hash,
                         "reviewer": approvals[0]["reviewer"], "note": approvals[0]["note"]}
        sign_status = "review_approved"
    elif row["status"] != "candidate" or not args.allow_candidate:
        raise ValueError("Manual sign review is pending; --allow-candidate permits only an unverified smoke cache")
    head = pinned_head(args.vae3d_root)
    if head != audit["vae3d_commit"]:
        raise ValueError("vae3d revision differs from A0 audit")
    source = Path(asset["path"])
    if source.stat().st_size != asset["bytes"] or sha256(source) != asset["sha256"]:
        raise ValueError("Source mesh differs from the fixed manifest")
    started = time.perf_counter()
    mesh, center, scale = load_mesh(
        str(source), normalization_extent=(args.resolution - 2) / args.resolution
    )
    if not mesh.is_watertight or not mesh.is_winding_consistent or not mesh.is_volume:
        raise ValueError("Mesh no longer satisfies closed candidate checks")
    load_seconds = time.perf_counter() - started
    scene = scene_for(mesh)
    n = args.resolution
    field = np.empty((n, n, n), np.float32)
    yz = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    started = time.perf_counter()
    for x in range(n):
        points = (np.column_stack([np.full(len(yz), x), yz]).astype(np.float32)
                  + 0.5) / n - 0.5
        values = scene.compute_signed_distance(o3d.core.Tensor(points), nsamples=args.nsamples)
        field[x] = values.numpy().reshape(n, n)
        if x % 32 == 0:
            print(f"A0 signed distance: {x}/{n}", flush=True)
    query_seconds = time.perf_counter() - started
    if not np.isfinite(field).all():
        raise ValueError("Nonfinite A0 field")
    mask, crossing = candidate_cells(field)
    cells, coords, indices = complete_support(mask)
    if len(coords) > args.max_sites:
        raise ValueError(f"A0 support has {len(coords)} points, exceeds max-sites")
    features = np.zeros((len(coords), 4), np.float32)
    features[:, 0] = np.clip(field[tuple(coords.T)] * n, -4, 4)
    surface = raw_mesh(field, n)
    surface.vertices = np.asarray(surface.vertices) * scale + center

    module_dir = Path(__file__).parent
    config = {"schema": CACHE_SCHEMA, "stage": "A0", "asset_id": asset["id"],
              "source": str(source), "source_sha256": asset["sha256"],
              "manifest": str(args.manifest), "manifest_sha256": manifest_hash,
              "manifest_index": args.index, "a0_audit": str(args.audit_report),
              "a0_audit_sha256": audit_hash, "sign_status": sign_status,
              "manual_review": review_source,
              "vae3d_commit": head, "open3d_version": o3d.__version__,
              "code_sha256": {name: sha256(module_dir / name)
                              for name in ("a0_signed_distance.py", "prepare_initial_targets.py")},
              "resolution": n, "sampling_offset": 0.5, "nsamples": args.nsamples,
              "normalization_center": np.asarray(center).tolist(),
              "normalization_scale": scale, "support_rule": "A0 native near/crossing, dilate 2, complete cells",
              "max_sites": args.max_sites}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    label = "approved" if sign_status == "review_approved" else "candidate"
    destination = args.output_root / asset["id"] / f"R{n}" / f"A0-{label}-{key[:12]}"
    if destination.exists():
        raise FileExistsError(f"Cache already exists: {destination}")
    destination.mkdir(parents=True)
    np.save(destination / "field.npy", field)
    np.savez_compressed(destination / "target.npz", coords=coords, features=features,
                        cells=cells, cell_indices=indices, resolution=np.int32(n),
                        sampling_offset=np.float32(0.5), stage=np.array("A0"),
                        field_layout=np.array("center_points_sdf_delta_v1"),
                        field_units=np.array("fine_voxel"), delta_units=np.array("fine_voxel"))
    surface.export(destination / "raw_mesh.ply")
    result = {"stage": "A0", "cache_key": key, "sign_status": sign_status,
              "field_min": float(field.min()), "field_max": float(field.max()),
              "negative_samples": int((field < 0).sum()), "crossing_cells": crossing,
              "support_cells": len(cells), "support_points": len(coords),
              "raw_mesh_vertices": len(surface.vertices), "raw_mesh_faces": len(surface.faces),
              "load_seconds": load_seconds, "query_seconds": query_seconds,
              "raw_mesh_extractor": "skimage.measure.marching_cubes (Lewiner)",
              "raw_mesh_coordinate_frame": "source", "raw_mesh_postprocessing": "none"}
    (destination / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {"cache": str(destination), "audit": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Audit all 50 balanced50 entries for A0 eligibility")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--load-audit", type=Path, required=True)
    audit.add_argument("--vae3d-root", type=Path, required=True)
    audit.add_argument("--resolution", type=int, default=128)
    audit.add_argument("--report", type=Path, required=True)
    prepare = subparsers.add_parser("prepare", help="Export one explicitly unverified A0 candidate")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--audit-report", type=Path, required=True)
    prepare.add_argument("--index", type=int, required=True)
    prepare.add_argument("--vae3d-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--resolution", type=int, default=128)
    prepare.add_argument("--nsamples", type=int, default=5)
    prepare.add_argument("--max-sites", type=int, default=2_000_000)
    review_mode = prepare.add_mutually_exclusive_group(required=True)
    review_mode.add_argument("--allow-candidate", action="store_true",
                             help="Export an unverified smoke cache, never official A0 ground truth")
    review_mode.add_argument("--review-file", type=Path,
                             help="Manual approval JSON bound to this manifest and audit")
    args = parser.parse_args()
    if args.resolution < 4 or (args.command == "prepare" and args.nsamples not in (1, 3, 5, 7)):
        parser.error("resolution must be >=4 and nsamples an odd value in {1,3,5,7}")
    result = audit_candidates(args) if args.command == "audit" else prepare_candidate(args)
    if args.command == "prepare":
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
