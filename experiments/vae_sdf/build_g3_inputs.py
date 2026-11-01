"""Build the G3 manifest, load audit and stratified split from the Objaverse pool.

One record per asset id, preferring the converted geometry GLB when the source
format cannot be read by trimesh. Each record carries the source face count so the
split can be stratified by tessellation instead of filtering it away, and the
load audit is emitted in the schema `run_initial_batch` expects at the chosen
resolution.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np

from vae3d.data import load_mesh

READABLE = {".obj", ".glb", ".gltf", ".stl", ".ply"}
CONVERTED = {".fbx", ".blend", ".dae"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chosens(root: Path, converted: Path):
    """Yield (asset_id, path, source_suffix, was_converted) for every usable asset."""

    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        asset_id = path.stem
        if suffix in CONVERTED:
            candidate = converted / f"{asset_id}.glb"
            if candidate.is_file():
                yield asset_id, candidate, suffix, True
        elif suffix in READABLE:
            yield asset_id, path, suffix, False


def inspect(item, resolution: int) -> dict:
    asset_id, path, source_suffix, was_converted = item
    row = {"id": asset_id, "path": str(path), "sha256": sha256(path),
           "source_suffix": source_suffix, "converted": was_converted,
           "bytes": path.stat().st_size}
    try:
        mesh, center, scale = load_mesh(str(path),
                                        normalization_extent=(resolution - 2) / resolution)
        extent = np.asarray(mesh.bounds[1], dtype=float) - np.asarray(mesh.bounds[0])
        row.update({
            "status": "loaded",
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "normalization_scale": float(scale),
            "extent_max": float(extent.max()),
            "extent_min": float(extent.min()),
        })
        if len(mesh.faces) < 50 or extent.max() <= 0:
            row["status"] = "degenerate"
    except Exception as error:  # noqa: BLE001 - the audit must record every failure
        row.update({"status": "failed",
                    "error": f"{type(error).__name__}: {error}"[:240],
                    "traceback": traceback.format_exc()[-300:]})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/TRELLIS/datasets/ObjaverseXL_github/"
                                     "raw/github/objects"))
    parser.add_argument("--converted", type=Path,
                        default=Path("/data/hi3d/g3/converted/glb-v1"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--strata", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole pool")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    items = list(chosens(args.root, args.converted))
    if args.limit:
        items = items[:args.limit]
    print(json.dumps({"candidates": len(items), "workers": args.workers}), flush=True)

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, row in enumerate(pool.map(lambda item: inspect(item, args.resolution),
                                            items), 1):
            rows.append(row)
            if done % 2000 == 0:
                print(json.dumps({"inspected": done,
                                  "loaded": sum(1 for r in rows if r["status"] == "loaded")}),
                      flush=True)

    ok = [r for r in rows if r["status"] == "loaded"]
    ok.sort(key=lambda r: r["id"])
    manifest = {
        "schema_version": 2,
        "selection": {"root": str(args.root.resolve()),
                      "converted": str(args.converted.resolve()),
                      "rule": "one record per asset id; converted GLB preferred for "
                              "fbx/blend/dae; failing or degenerate meshes dropped",
                      "resolution": args.resolution,
                      "candidates": len(items), "loaded": len(ok)},
        "assets": [{"id": r["id"], "path": r["path"], "sha256": r["sha256"],
                    "category": "unclassified",
                    "faces": r["faces"], "vertices": r["vertices"],
                    "source_suffix": r["source_suffix"], "converted": r["converted"]}
                   for r in ok],
    }
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    manifest_path = args.output_root / "manifest-g3-full.json"
    manifest_path.write_text(manifest_bytes, encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes.encode()).hexdigest()

    audit = {
        "schema": "g0_load_audit_v1",
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "resolution": args.resolution,
        "counts": {status: sum(1 for r in rows if r["status"] == status)
                   for status in ("loaded", "failed", "degenerate")},
        "assets": [{"index": index, "id": asset["id"], "path": asset["path"],
                    "status": "loaded"} for index, asset in enumerate(manifest["assets"])],
    }
    (args.output_root / "load-audit-g3-full.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Stratify the split by face-count quantile so no tessellation band is
    # over- or under-represented in validation and test.
    faces = np.array([asset["faces"] for asset in manifest["assets"]], dtype=float)
    order = np.argsort(faces, kind="stable")
    strata = np.empty(len(order), dtype=int)
    for part, chunk in enumerate(np.array_split(order, args.strata)):
        strata[chunk] = part
    rng = np.random.default_rng(args.seed)
    train, val, test = [], [], []
    for part in range(args.strata):
        members = [int(i) for i in order if strata[i] == part]
        rng.shuffle(members)
        n_val = max(1, int(round(len(members) * (1 - args.train_fraction) / 2)))
        n_test = max(1, int(round(len(members) * (1 - args.train_fraction) / 2)))
        test.extend(members[:n_test])
        val.extend(members[n_test:n_test + n_val])
        train.extend(members[n_test + n_val:])
    train.sort(), val.sort(), test.sort()
    split = {
        "schema": "g1_asset_split_v1",
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "rule": f"face-count {args.strata}-quantile strata, "
                f"{args.train_fraction:.0%}/{1 - args.train_fraction:.0%} split inside every "
                "stratum; tessellation is stratified rather than filtered so the pool "
                "distribution survives into the held-out sets",
        "train_indices": train, "train_ids": [manifest["assets"][i]["id"] for i in train],
        "val_indices": val, "val_ids": [manifest["assets"][i]["id"] for i in val],
        "test_indices": test, "test_ids": [manifest["assets"][i]["id"] for i in test],
        "strata": {str(part): {"train": sum(1 for i in train if strata[i] == part),
                               "val": sum(1 for i in val if strata[i] == part),
                               "test": sum(1 for i in test if strata[i] == part),
                               "faces_min": int(faces[strata == part].min()),
                               "faces_max": int(faces[strata == part].max())}
                   for part in range(args.strata)},
    }
    (args.output_root / "split-g3-full-v1.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {"rows": len(rows), "loaded": len(ok),
               "failed": audit["counts"]["failed"],
               "degenerate": audit["counts"]["degenerate"],
               "converted_share": sum(1 for a in manifest["assets"] if a["converted"]) / max(len(ok), 1),
               "faces_median": int(np.median(faces)) if len(faces) else None,
               "faces_p10": int(np.percentile(faces, 10)) if len(faces) else None,
               "faces_p90": int(np.percentile(faces, 90)) if len(faces) else None,
               "split": {role: len(value) for role, value in
                         (("train", train), ("val", val), ("test", test))},
               "manifest": str(manifest_path), "manifest_sha256": manifest_sha}
    (args.output_root / "build-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())