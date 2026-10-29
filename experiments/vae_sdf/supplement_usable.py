"""Extend the first-50 pilot to 50 loadable meshes without changing its audit."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import time
from collections import Counter
from pathlib import Path

from .preflight import MESH_SUFFIXES, sha256


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate_names(objects: Path, after: str):
    """Yield supported filenames after the pilot in bytewise order, in growing batches."""
    limit = 64
    yielded: set[str] = set()
    while True:
        with os.scandir(objects) as entries:
            names = heapq.nsmallest(
                limit,
                (entry.name for entry in entries
                 if entry.is_file() and entry.name > after
                 and Path(entry.name).suffix.lower() in MESH_SUFFIXES),
            )
        for name in names:
            if name not in yielded:
                yielded.add(name)
                yield name
        if len(names) < limit:
            return
        limit *= 2


def run(args: argparse.Namespace) -> dict:
    original_bytes = args.manifest.read_bytes()
    original_audit_bytes = args.audit.read_bytes()
    original = json.loads(original_bytes)
    audit = json.loads(original_audit_bytes)
    preflight = read_json(args.preflight)
    assets = original["assets"]
    results = audit["assets"]
    if len(assets) != 50 or len(results) != 50 or audit["resolution"] != args.resolution:
        raise ValueError("Expected the audited first-50 pilot at the requested resolution")
    expected = {row["path"]: row for row in preflight["assets"]}
    for index, (asset, result) in enumerate(zip(assets, results)):
        path = Path(asset["path"])
        if (result["index"] != index or result["id"] != asset["id"]
                or result["path"] != asset["path"]):
            raise ValueError(f"Original manifest/audit mismatch at index {index}")
        prior = expected.get(asset["path"])
        if not prior or not path.is_file() or path.stat().st_size != prior["bytes"]:
            raise ValueError(f"Original mesh missing or size changed: {path}")
        if sha256(path) != prior["sha256"]:
            raise ValueError(f"Original mesh content changed: {path}")
    root = Path(original["selection"]["root"])
    objects = root / "objects"
    if not objects.is_dir():
        raise FileNotFoundError(objects)
    after = Path(assets[-1]["path"]).name
    accepted = []
    checked = []
    from vae3d.data import load_mesh

    for name in candidate_names(objects, after):
        path = objects / name
        started = time.perf_counter()
        record = {"id": path.stem, "path": str(path.resolve())}
        try:
            mesh, center, scale = load_mesh(
                str(path), normalization_extent=(args.resolution - 2) / args.resolution
            )
            record.update(status="loaded", faces=len(mesh.faces), vertices=len(mesh.vertices),
                          watertight_geometry=bool(mesh.is_watertight),
                          winding_consistent=bool(mesh.is_winding_consistent),
                          center=center.tolist(), scale=scale,
                          bytes=path.stat().st_size, sha256=sha256(path))
            accepted.append(record)
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record["seconds"] = time.perf_counter() - started
        checked.append(record)
        print(f"candidate {len(checked)}: {record['status']} {name} "
              f"({len(accepted)}/8 accepted, {record['seconds']:.2f}s)", flush=True)
        if len(accepted) == 8:
            break
    if len(accepted) != 8:
        raise RuntimeError(f"Only {len(accepted)} loadable supplements found")

    retained = [(asset, result) for asset, result in zip(assets, results)
                if result["status"] == "loaded"]
    if len(retained) != 42:
        raise ValueError(f"Expected 42 original loadable meshes, found {len(retained)}")
    combined_assets = [asset for asset, _ in retained]
    combined_results = [dict(result, index=index) for index, (_, result) in enumerate(retained)]
    for record in accepted:
        index = len(combined_assets)
        combined_assets.append({"id": record["id"], "path": record["path"],
                                "category": "unclassified"})
        combined_results.append({key: value for key, value in record.items()
                                 if key not in {"bytes", "sha256"}} | {"index": index})

    source = {"manifest": str(args.manifest),
              "manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
              "audit": str(args.audit),
              "audit_sha256": hashlib.sha256(original_audit_bytes).hexdigest(),
              "preflight": str(args.preflight)}
    manifest = {"schema_version": 1,
                "selection": {"root": str(root),
                              "rule": "42 loadable from first50, then first 8 loadable by filename order after first50",
                              "limit": 50, "source": source},
                "assets": combined_assets}
    combined_audit = {"schema_version": 1, "resolution": args.resolution,
                      "counts": dict(Counter(row["status"] for row in combined_results)),
                      "assets": combined_results}
    supplement = {"schema_version": 1, "source": source, "resolution": args.resolution,
                  "rule": "filename bytewise ascending after original first50",
                  "original_loaded": len(retained), "needed": 8,
                  "checked": len(checked), "counts": dict(Counter(row["status"] for row in checked)),
                  "assets": checked}
    write_json(args.output_manifest, manifest)
    write_json(args.output_audit, combined_audit)
    write_json(args.output_supplement, supplement)
    return supplement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--output-supplement", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"checked": report["checked"], "counts": report["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
