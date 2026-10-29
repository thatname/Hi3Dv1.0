"""Scan at most N subsequent mesh files for additional watertight candidates.

The existing 50-loadable manifest is read only. A successful load with
``mesh.is_watertight`` is only a topology candidate, not a verified solid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from .preflight import sha256
from .supplement_usable import candidate_names, write_json


def load_json(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def run(args: argparse.Namespace) -> dict:
    if args.limit < 1 or args.needed < 1 or args.resolution < 4:
        raise ValueError("limit, needed and resolution must be positive")
    manifest, manifest_hash = load_json(args.manifest)
    audit, audit_hash = load_json(args.audit)
    supplement, supplement_hash = load_json(args.previous_scan)
    existing = manifest["assets"]
    checked_existing = audit["assets"]
    if (len(existing) != 50 or len(checked_existing) != 50
            or audit["resolution"] != args.resolution):
        raise ValueError("Expected the 50-loadable manifest and matching audit")
    for index, (asset, result) in enumerate(zip(existing, checked_existing)):
        if (result["index"] != index or result["id"] != asset["id"]
                or result["path"] != asset["path"] or result["status"] != "loaded"):
            raise ValueError(f"Manifest/audit mismatch at index {index}")
    prior_rows = supplement["assets"]
    if not prior_rows:
        raise ValueError("Previous scan has no assets")
    after = Path(prior_rows[-1]["path"]).name
    if (prior_rows[-1]["id"] != existing[-1]["id"]
            and supplement.get("source", {}).get("manifest_sha256") != manifest_hash):
        raise ValueError("Previous scan is not linked to the usable50 manifest")

    root = Path(manifest["selection"]["root"])
    objects = root / "objects"
    if not objects.is_dir():
        raise FileNotFoundError(objects)
    source = {
        "manifest": str(args.manifest), "manifest_sha256": manifest_hash,
        "audit": str(args.audit), "audit_sha256": audit_hash,
        "previous_scan": str(args.previous_scan), "previous_scan_sha256": supplement_hash,
    }
    base_closed = [asset for asset, result in zip(existing, checked_existing)
                   if result["watertight_geometry"] is True]
    existing_closed = base_closed
    if args.existing_candidates:
        candidates, candidates_hash = load_json(args.existing_candidates)
        existing_closed = candidates["assets"]
        if (len(existing_closed) < len(base_closed)
                or existing_closed[:len(base_closed)] != base_closed
                or len({row["id"] for row in existing_closed}) != len(existing_closed)):
            raise ValueError("Existing candidate manifest does not extend the base candidates")
        prior_closed = {row["id"] for row in prior_rows
                        if row.get("status") == "closed_candidate"}
        if not prior_closed.issubset({row["id"] for row in existing_closed}):
            raise ValueError("Previous scan's closed candidates are missing")
        source["existing_candidates"] = str(args.existing_candidates)
        source["existing_candidates_sha256"] = candidates_hash
    scanned: list[dict] = []
    from vae3d.data import load_mesh

    def report(stop_reason: str) -> dict:
        counts = dict(Counter(row["status"] for row in scanned))
        return {"schema_version": 1, "source": source, "resolution": args.resolution,
                "selection": {"root": str(root), "after_filename": after,
                              "rule": "next supported filenames in bytewise ascending order",
                              "max_checked": args.limit, "needed_closed": args.needed},
                "existing_closed_candidates": len(existing_closed),
                "checked": len(scanned), "new_closed_candidates": counts.get("closed_candidate", 0),
                "counts": counts, "stop_reason": stop_reason, "assets": scanned}

    for name in candidate_names(objects, after):
        if len(scanned) >= args.limit:
            break
        path = objects / name
        started = time.perf_counter()
        row = {"id": path.stem, "path": str(path.resolve())}
        try:
            mesh, center, scale = load_mesh(
                str(path), normalization_extent=(args.resolution - 2) / args.resolution
            )
            closed = bool(mesh.is_watertight)
            row.update(status="closed_candidate" if closed else "nonclosed",
                       faces=len(mesh.faces), vertices=len(mesh.vertices),
                       watertight_geometry=closed,
                       winding_consistent=bool(mesh.is_winding_consistent),
                       center=center.tolist(), scale=scale)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row["bytes"] = path.stat().st_size
        row["sha256"] = sha256(path)
        row["seconds"] = time.perf_counter() - started
        scanned.append(row)
        closed_count = sum(item["status"] == "closed_candidate" for item in scanned)
        print(f"{len(scanned)}/{args.limit}: {row['status']} {name} "
              f"({closed_count}/{args.needed} new closed, {row['seconds']:.2f}s)", flush=True)
        write_json(args.output_report, report("running"))
        if closed_count >= args.needed:
            break

    reason = ("target_reached" if sum(row["status"] == "closed_candidate" for row in scanned)
              >= args.needed else "limit_reached" if len(scanned) == args.limit
              else "source_exhausted")
    final = report(reason)
    write_json(args.output_report, final)
    new_closed = [row for row in scanned if row["status"] == "closed_candidate"]
    pending = [row for row in scanned if row["status"] == "nonclosed"]
    write_json(args.output_manifest, {
        "schema_version": 1,
        "selection": {"root": str(root), "rule": "existing watertight candidates plus new scan candidates",
                      "source_report": str(args.output_report), "stop_reason": reason,
                      "existing_count": len(existing_closed), "new_count": len(new_closed)},
        "assets": existing_closed + [
            {"id": row["id"], "path": row["path"], "category": "unclassified"}
            for row in new_closed],
    })
    write_json(args.output_pending, {
        "schema_version": 1, "source_report": str(args.output_report),
        "count": len(pending), "assets": pending,
    })
    print(json.dumps({"checked": final["checked"], "counts": final["counts"],
                      "stop_reason": reason, "total_closed_candidates": len(existing_closed) + len(new_closed)}))
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--previous-scan", type=Path, required=True)
    parser.add_argument("--existing-candidates", type=Path,
                        help="Prior closed-candidate manifest when continuing a scan")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--needed", type=int, default=19)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-pending", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    return 0 if result["stop_reason"] == "target_reached" else 2


if __name__ == "__main__":
    raise SystemExit(main())
