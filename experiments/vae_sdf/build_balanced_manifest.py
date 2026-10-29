"""Build a 20 watertight-candidate / 30 non-watertight G0 pilot manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preflight import sha256
from .scan_closed_candidates import load_json
from .supplement_usable import write_json


def run(args: argparse.Namespace) -> dict:
    usable, usable_hash = load_json(args.usable_manifest)
    audit, audit_hash = load_json(args.usable_audit)
    scans = [load_json(path) for path in args.scan]
    closed_manifest, closed_hash = load_json(args.closed_manifest)
    assets = usable["assets"]
    results = audit["assets"]
    if len(assets) != 50 or len(results) != 50 or audit["counts"] != {"loaded": 50}:
        raise ValueError("Expected the audited 50-loadable source manifest")
    for index, (asset, result) in enumerate(zip(assets, results)):
        if (result["index"] != index or result["id"] != asset["id"]
                or result["path"] != asset["path"] or result["status"] != "loaded"):
            raise ValueError(f"Source manifest/audit mismatch at index {index}")
    if not scans or scans[0][0]["source"]["manifest_sha256"] != usable_hash:
        raise ValueError("The first topology scan is not linked to usable50")
    for (prior, prior_hash), (current, _) in zip(scans, scans[1:]):
        if current["source"]["previous_scan_sha256"] != prior_hash:
            raise ValueError("Topology scan chain is broken")
    if any(scan["stop_reason"] == "running" for scan, _ in scans):
        raise ValueError("Topology scan is still running")

    base_closed = [asset for asset, result in zip(assets, results)
                   if result["watertight_geometry"] is True]
    base_open = [asset for asset, result in zip(assets, results)
                 if result["watertight_geometry"] is False]
    if len(base_closed) != 6 or len(base_open) < 30:
        raise ValueError("Source audit has an unexpected topology split")
    found_closed = [row for scan, _ in scans for row in scan["assets"]
                    if row["status"] == "closed_candidate"]
    if len(found_closed) != 14:
        raise ValueError("Topology scans have not found 14 additional closed candidates")
    expected_closed = base_closed + [
        {"id": row["id"], "path": row["path"], "category": "unclassified"}
        for row in found_closed
    ]
    if closed_manifest["assets"] != expected_closed:
        raise ValueError("Closed candidate manifest differs from the scan evidence")

    selected = []
    seen = set()
    for asset, label in ([(row, "watertight_candidate") for row in expected_closed]
                         + [(row, "non_watertight") for row in base_open[:30]]):
        if asset["id"] in seen:
            raise ValueError(f"Duplicate asset ID: {asset['id']}")
        seen.add(asset["id"])
        path = Path(asset["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append({"id": asset["id"], "path": str(path),
                         "category": "unclassified", "topology_candidate": label,
                         "bytes": path.stat().st_size, "sha256": sha256(path)})
    if len(selected) != 50:
        raise ValueError("Balanced manifest must contain exactly 50 unique meshes")
    report = {"schema_version": 1,
              "selection": {"root": usable["selection"]["root"],
                            "rule": "all 20 watertight candidates; first 30 non-watertight from usable50 in source order",
                            "counts": {"watertight_candidate": 20, "non_watertight": 30},
                            "source": {
                                "usable_manifest": str(args.usable_manifest),
                                "usable_manifest_sha256": usable_hash,
                                "usable_audit": str(args.usable_audit),
                                "usable_audit_sha256": audit_hash,
                                "topology_scans": [
                                    {"path": str(path), "sha256": digest}
                                    for path, (_, digest) in zip(args.scan, scans)],
                                "closed_manifest": str(args.closed_manifest),
                                "closed_manifest_sha256": closed_hash}},
              "assets": selected}
    write_json(args.output, report)
    print(json.dumps({"count": len(selected), "selection": report["selection"]["counts"]}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usable-manifest", type=Path, required=True)
    parser.add_argument("--usable-audit", type=Path, required=True)
    parser.add_argument("--scan", type=Path, action="append", required=True,
                        help="Topology scan report in chronological order; repeat for each scan")
    parser.add_argument("--closed-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
