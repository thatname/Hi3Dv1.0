"""Record how many pool assets survive each G3 filter.

The numbers existed only in chat answers and in scattered files: the build summary
holds the manifest-stage counts without per-asset detail, the capacity policy holds
only its own 512, and the A1/A2 and A4 failures live in two different directories.
Re-run this once the run is finished to freeze the final funnel.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path,
                        default=Path("/data/hi3d/g3/converted/glb-v1/conversion-manifest.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    E = args.experiment
    build = json.loads((E / "manifest" / "build-summary.json").read_text())
    manifest = json.loads((E / "manifest" / "manifest-g3-full.json").read_text())
    assets = manifest["assets"]
    pool = {}
    if args.conversion_manifest.is_file():
        cm = json.loads(args.conversion_manifest.read_text())
        pool = {"requested_conversions": cm.get("requested"),
                "converted": cm.get("converted"),
                "conversion_failed": cm.get("failed"),
                "conversion_timeout": cm.get("timeout"),
                "source_root": cm.get("root")}

    split_path = E / "manifest" / "split-g3-full-v2-capacity.json"
    split = json.loads(split_path.read_text()) if split_path.is_file() else {}
    capacity = json.loads((E / "manifest" / "capacity-policy-v1.json").read_text())

    a1a2 = Counter()
    for path in (E / "targets" / "a1a2" / "assets").glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            a1a2["unreadable"] += 1
            continue
        if row.get("status") == "success":
            a1a2["success"] += 1
            continue
        error = (row.get("error") or "")
        if "max-sites" in error:
            a1a2["failed_support_over_max_sites"] += 1
        elif "no crossing cells" in error:
            a1a2["failed_no_crossing_cells"] += 1
        elif error.startswith("FileExistsError"):
            a1a2["failed_stale_cache"] += 1
        else:
            a1a2["failed_other"] += 1

    ledger_path = E / "targets" / "a4" / "skipped-assets.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else []
    deferred_path = E / "targets" / "a1a2" / "deferred-assets.json"
    deferred = json.loads(deferred_path.read_text()) if deferred_path.is_file() else []

    present = 0
    for asset in assets:
        base = E / "targets" / "a4" / asset["id"] / "R256"
        if base.is_dir() and any(base.glob("*/a4/target.npz")):
            present += 1

    available = len(assets)
    excluded = capacity.get("excluded_total", 0)
    trainable = available - excluded
    runtime_gaps = (a1a2["failed_support_over_max_sites"] + a1a2["failed_no_crossing_cells"]
                    + a1a2["failed_stale_cache"] + a1a2["failed_other"] + len(ledger))

    report = {
        "schema": "g3_asset_funnel_v1",
        "pool": pool,
        "manifest": {
            "candidates": build.get("rows"),
            "available": available,
            "load_failed": build.get("failed"),
            "degenerate": build.get("degenerate"),
            "converted_share": build.get("converted_share"),
            "faces_median": build.get("faces_median"),
        },
        "capacity": {
            "threshold_cells": capacity.get("threshold_cells"),
            "excluded": excluded,
            "excluded_fraction": capacity.get("excluded_fraction"),
            "trainable_total": trainable,
        },
        "split": {k: len(v) for k, v in split.items() if k.endswith("_indices")},
        "runtime": {
            "a1a2": dict(a1a2),
            "a4_skipped_ledger": len(ledger),
            "a4_gaps": len(ledger),
            "deferred_pending": len(deferred),
            "runtime_gap_total": runtime_gaps,
            "a4_targets_present": present,
            "a4_remaining": available - present,
        },
        "expected_trainable_after_gaps": trainable - runtime_gaps,
    }

    output = args.output or (E / "reports" / "asset-funnel.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())