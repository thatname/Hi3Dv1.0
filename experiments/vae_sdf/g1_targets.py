"""Select an A4 target from the legacy batch or a fixed four-route batch."""

from __future__ import annotations

import json
from pathlib import Path


ROUTE = "flexicubes_smooth"


def target_rows(path: Path, manifest: dict, *, four_route: bool) -> list[dict]:
    summary = json.loads(path.read_text())
    assets = manifest["assets"]
    rows = summary["assets"]
    if len(assets) != 50 or len(rows) != 50:
        raise ValueError("G1 requires the fixed balanced-50 target batch")
    if not four_route:
        return rows
    if (summary.get("schema") != "g0_four_route_paired_summary_v1"
            or summary.get("paired_indices") != list(range(50))
            or Path(summary.get("root", "")).resolve() != path.parent.resolve()):
        raise ValueError("Four-route summary is not the complete fixed-50 protocol")
    selected = []
    for index, (asset, row) in enumerate(zip(assets, rows)):
        if (row["index"] != index or row["asset_id"] != asset["id"]
                or row["source_sha256"] != asset["sha256"]):
            raise ValueError(f"Four-route asset identity mismatch at {index}")
        route = row["routes"][ROUTE]
        if route["status"] != "success":
            raise ValueError(f"Missing {ROUTE} target at {index}")
        cache_root = Path(route["cache"])
        if not cache_root.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError(f"Four-route cache is outside the selected batch at {index}")
        selected.append({
            "index": index, "id": asset["id"], "status": "success",
            "cache": str(cache_root / "a4"), "config_root": str(cache_root),
            "source_sha256": row["source_sha256"],
            "a0_status": None, "a0_concerns": None,
            "audit": route["a4"]["audit"], "route_id": ROUTE,
        })
    return selected
