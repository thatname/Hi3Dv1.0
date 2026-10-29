"""Select a reproducible first-N mesh pilot from an Objaverse-style objects directory."""

from __future__ import annotations

import argparse
import heapq
import json
import os
from pathlib import Path

from .preflight import MESH_SUFFIXES


def select(root: Path, limit: int) -> dict:
    if limit < 1:
        raise ValueError("limit must be positive")
    objects = root / "objects"
    if not objects.is_dir():
        raise FileNotFoundError(f"Expected objects directory: {objects}")
    # Object filenames are hashes. This selects by bytewise filename order
    # without loading the entire potentially large directory into memory.
    with os.scandir(objects) as entries:
        names = heapq.nsmallest(
            limit,
            (entry.name for entry in entries
             if entry.is_file() and Path(entry.name).suffix.lower() in MESH_SUFFIXES),
        )
    if len(names) < limit:
        raise ValueError(f"Only {len(names)} supported mesh files found; requested {limit}")
    return {
        "schema_version": 1,
        "selection": {"root": str(root.resolve()), "rule": "objects/ filename bytewise ascending", "limit": limit},
        "assets": [
            {"id": Path(name).stem, "path": str((objects / name).resolve()), "category": "unclassified"}
            for name in names
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = select(args.root, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Selected {len(manifest['assets'])} assets into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
