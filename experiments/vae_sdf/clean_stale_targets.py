"""Remove A1/A2 caches left behind by workers that were killed mid-asset.

A worker writes its cache directory in place and only records a result once the
pair is complete. Killing the pipeline therefore leaves half-written caches, and
the next attempt dies on `FileExistsError: Cache already exists`. That failure is
written to `assets/NN.json`, and the producer queue is derived from "assets with
no result file", so the asset is never retried: every restart quietly drops the
assets that were in flight.

This sweeps the cache root and deletes the caches that no successful result
backs, so the assets return to the queue on the next launch. Only directories
with no success record are touched; assets are re-checked against the manifest
so a stray directory cannot delete something unrelated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def classify(cache: Path, results: Path, index_of: dict[str, int]) -> tuple[str, str, int]:
    """Return (kind, detail, bytes) for one cache directory."""

    asset_id = cache.name
    size = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
    index = index_of.get(asset_id)
    if index is None:
        return "unknown_asset", "asset id is not in the manifest", size
    result = results / f"{index:02d}.json"
    if not result.is_file():
        return "no_result", f"index {index}", size
    try:
        row = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unreadable_result", f"index {index}", size
    if row.get("status") == "success":
        return "healthy", f"index {index}", size
    error = (row.get("error") or "unknown").splitlines()[0]
    prefix = error.split(":")[0][:40]
    if error.startswith("FileExistsError") or "Incomplete A1/A2 cache pair" in error:
        return "stale_after_kill", f"index {index} {prefix}", size
    return "genuine_failure", f"index {index} {prefix}", size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a1a2-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true",
                        help="Delete the stale caches; without it only report")
    parser.add_argument("--drop-results", action="store_true",
                        help="Also delete the failed result file so the asset requeues")
    parser.add_argument("--min-age-minutes", type=float, default=15.0,
                        help="Skip caches touched more recently than this (may be in flight)")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    index_of = {asset["id"]: index for index, asset in enumerate(manifest["assets"])}
    results = args.a1a2_root / "assets"
    caches = args.a1a2_root / "targets"
    if not caches.is_dir():
        raise SystemExit(f"No cache root at {caches}")

    now = time.time()
    groups: dict[str, list[tuple[Path, str, int]]] = {}
    for cache in sorted(p for p in caches.iterdir() if p.is_dir()):
        kind, detail, size = classify(cache, results, index_of)
        groups.setdefault(kind, []).append((cache, detail, size))

    report = {}
    for kind, rows in sorted(groups.items()):
        report[kind] = {"count": len(rows), "bytes": sum(r[2] for r in rows)}
    print(json.dumps({"caches": sum(len(v) for v in groups.values()),
                      "groups": report, "apply": args.apply}, indent=2))

    # a cache with no result at all is either in flight or was interrupted; the
    # age guard keeps the sweep from deleting a live one
    candidates = groups.get("stale_after_kill", []) + groups.get("no_result", [])
    removable = [row for row in candidates
                 if now - row[0].stat().st_mtime > args.min_age_minutes * 60]
    for cache, detail, _ in groups.get("stale_after_kill", [])[:8]:
        age = (now - cache.stat().st_mtime) / 60
        print(f"  stale {cache.name[:16]}…  {detail}  idle {age:.0f} min")
    print(f"removable now: {len(removable)} (older than {args.min_age_minutes:.0f} min)")

    removed = freed = dropped = 0
    if args.apply:
        for cache, detail, size in removable:
            shutil.rmtree(cache, ignore_errors=True)
            if not cache.exists():
                removed += 1
                freed += size
            if args.drop_results:
                parts = detail.split()
                if len(parts) > 1 and parts[0] == "index":
                    result = results / f"{int(parts[1]):02d}.json"
                    if result.is_file():
                        result.unlink()
                        dropped += 1
        print(f"removed {removed} caches, freed {freed / 2 ** 30:.2f} GB, "
              f"dropped {dropped} failed results")

    if args.json_out:
        payload = {"groups": report, "removable": len(removable),
                   "removed": removed, "freed_bytes": freed, "dropped_results": dropped}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())