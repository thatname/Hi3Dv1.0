"""Read-only mesh load audit for a fixed G0 manifest."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.resolution < 4:
        parser.error("resolution must be at least 4")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    from vae3d.data import load_mesh

    results = []
    for index, asset in enumerate(manifest["assets"]):
        started = time.perf_counter()
        result = {"index": index, "id": asset["id"], "path": asset["path"]}
        try:
            mesh, center, scale = load_mesh(
                asset["path"], normalization_extent=(args.resolution - 2) / args.resolution
            )
            result.update(
                status="loaded", faces=len(mesh.faces), vertices=len(mesh.vertices),
                watertight_geometry=bool(mesh.is_watertight),
                winding_consistent=bool(mesh.is_winding_consistent),
                center=center.tolist(), scale=scale,
            )
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        result["seconds"] = time.perf_counter() - started
        results.append(result)
        print(f"{index + 1}/{len(manifest['assets'])}: {result['status']} "
              f"{asset['id']} {result['seconds']:.2f}s", flush=True)
    report = {"schema_version": 1, "resolution": args.resolution,
              "counts": dict(Counter(x["status"] for x in results)), "assets": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"]))
    return 0 if report["counts"].get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
