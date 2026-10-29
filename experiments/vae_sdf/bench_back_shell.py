"""Time one vae3d UE50/back-shell field query without training or fitting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--raycast-backend", choices=("cpu", "vulkan"), default="cpu")
    parser.add_argument("--distance-backend", choices=("cpu", "cuda", "rtx"), default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.resolution < 4:
        parser.error("resolution must be at least 4")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    asset = manifest["assets"][args.index]

    # Imports are delayed so --help works before the remote environment is ready.
    from vae3d.back_shell import back_shell_grid
    from vae3d.data import load_mesh

    started = time.perf_counter()
    mesh, center, scale = load_mesh(
        asset["path"], normalization_extent=(args.resolution - 2) / args.resolution
    )
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    field, audit = back_shell_grid(
        mesh, args.resolution, return_audit=True,
        raycast_backend=args.raycast_backend,
        distance_backend=args.distance_backend, sampling_offset=0.5,
    )
    query_seconds = time.perf_counter() - started
    report = {
        "asset_id": asset["id"], "source": asset["path"],
        "resolution": args.resolution, "faces": len(mesh.faces),
        "center": center.tolist(), "scale": scale,
        "load_seconds": load_seconds, "query_seconds": query_seconds,
        "field_min": float(field.min()), "field_max": float(field.max()),
        "audit": audit,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"asset_id": asset["id"], "resolution": args.resolution,
                      "load_seconds": load_seconds, "query_seconds": query_seconds,
                      "faces": len(mesh.faces), "report": str(args.report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
