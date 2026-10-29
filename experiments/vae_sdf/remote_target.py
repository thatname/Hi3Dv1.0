"""Read one audited A1/A2 raw mesh or scalar-field slice for the local browser."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


def target_path(batch_root: Path, index: int, stage: str) -> tuple[Path, dict]:
    batch_root = batch_root.resolve()
    summary = json.loads((batch_root / "summary.json").read_text())
    plan = json.loads((batch_root / "plan.json").read_text())
    if summary.get("denominator") != 50 or not 0 <= index < 50:
        raise ValueError("Expected a fixed 50-asset batch and valid index")
    row = summary["assets"][index]
    if row["index"] != index or row["status"] != "success":
        raise ValueError("Asset has no successful A1/A2 cache")
    cache = Path(row["caches"][stage]).resolve()
    if not cache.is_relative_to(batch_root / "targets"):
        raise ValueError("Cache path escapes the fixed batch root")
    config = json.loads((cache / "resolved_config.json").read_text())
    if (config["stage"] != stage or config["asset_id"] != row["id"]
            or config["source_sha256"] != row["source_sha256"]
            or config["manifest_sha256"] != plan["manifest_sha256"]
            or config["resolution"] != plan["resolution"]):
        raise ValueError("Cache metadata differs from the fixed batch")
    return cache, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--stage", choices=("A1", "A2"), required=True)
    parser.add_argument("--kind", choices=("mesh", "slice"), required=True)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-index", type=int, default=64)
    args = parser.parse_args()
    cache, config = target_path(args.batch_root, args.index, args.stage)

    import numpy as np

    stream = sys.stdout.buffer
    if args.kind == "mesh":
        import trimesh

        mesh = trimesh.load(cache / "raw_mesh.ply", process=False)
        vertices = np.asarray(mesh.vertices, dtype="<f4", order="C")
        faces = np.asarray(mesh.faces, dtype="<u4", order="C")
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or
                faces.ndim != 2 or faces.shape[1] != 3 or
                not np.isfinite(vertices).all() or
                (faces.size and int(faces.max()) >= len(vertices))):
            raise ValueError("Invalid raw target mesh")
        stream.write(struct.pack("<II", len(vertices), len(faces)))
        stream.write(vertices.tobytes())
        stream.write(faces.tobytes())
    else:
        field = np.load(cache / "field.npy", mmap_mode="r", allow_pickle=False)
        resolution = config["resolution"]
        if field.shape != (resolution,) * 3 or not 0 <= args.slice_index < resolution:
            raise ValueError("Invalid field shape or slice index")
        axis = "xyz".index(args.axis)
        slice_values = np.asarray(np.take(field, args.slice_index, axis=axis),
                                  dtype="<f4", order="C")
        if not np.isfinite(slice_values).all():
            raise ValueError("Nonfinite scalar-field slice")
        stream.write(struct.pack("<I", resolution))
        stream.write(slice_values.tobytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
