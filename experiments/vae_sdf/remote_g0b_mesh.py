"""Stream one audited G0b raw extraction mesh to the local comparison browser."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


MESH_NAMES = {"mc": "mc_raw_mesh.ply", "flexicubes": "flexicubes_raw_mesh.ply"}


def cache_for(root: Path, manifest_path: Path, index: int) -> Path:
    root = root.resolve()
    manifest_bytes = manifest_path.read_bytes()
    assets = json.loads(manifest_bytes)["assets"]
    summary = json.loads((root / "summary.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    if (not 0 <= index < len(assets) or len(assets) != 50
            or summary.get("denominator") != 50
            or len(summary.get("assets", [])) != 50
            or plan.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()):
        raise ValueError("G0b batch differs from the fixed manifest")
    asset, row = assets[index], summary["assets"][index]
    if (row.get("index") != index or row.get("id") != asset["id"]
            or row.get("source_sha256") != asset["sha256"]
            or row.get("status") != "success"):
        raise ValueError("G0b asset is unavailable or differs from manifest")
    cache = Path(row["cache"]).resolve()
    if not cache.is_relative_to(root / "targets"):
        raise ValueError("G0b cache escapes fixed batch directory")
    config = json.loads((cache / "resolved_config.json").read_text())
    if (config.get("schema") != "g0b_field_interface_v1"
            or config.get("asset_id") != asset["id"]
            or config.get("source_sha256") != asset["sha256"]
            or config.get("manifest_sha256") != plan["manifest_sha256"]
            or config.get("resolution") != 128):
        raise ValueError("G0b cache metadata differs from manifest")
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--extractor", choices=tuple(MESH_NAMES), required=True)
    args = parser.parse_args()
    cache = cache_for(args.root, args.manifest, args.index)

    import numpy as np
    import trimesh

    mesh = trimesh.load(cache / MESH_NAMES[args.extractor], process=False)
    vertices = np.asarray(mesh.vertices, dtype="<f4", order="C")
    faces = np.asarray(mesh.faces, dtype="<u4", order="C")
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.isfinite(vertices).all()
            or int(faces.max()) >= len(vertices)):
        raise ValueError("Invalid G0b raw extraction mesh")
    stream = sys.stdout.buffer
    stream.write(struct.pack("<II", len(vertices), len(faces)))
    stream.write(vertices.tobytes())
    stream.write(faces.tobytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
