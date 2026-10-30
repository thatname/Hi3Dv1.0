"""Stream source, MC, and FlexiCubes meshes in one compressed G0b response."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

from .remote_g0b_mesh import MESH_NAMES, cache_for


def encode_mesh(mesh) -> bytes:
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype="<f4", order="C")
    faces = np.asarray(mesh.faces, dtype="<u4", order="C")
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.isfinite(vertices).all()
            or int(faces.max()) >= len(vertices)):
        raise ValueError("Invalid mesh for G0b comparison")
    return struct.pack("<II", len(vertices), len(faces)) + vertices.tobytes() + faces.tobytes()


def preview_mesh(mesh, max_faces: int):
    if max_faces == 0 or len(mesh.faces) <= max_faces:
        return mesh
    import numpy as np
    import open3d as o3d
    import trimesh

    source = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    reduced = source.simplify_quadric_decimation(max_faces)
    if not reduced.has_triangles():
        raise ValueError("Display preview became empty")
    return trimesh.Trimesh(
        vertices=np.asarray(reduced.vertices), faces=np.asarray(reduced.triangles), process=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--preview-faces", type=int, default=6000,
                        help="Maximum display faces per mesh; 0 streams full raw meshes")
    args = parser.parse_args()
    if args.preview_faces < 0:
        parser.error("preview-faces must be nonnegative")
    cache = cache_for(args.root, args.manifest, args.index)
    manifest = json.loads(args.manifest.read_text())["assets"]
    audit = json.loads(args.audit.read_text())
    rows = audit["assets"]
    if (len(rows) != len(manifest) or audit["resolution"] != 128
            or rows[args.index]["index"] != args.index
            or rows[args.index]["id"] != manifest[args.index]["id"]
            or rows[args.index]["path"] != manifest[args.index]["path"]
            or rows[args.index]["status"] != "loaded"):
        raise ValueError("G0 source load audit differs from fixed manifest")

    import numpy as np
    import trimesh
    from vae3d.data import load_mesh

    source, center, scale = load_mesh(
        manifest[args.index]["path"], normalization_extent=126 / 128)
    source.vertices = np.asarray(source.vertices) * scale + center
    source_low = np.asarray(source.vertices).min(axis=0).astype("<f4")
    source_high = np.asarray(source.vertices).max(axis=0).astype("<f4")
    payloads = [encode_mesh(preview_mesh(source, args.preview_faces))]
    for extractor in ("mc", "flexicubes"):
        mesh = trimesh.load(cache / MESH_NAMES[extractor], process=False)
        payloads.append(encode_mesh(preview_mesh(mesh, args.preview_faces)))
    header = struct.pack("<4sIII6f", b"G0B2", *(len(x) for x in payloads),
                         *source_low, *source_high)
    sys.stdout.buffer.write(zlib.compress(header + b"".join(payloads), level=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
