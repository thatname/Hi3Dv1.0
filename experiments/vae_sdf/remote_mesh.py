"""Export one audited G0 source mesh as a compact binary stream over SSH.

Run on the remote host. stdout contains only the binary mesh; diagnostics use
stderr. The caller can select only an index from the fixed manifest.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["assets"]
    audit = json.loads(args.audit.read_text(encoding="utf-8"))["assets"]
    if not 0 <= args.index < len(manifest) or len(manifest) != len(audit):
        parser.error("index outside the audited manifest")
    source, result = manifest[args.index], audit[args.index]
    if (result["index"] != args.index or result["id"] != source["id"]
            or result["path"] != source["path"] or result["status"] != "loaded"):
        parser.error("asset is missing, failed, or audit does not match manifest")

    import numpy as np
    from vae3d.data import load_mesh

    resolution = int(json.loads(args.audit.read_text(encoding="utf-8"))["resolution"])
    mesh, center, scale = load_mesh(
        source["path"], normalization_extent=(resolution - 2) / resolution
    )
    # Restore the source coordinate frame after vae3d's validated loader. The
    # browser's camera fit is display-only and does not change these bytes.
    vertices = np.asarray(mesh.vertices * scale + center, dtype="<f4", order="C")
    faces = np.asarray(mesh.faces, dtype="<u4", order="C")
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or
            faces.ndim != 2 or faces.shape[1] != 3 or
            not np.isfinite(vertices).all() or
            (faces.size and int(faces.max()) >= len(vertices))):
        raise ValueError("Invalid triangle mesh from audited source")
    stream = sys.stdout.buffer
    stream.write(struct.pack("<II", len(vertices), len(faces)))
    stream.write(vertices.tobytes())
    stream.write(faces.tobytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
