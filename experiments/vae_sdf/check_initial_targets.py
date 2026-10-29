"""Check saved A1/A2 cache shape, units, shared support and raw mesh files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(a1_path: Path, a2_path: Path) -> dict:
    import numpy as np
    import trimesh

    caches = {}
    for stage, path in (("A1", a1_path), ("A2", a2_path)):
        config = json.loads((path / "resolved_config.json").read_text())
        audit = json.loads((path / "audit.json").read_text())
        field = np.load(path / "field.npy", allow_pickle=False)
        with np.load(path / "target.npz", allow_pickle=False) as data:
            target = {key: data[key] for key in data.files}
        if config["stage"] != stage or audit["stage"] != stage:
            raise ValueError(f"Stage mismatch in {path}")
        r = config["resolution"]
        coords, features = target["coords"], target["features"]
        cells, indices = target["cells"], target["cell_indices"]
        if field.shape != (r, r, r) or features.shape != (len(coords), 4):
            raise ValueError(f"Invalid field or feature shape in {path}")
        if (str(target["field_layout"]) != "center_points_sdf_delta_v1"
                or str(target["field_units"]) != "fine_voxel"
                or str(target["delta_units"]) != "fine_voxel"
                or float(target["sampling_offset"]) != 0.5):
            raise ValueError(f"Invalid field layout or units in {path}")
        if (np.any(features[:, 1:] != 0) or
                not np.allclose(features[:, 0],
                                np.clip(field[tuple(coords.T)] * r, -4, 4), atol=1e-6)):
            raise ValueError(f"Target values differ from dense field in {path}")
        corners = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                              (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))
        if not np.array_equal(coords[indices], cells[:, None, :] + corners[None, :, :]):
            raise ValueError(f"Incomplete shared cells in {path}")
        mesh = trimesh.load(path / "raw_mesh.ply", process=False)
        if (len(mesh.vertices) != audit["raw_mesh_vertices"]
                or len(mesh.faces) != audit["raw_mesh_faces"]):
            raise ValueError(f"Raw mesh size differs from audit in {path}")
        caches[stage] = {"config": config, "audit": audit, "field": field,
                         "coords": coords, "cells": cells, "indices": indices}
    a1, a2 = caches["A1"], caches["A2"]
    for key in ("source_sha256", "manifest_sha256", "resolution", "sampling_offset",
                "raycast_backend", "distance_backend", "normalization_center",
                "normalization_scale", "support_stages"):
        if a1["config"][key] != a2["config"][key]:
            raise ValueError(f"A1/A2 config differs at {key}")
    for key in ("coords", "cells", "indices"):
        if not np.array_equal(a1[key], a2[key]):
            raise ValueError(f"A1/A2 support differs at {key}")
    from vae3d.center_points import complete_fitting_support
    from .prepare_initial_targets import candidate_cells

    mask1, _ = candidate_cells(a1["field"])
    mask2, _ = candidate_cells(a2["field"])
    expected_cells = complete_fitting_support(
        np.argwhere(mask1 | mask2).astype(np.int32), a1["config"]["resolution"]
    )
    if not np.array_equal(expected_cells, a1["cells"]):
        raise ValueError("Common support differs from pinned vae3d completion rule")
    added = int(np.count_nonzero((a1["field"] >= 0) & (a2["field"] < 0)))
    if added != a2["audit"]["query_audit"]["added_shell_samples"]:
        raise ValueError("A2 shell addition audit differs from saved fields")
    return {"resolution": a1["config"]["resolution"],
            "support_cells": len(a1["cells"]), "support_points": len(a1["coords"]),
            "field_changed_samples": int(np.count_nonzero(a1["field"] != a2["field"])),
            "added_negative_samples": added,
            "reference_max_abs_error": a2["audit"]["reference_comparison"]["max_abs_error"]
            if a2["audit"]["reference_comparison"] else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.a1, args.a2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
