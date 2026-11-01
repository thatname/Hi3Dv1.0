"""Validate fixed G1 asset-level train/validation split against the manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _validate(path: Path, manifest: dict, manifest_sha256: str, role: str):
    payload = path.read_bytes()
    split = json.loads(payload)
    if (split.get("schema") != "g1_asset_split_v1"
            or split.get("manifest_sha256") != manifest_sha256
            or role not in {"train", "val"}):
        raise ValueError("G1 split schema, manifest, or role mismatch")
    train, val = split["train_indices"], split["val_indices"]
    if (not train or not val or len(train) != len(set(train))
            or len(val) != len(set(val)) or set(train) & set(val)):
        raise ValueError("G1 split contains empty, duplicate, or overlapping assets")
    assets = manifest["assets"]
    for group, selected in (("train", train), ("val", val)):
        if (any(not isinstance(i, int) or i < 0 or i >= len(assets) for i in selected)
                or split[group + "_ids"] != [assets[i]["id"] for i in selected]):
            raise ValueError(f"G1 {group} split IDs differ from manifest")
    return split, hashlib.sha256(payload).hexdigest()


def check_split(path: Path, manifest: dict, manifest_sha256: str,
                role: str, indices: list[int]) -> str:
    split, digest = _validate(path, manifest, manifest_sha256, role)
    if indices != split[role + "_indices"]:
        raise ValueError(f"G1 requested indices differ from fixed {role} split")
    return digest


def check_split_subset(path: Path, manifest: dict, manifest_sha256: str,
                       role: str, indices: list[int]) -> str:
    """Validate a split for a run that only uses the assets finished so far.

    A streaming run cannot cover the whole split, because targets land while the
    pool is still being generated. Every selected asset must still come from the
    declared role, so held-out assets never leak into training.
    """

    split, digest = _validate(path, manifest, manifest_sha256, role)
    allowed = set(split[role + "_indices"])
    outside = sorted(set(indices) - allowed)
    if outside:
        raise ValueError(f"G1 {role} run uses assets outside the split: {outside[:5]}")
    if not indices:
        raise ValueError(f"G1 {role} run selected no assets")
    return digest