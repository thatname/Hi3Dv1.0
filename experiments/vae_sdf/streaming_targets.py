"""Serve A4 targets from a pool that is still being generated.

`train_g1` and `train_g1_ddp` used to load every asset once at startup, which is
impossible for tens of thousands of R256 targets. This module scans the target
root for the assets that are already finished, freezes that set for the run, and
loads one target at a time on demand. Freezing the set keeps the visit order,
the resume behaviour and the E0/E4 comparison well defined: assets that land
after the run starts are picked up by the next run, not mid-run.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

from .g1_targets import ROUTE


def scan_ready(targets_root: Path, resolution: int, route: str = ROUTE) -> dict[str, Path]:
    """Map asset id -> finished A4 directory for every completed target."""

    ready: dict[str, Path] = {}
    if not targets_root.is_dir():
        return ready
    for asset in os.scandir(targets_root):
        if not asset.is_dir():
            continue
        for cache in (Path(asset.path) / f"R{resolution}").glob(f"{route}-*"):
            stage = cache / "a4"
            if (stage / "target.npz").is_file() and (stage / "audit.json").is_file():
                ready[asset.name] = stage
                break
    return ready


class LazyTargets:
    """Index-addressed target loader with a tiny resident cache."""

    def __init__(self, targets_root: Path, manifest: dict, manifest_hash: str, resolution: int,
                 device: str, asset_device: str | None, indices: list[int] | None = None,
                 route: str = ROUTE, max_resident: int = 1, seed: int = 0):
        self.targets_root = targets_root
        self.manifest = manifest
        self.manifest_hash = manifest_hash
        self.resolution = resolution
        self.device = device
        self.asset_device = asset_device
        self.route = route
        self.max_resident = max(1, max_resident)
        self.ready = scan_ready(targets_root, resolution, route)
        available = [index for index, asset in enumerate(manifest["assets"])
                     if asset["id"] in self.ready]
        wanted = set(indices) if indices else set(available)
        # a fixed order keeps the run reproducible and lets the second arm replay
        # exactly the same sequence over the same assets
        self.indices = sorted(index for index in available if index in wanted)
        if not self.indices:
            raise ValueError(f"No finished {route} targets under {targets_root}")
        self.rows = _RowsView(self)
        self._loaded: OrderedDict[int, dict] = OrderedDict()

    def __len__(self) -> int:
        return len(self.indices)

    def index_at(self, position: int) -> int:
        return self.indices[position % len(self.indices)]

    def row(self, index: int) -> dict:
        asset = self.manifest["assets"][index]
        cache = self.ready[asset["id"]]
        return {"index": index, "id": asset["id"], "status": "success",
                "cache": str(cache), "config_root": str(cache.parent),
                "route_id": self.route, "source_sha256": asset["sha256"],
                "audit": json.loads((cache / "audit.json").read_text(encoding="utf-8"))}

    def __getitem__(self, index: int) -> dict:
        cached = self._loaded.get(index)
        if cached is not None:
            self._loaded.move_to_end(index)
            return cached
        from .train_g1 import load_asset

        asset = load_asset(index, self.manifest, self.rows, self.manifest_hash,
                           self.device, self.resolution, self.asset_device)
        self._loaded[index] = asset
        while len(self._loaded) > self.max_resident:
            self._loaded.popitem(last=False)
        return asset

    def describe(self) -> dict:
        return {"mode": "streaming", "targets_root": str(self.targets_root),
                "ready_at_launch": len(self.ready), "used": len(self.indices),
                "max_resident": self.max_resident,
                "note": "targets finished after launch join the next run, not this one"}


class _RowsView:
    """Present rows by manifest index so the stock loader can stay unchanged."""

    def __init__(self, parent: LazyTargets):
        self.parent = parent

    def __getitem__(self, index: int) -> dict:
        return self.parent.row(index)