"""Read-only G0 source, environment, and asset preflight.

The manifest is JSON with an ``assets`` list. Each asset needs a stable ``id``,
an absolute ``path``, and one of the categories listed in CATEGORIES. This tool
does not generate targets or change the source meshes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


VAE3D_COMMIT = "422e79988ccda06ef0995bf7768cd05a8894a124"
CATEGORIES = {"closed_simple", "closed_complex", "open", "thin", "normal_anomaly"}
MESH_SUFFIXES = {".obj", ".glb", ".gltf", ".ply", ".stl", ".off"}
PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "trimesh": "trimesh",
    "scikit-image": "scikit-image",
    "open3d": "open3d",
    "torch": "torch",
    "DracoPy": "DracoPy",
    "warp-lang": "warp-lang",
    "slangpy": "slangpy",
    "kaolin": "kaolin",
    "nvdiffrast": "nvdiffrast",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str | None:
    if not root.is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_versions() -> dict[str, str | None]:
    result = {}
    for label, distribution in PACKAGES.items():
        try:
            result[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[label] = None
    return result


def inspect_manifest(path: Path) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"manifest: {exc}"]
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        return [], ["manifest: expected schema_version 1"]
    assets = document.get("assets")
    if not isinstance(assets, list) or not assets:
        return [], ["manifest: assets must be a nonempty list"]

    inspected: list[dict] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for index, asset in enumerate(assets):
        label = f"assets[{index}]"
        if not isinstance(asset, dict):
            errors.append(f"{label}: expected object")
            continue
        asset_id = asset.get("id")
        category = asset.get("category")
        raw_path = asset.get("path")
        if not isinstance(asset_id, str) or not asset_id.strip():
            errors.append(f"{label}: id must be a nonempty string")
            continue
        if asset_id in seen_ids:
            errors.append(f"{label}: duplicate id {asset_id}")
        seen_ids.add(asset_id)
        if category not in CATEGORIES:
            errors.append(f"{label}: category must be one of {sorted(CATEGORIES)}; found {category}")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            errors.append(f"{label}: path must be absolute")
            continue
        mesh_path = Path(raw_path)
        if mesh_path.suffix.lower() not in MESH_SUFFIXES:
            errors.append(f"{label}: unsupported mesh suffix {mesh_path.suffix}")
        if not mesh_path.is_file():
            errors.append(f"{label}: missing file {mesh_path}")
            continue
        digest = sha256(mesh_path)
        if digest in seen_hashes:
            errors.append(f"{label}: same content as {seen_hashes[digest]}")
        seen_hashes[digest] = asset_id
        inspected.append(
            {"id": asset_id, "category": category, "path": str(mesh_path),
             "sha256": digest, "bytes": mesh_path.stat().st_size}
        )
    return inspected, errors


def run(manifest: Path, vae3d_root: Path) -> dict:
    assets, errors = inspect_manifest(manifest)
    source_head = git_head(vae3d_root)
    if source_head != VAE3D_COMMIT:
        errors.append(f"vae3d source must be {VAE3D_COMMIT}; found {source_head}")
    versions = package_versions()
    for package in ("numpy", "scipy", "trimesh", "scikit-image", "open3d", "torch", "kaolin", "nvdiffrast"):
        if versions[package] is None:
            errors.append(f"missing package: {package}")
    if sys.version_info[:2] != (3, 10):
        errors.append(f"expected Python 3.10; found {platform.python_version()}")
    return {
        "schema_version": 1,
        "ready": not errors,
        "python": platform.python_version(),
        "vae3d_root": str(vae3d_root),
        "vae3d_head": source_head,
        "packages": versions,
        "assets": assets,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vae3d-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    report = run(args.manifest, args.vae3d_root)
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
