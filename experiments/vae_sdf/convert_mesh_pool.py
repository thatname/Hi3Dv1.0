"""Convert the formats the target pipeline cannot load into geometry-only GLB.

`vae3d.data.load_mesh` is trimesh based and cannot read fbx, blend or collada, so
the pool audit leaves about two thirds of the corpus unusable. This driver runs
headless Blender per file, exports a geometry-only GLB (materials and textures
are dropped: the target pipeline only needs triangles), and keeps a resumable
manifest so the conversion can be interrupted and restarted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path

BLENDER = "/usr/local/bin/blender"

SCRIPT = r"""
import bpy
import sys

src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
suffix = src.rsplit(".", 1)[-1].lower()
importer = {"fbx": bpy.ops.import_scene.fbx,
            "blend": bpy.ops.wm.open_mainfile,
            "dae": bpy.ops.wm.collada_import,
            "gltf": bpy.ops.import_scene.gltf,
            "glb": bpy.ops.import_scene.gltf}[suffix]
if suffix == "blend":
    importer(filepath=src)
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
else:
    importer(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("no mesh objects")
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB",
                          export_materials="NONE", export_apply=True)
print("OK", len(meshes), sum(len(o.data.polygons) for o in meshes))
"""


def convert(blender: str, source: Path, destination: Path, timeout: float) -> dict:
    row = {"source": source.name, "target": destination.name}
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [blender, "-b", "--python-expr", SCRIPT, "--", str(source), str(destination)],
            capture_output=True, text=True, timeout=timeout)
        row["seconds"] = round(time.perf_counter() - started, 2)
        row["faces"] = None
        for line in result.stdout.splitlines():
            if line.startswith("OK "):
                parts = line.split()
                row["faces"] = int(parts[2])
        if result.returncode != 0 or row["faces"] in (None, 0) or not destination.is_file():
            row["status"] = "failed"
            tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
            row["error"] = " | ".join(tail)[:400]
            destination.unlink(missing_ok=True)
        else:
            row["status"] = "converted"
            row["bytes"] = destination.stat().st_size
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        destination.unlink(missing_ok=True)
    except Exception as error:  # noqa: BLE001 - the manifest must record every mode
        row["status"] = "failed"
        row["error"] = f"{type(error).__name__}: {error}"[:400]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/TRELLIS/datasets/ObjaverseXL_github/"
                                     "raw/github/objects"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extensions", nargs="+", default=[".fbx", ".blend", ".dae"])
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole pool")
    parser.add_argument("--blender", default=BLENDER)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    todo = [path for path in sorted(args.root.iterdir())
            if path.is_file() and path.suffix.lower() in args.extensions
            and not (args.output / (path.stem + ".glb")).is_file()]
    if args.limit:
        todo = todo[:args.limit]
    print(json.dumps({"todo": len(todo), "workers": args.workers}), flush=True)

    rows = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert, args.blender, path,
                              args.output / (path.stem + ".glb"), args.timeout): path
                   for path in todo}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if done % 50 == 0 or done == len(todo):
                ok = sum(1 for r in rows if r["status"] == "converted")
                print(json.dumps({"done": done, "converted": ok,
                                  "elapsed_s": round(time.perf_counter() - started, 1)}),
                      flush=True)

    summary = {"root": str(args.root), "output": str(args.output),
               "extensions": args.extensions, "workers": args.workers,
               "requested": len(todo), "converted": sum(1 for r in rows
                                                        if r["status"] == "converted"),
               "failed": sum(1 for r in rows if r["status"] == "failed"),
               "timeout": sum(1 for r in rows if r["status"] == "timeout"),
               "bytes_out": sum(r.get("bytes", 0) for r in rows),
               "elapsed_seconds": round(time.perf_counter() - started, 1),
               "rows": sorted(rows, key=lambda r: r["source"])}
    (args.output / "conversion-manifest.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())