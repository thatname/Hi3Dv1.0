"""Read a G1 run for the local viewer without loading CUDA or checkpoint weights."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def read_json(path: Path, *, optional: bool = False):
    for attempt in range(4):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if optional:
                return None
            raise
        except json.JSONDecodeError:
            if attempt == 3:
                raise
            time.sleep(.05)


def snapshot(run: Path) -> dict:
    run = run.resolve()
    if not run.is_relative_to(Path("/data/hi3d").resolve()):
        raise ValueError("G1 run must be inside /data/hi3d")
    config = read_json(run / "resolved_config.json")
    if config.get("schema") not in {"g1_a4_overfit_v1", "g1_a4_overfit_v2",
                                    "g1_four_route_v1"}:
        raise ValueError("Unsupported G1 training report schema")
    trace = read_json(run / "trace.json", optional=True)
    if trace is None:
        trace = config.get("trace", [])
    if not isinstance(trace, list) or len(trace) > 100000:
        raise ValueError("Invalid G1 trace")
    compact = []
    for row in trace:
        if not isinstance(row, dict) or not isinstance(row.get("step"), int):
            raise ValueError("Invalid G1 trace row")
        compact.append({key: row.get(key) for key in (
            "step", "index", "loss", "field", "delta", "support", "validity",
            "kl_diagnostic", "latent_sites", "fine_cells", "elapsed_seconds", "gradients")})
    ckpt = run / "checkpoint.pt"
    stat = ckpt.stat() if ckpt.is_file() else None
    return {
        "schema": config["schema"], "run": str(run),
        "status": config.get("status", "unknown"),
        "error": config.get("error"), "indices": config.get("indices", []),
        "planned_steps": config.get("steps"), "beta": config.get("beta"),
        "teacher_forced_support": config.get("teacher_forced_support"),
        "lr": config.get("lr"), "seed": config.get("seed"),
        "gpu": config.get("gpu"), "torch_version": config.get("torch_version"),
        "manifest_sha256": config.get("manifest_sha256"),
        "target_route": config.get("target_route"),
        "pretrained_loading": config.get("pretrained_loading"),
        "encoder_initialization": config.get("encoder_initialization"),
        "peak_cuda_allocated_bytes": config.get("peak_cuda_allocated_bytes"),
        "elapsed_seconds": config.get("elapsed_seconds"),
        "surface_weight": config.get("surface_weight"),
        "surface_sigma": config.get("surface_sigma"),
        "truncation": config.get("truncation"),
        "validity_weight": config.get("validity_weight"),
        "checkpoint": {"available": stat is not None,
                       "bytes": stat.st_size if stat else None,
                       "mtime_ns": stat.st_mtime_ns if stat else None},
        "assets": [{key: asset.get(key) for key in (
            "index", "id", "a0_status", "a0_concerns", "inverted_crossing_cells")}
                   for asset in config.get("assets", [])],
        "trace": compact,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    json.dump(snapshot(args.run), sys.stdout, ensure_ascii=False, allow_nan=False,
              separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
