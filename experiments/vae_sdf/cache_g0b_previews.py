"""Save the fixed G0b comparison previews locally for instant repeat browsing.

The existing viewer must be running while this one-time snapshot is built.
Afterward, view_meshes --preview-cache starts without contacting SSH.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def validate(payload: bytes) -> None:
    if len(payload) < 40 or payload[:4] != b"G0B2":
        raise ValueError("Invalid G0b preview header")
    lengths = struct.unpack_from("<III", payload, 4)
    if any(length < 8 for length in lengths) or len(payload) != 40 + sum(lengths):
        raise ValueError("Invalid G0b preview lengths")


def fetch(index: int, root: str, output: Path, timeout: int,
          reuse: bool) -> tuple[int, int, float, bool]:
    path = output / f"bundle_{index:02d}.zlib"
    if reuse and path.is_file():
        try:
            payload = zlib.decompress(path.read_bytes())
            validate(payload)
            return index, len(payload), 0.0, True
        except (ValueError, zlib.error):
            pass
    started = time.perf_counter()
    with urllib.request.urlopen(f"{root}/bundle/{index}", timeout=timeout) as response:
        payload = response.read()
    validate(payload)
    compressed = zlib.compress(payload, level=1)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(compressed)
    temporary.replace(path)
    return index, len(payload), time.perf_counter() - started, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path,
                        default=Path(".local-cache/g0b-preview-v1"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error("workers and timeout must be positive")
    root = args.viewer.rstrip("/")
    with urllib.request.urlopen(f"{root}/api/catalog", timeout=args.timeout) as response:
        catalog = json.load(response)
    rows = catalog.get("rows", [])
    if (len(rows) != 50 or not catalog.get("has_g0b")
            or any(row.get("index") != index or row.get("g0b", {}).get("status") != "success"
                   for index, row in enumerate(rows))):
        raise ValueError("Viewer catalog is not the completed balanced50 G0b set")
    args.output.mkdir(parents=True, exist_ok=True)
    previous_catalog = args.output / "catalog.json"
    reuse = previous_catalog.is_file() and json.loads(previous_catalog.read_text()) == catalog
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, index, root, args.output, args.timeout, reuse)
                   for index in range(50)]
        for future in as_completed(futures):
            index, size, seconds, cached = future.result()
            print(f"index {index:02d}: {size} bytes, {seconds:.2f}s"
                  + (" (cached)" if cached else ""), flush=True)
    temporary = args.output / "catalog.json.tmp"
    temporary.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output / "catalog.json")
    print(f"Local G0b preview snapshot ready: {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
