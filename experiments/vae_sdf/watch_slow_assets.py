"""Defer A1/A2 assets that have been running far longer than the median.

One 12 M-face mesh can keep a worker busy for tens of minutes and, because
open3d's raycast parallelism ignores every thread knob, drag most of the machine
with it. Killing such a worker and marking the asset `.defer` keeps the bulk
pass fair; the asset is retried in a final pass when nothing else competes, so
its wide parallelism is affordable then.

Runs beside the orchestrator and is read-only apart from the claim markers.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600.0,
                        help="Defer an asset after this long (median is ~120 s)")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--apply", action="store_true",
                        help="Without it only report what would be deferred")
    args = parser.parse_args()

    seen: set[int] = set()
    rows: list[dict] = []
    if args.ledger.is_file():
        try:
            rows = json.loads(args.ledger.read_text(encoding="utf-8"))
            seen = {row["index"] for row in rows}
        except (OSError, ValueError, KeyError):
            rows, seen = [], set()

    while True:
        now = time.time()
        for claim in args.claims.glob("*.claim"):
            try:
                payload = json.loads(claim.read_text(encoding="utf-8"))
                started = float(payload["started"])
                pid = int(payload["pid"])
                index = int(payload["index"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            age = now - started
            if age < args.max_seconds or index in seen:
                continue
            event = {"index": index, "pid": pid, "seconds": round(age, 1)}
            print(json.dumps({"event": "defer", **event, "apply": args.apply}), flush=True)
            if not args.apply:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deferred = claim.with_suffix(".defer")
            deferred.write_text(json.dumps(event), encoding="utf-8")
            claim.unlink(missing_ok=True)
            rows.append(event)
            seen.add(index)
            args.ledger.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())