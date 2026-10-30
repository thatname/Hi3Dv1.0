"""Local, read-only G1 training monitor; independent of the G0 mesh viewer.

Example:
  python3 -m experiments.vae_sdf.view_g1 --port 8767 --no-browser

The server binds 127.0.0.1 and polls one fixed remote run via SSH. It never
loads model weights, starts training, or writes to the remote filesystem.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


HTML = Path(__file__).with_name("g1_train.html")


class RemoteRun:
    def __init__(self, host: str, root: str, remote_python: str, repo: str,
                 timeout: int, cache_seconds: float = 2.0):
        if not host or host.startswith("-") or timeout < 1:
            raise ValueError("Invalid SSH host or timeout")
        self.host, self.root, self.python, self.repo = host, root, remote_python, repo
        self.timeout, self.cache_seconds = timeout, cache_seconds
        self.lock = threading.Lock()
        self.cached_at = 0.0
        self.cached = None

    def state(self) -> dict:
        with self.lock:
            now = time.monotonic()
            if self.cached is not None and now - self.cached_at < self.cache_seconds:
                return self.cached
            command = ["env", f"PYTHONPATH={self.repo}", self.python,
                       "-m", "experiments.vae_sdf.remote_g1_status", "--run", self.root]
            result = subprocess.run(
                ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 self.host, shlex.join(command)], capture_output=True,
                timeout=self.timeout, check=False,
            )
            if result.returncode:
                detail = result.stderr.decode("utf-8", "replace").strip()
                if "FileNotFoundError" in detail and "resolved_config.json" in detail:
                    self.cached = {"schema": "g1_a4_overfit_v2", "run": self.root,
                                   "status": "waiting", "indices": [], "assets": [],
                                   "planned_steps": None, "trace": [],
                                   "checkpoint": {"available": False}}
                    self.cached_at = time.monotonic()
                    return self.cached
                raise RuntimeError(f"Remote G1 status failed ({result.returncode}): {detail[:400]}")
            state = json.loads(result.stdout)
            self.cached, self.cached_at = state, time.monotonic()
            return state


def handler_for(run: RemoteRun):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    body, mime = HTML.read_bytes(), "text/html; charset=utf-8"
                elif path == "/api/state":
                    body = json.dumps(run.state(), ensure_ascii=False, allow_nan=False).encode()
                    mime = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                self.send_error(502, str(error)[:400])

        def log_message(self, format, *args):
            if len(args) >= 2 and str(args[1]) != "200":
                super().log_message(format, *args)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="le.chang@124.156.97.173")
    parser.add_argument("--run", default="/data/hi3d/g0-prep/g1-a4-fifty-v2")
    parser.add_argument("--remote-python", default="/data/hi3d/hi3d-validation/.venv/bin/python")
    parser.add_argument("--remote-repo", default="/data/hi3d/hi3d-vae-field")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be in 0..65535")
    run = RemoteRun(args.host, args.run, args.remote_python, args.remote_repo, args.timeout)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(run))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"G1 training monitor: {url} — remote run {args.run}. Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
