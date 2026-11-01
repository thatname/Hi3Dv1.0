"""Local, read-only G3 pipeline monitor; separate from the training page on 8769/8770.

Example:
  python3 -m experiments.vae_sdf.view_g3_pipeline --port 8771 --no-browser \
      --experiment /data/hi3d/g3/exp-35k-r256-e4-v1

The server binds 127.0.0.1 and polls one fixed remote experiment over SSH. It
never writes to the remote filesystem and never loads model weights.
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


HTML = Path(__file__).with_name("g3_pipeline.html")


class RemotePipeline:
    def __init__(self, host: str, experiment: str, remote_python: str, repo: str,
                 resolution: int, timeout: int, cache_seconds: float = 5.0):
        if not host or host.startswith("-") or timeout < 1:
            raise ValueError("Invalid SSH host or timeout")
        self.host, self.experiment = host, experiment
        self.python, self.repo, self.resolution = remote_python, repo, resolution
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
                       "-m", "experiments.vae_sdf.remote_g3_status",
                       "--experiment", self.experiment,
                       "--resolution", str(self.resolution)]
            result = subprocess.run(
                ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 self.host, shlex.join(command)], capture_output=True,
                timeout=self.timeout, check=False,
            )
            if result.returncode:
                detail = result.stderr.decode("utf-8", "replace").strip()
                raise RuntimeError(f"Remote G3 status failed ({result.returncode}): {detail[:400]}")
            state = json.loads(result.stdout)
            self.cached, self.cached_at = state, time.monotonic()
            return state


def handler_for(run: RemotePipeline):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    body, mime = HTML.read_bytes(), "text/html; charset=utf-8"
                elif path == "/api/state":
                    body = json.dumps(run.state(), ensure_ascii=False,
                                      allow_nan=False).encode()
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
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--host", default="tc-hk-h20-02-ide")
    parser.add_argument("--experiment", default="/data/hi3d/g3/exp-35k-r256-e4-v1")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--remote-python", default="/data/hi3d/hi3d-validation/.venv/bin/python")
    parser.add_argument("--remote-repo", default="/data/hi3d/hi3d-vae-field")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    run = RemotePipeline(args.host, args.experiment, args.remote_python, args.remote_repo,
                         args.resolution, args.timeout)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(run))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"G3 pipeline monitor: {url} — remote {args.experiment}. Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(.5), webbrowser.open(url)),
                         daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())