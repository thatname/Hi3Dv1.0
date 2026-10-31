"""Serve field slices from one remote diagnostic export directory.

The asset list is no longer fixed: pass --root and --indices to point the viewer
at any export produced by the field-diagnostic exporter. Defaults keep the
original index 0/7 step-1000 behaviour.
"""

from __future__ import annotations

import argparse
import subprocess
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


HTML = Path(__file__).with_name("g1_field_diagnostic.html")
REMOTE_ROOT = "/data/hi3d/g1/field-diagnostic-0-7-step1000-v1"
DEFAULT_INDICES = (0, 7)


def asset_options(indices, val_indices) -> str:
    roles = {0: "验证集", 1: "训练集"}
    return "\n".join(
        f'<option value="{index}">{index} · {roles[0 if index in set(val_indices) else 1]}</option>'
        for index in indices)


def handler_for(host: str, timeout: int, root: str = REMOTE_ROOT,
                indices=DEFAULT_INDICES, val_indices=()):
    allowed = {int(index) for index in indices}
    options = asset_options(indices, val_indices)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    body = HTML.read_bytes().replace(b"{{ASSET_OPTIONS}}", options.encode())
                    mime, encoding = "text/html; charset=utf-8", None
                elif path.startswith("/api/slice/") and path.rsplit("/", 1)[-1].isdigit() \
                        and int(path.rsplit("/", 1)[-1]) in allowed:
                    index = int(path.rsplit("/", 1)[-1])
                    result = subprocess.run(
                        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                         host, f"cat {root}/index{index:02d}_slice.json.gz"],
                        capture_output=True, timeout=timeout, check=False)
                    if result.returncode:
                        raise RuntimeError(result.stderr.decode("utf-8", "replace")[:300])
                    body, mime, encoding = result.stdout, "application/json", "gzip"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                if encoding:
                    self.send_header("Content-Encoding", encoding)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                self.send_error(502, str(error)[:350])

        def log_message(self, fmt, *args):
            if len(args) >= 2 and str(args[1]) != "200":
                super().log_message(fmt, *args)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="le.chang@124.156.97.173")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--root", default=REMOTE_ROOT,
                        help="Remote directory holding indexNN_slice.json.gz exports")
    parser.add_argument("--indices", type=int, nargs="+", default=list(DEFAULT_INDICES),
                        help="Assets offered in the page's selector")
    parser.add_argument("--val-indices", type=int, nargs="*", default=[0],
                        help="Indices shown as held-out validation in the selector")
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        handler_for(args.host, args.timeout, args.root, args.indices, args.val_indices))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"G1 field diagnostic: {url}. Ctrl+C to stop.", flush=True)
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
