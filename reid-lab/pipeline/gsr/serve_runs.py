"""Serve C:/Research/gbf-data/gsr-runs to this computer only, for the local clip viewer (viewer.html).

Run (any Python 3.10+): python -I serve_runs.py [PORT]   (default 8766; binds 127.0.0.1)
Plain http.server cannot answer HTTP Range requests, so a browser cannot seek in the videos;
this handler answers single byte ranges. Directory listings are off. SN-GSR-2025 is for research
only: keep this bound to 127.0.0.1.
"""

from __future__ import annotations

import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = r"C:\Research\gbf-data\gsr-runs"
RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class Handler(SimpleHTTPRequestHandler):
    def list_directory(self, path):  # no listings
        self.send_error(403, "listing disabled")
        return None

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        self._left = None
        path = self.translate_path(self.path)
        m = RANGE.match((self.headers.get("Range") or "").strip())
        if not m or os.path.isdir(path) or not (m[1] or m[2]):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "not found")
            return None
        size = os.fstat(f.fileno()).st_size
        if m[1]:
            start, end = int(m[1]), min(int(m[2]) if m[2] else size - 1, size - 1)
        else:  # suffix range: the last N bytes
            start, end = max(0, size - int(m[2])), size - 1
        if start > end or start >= size:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        f.seek(start)
        self._left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        if self._left is None:
            return super().copyfile(source, outputfile)
        left = self._left
        while left > 0:
            buf = source.read(min(1 << 16, left))
            if not buf:
                break
            outputfile.write(buf)
            left -= len(buf)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True  # the browser cancelled a video request: normal

    def log_message(self, format, *args):
        pass


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=ROOT))
    print(f"serving {ROOT} on http://127.0.0.1:{port}/viewer.html", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
