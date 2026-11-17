"""Frame-exact video reading for tracklab: frame idx is the idx-th frame ffmpeg decodes.

tracklab 1.3.24's VideoReader seeks OpenCV to idx - 1 (frames 0 and 1 come out the same, and
every later frame one early) and reopens the file for every frame. gbf's frame index is the
decode order of the CFR file (ADR 0003), and gbf decodes with ffmpeg to rgb24; this reader
uses the same ffmpeg arguments and keeps one process reading forward (restarting when a
module asks for an earlier frame), so frame idx here is gbf's frame idx, pixel for pixel.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections import OrderedDict

import numpy as np


def _size(path: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    width, height = (int(v) for v in out.split(",")[:2])
    return width, height


class ExactVideoReader:
    """Drop-in for tracklab.utils.cv2.VideoReader: returns BGR frames like OpenCV does."""

    def __init__(self, cache: int = 64) -> None:
        self.filename: str | None = None
        self._proc: subprocess.Popen | None = None
        self._next = 0
        self._shape: tuple[int, int] | None = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._limit = cache

    def set_filename(self, filename: str) -> None:
        if filename != self.filename:
            self._stop()
            self.filename = filename
            width, height = _size(filename)
            self._shape = (height, width)
            self._cache.clear()

    def _stop(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
        self._proc = None
        self._next = 0

    def _start(self) -> None:
        self._stop()
        args = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error",
                "-i", self.filename, "-map", "0:v:0", "-an", "-sn", "-dn",
                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]  # fmt: skip
        self._proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)

    def rgb(self, idx: int) -> np.ndarray:
        """Frame ``idx`` as an RGB array (height, width, 3)."""
        assert self.filename is not None, "You should first set the filename"
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        if self._proc is None or idx < self._next:
            self._start()
        height, width = self._shape
        size = height * width * 3
        frame = None
        while self._next <= idx:
            buf = self._proc.stdout.read(size)
            if len(buf) < size:
                raise AssertionError(f"Read past the end of the video file (frame {idx})")
            frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            self._cache[self._next] = frame
            if len(self._cache) > self._limit:
                self._cache.popitem(last=False)
            self._next += 1
        return frame

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.ascontiguousarray(self.rgb(int(idx))[..., ::-1])  # BGR, as OpenCV returns


def frame_hashes(path: str, indices: list[int]) -> dict[int, str]:
    """sha256 of the rgb24 bytes of the given frames."""
    reader = ExactVideoReader()
    reader.set_filename(path)
    return {i: hashlib.sha256(reader.rgb(i).tobytes()).hexdigest() for i in sorted(indices)}
