"""Run the sn-gamestate baseline (tracklab) on one video, frame-exactly.

Replaces tracklab's VideoReader with tools/exact_reader.ExactVideoReader before anything
loads a frame, checks that tracklab's image loader now returns gbf's frames (the sha256 of
the decoded rgb24 bytes of a few frames, compared with the hashes gbf's own decoder printed,
when --gbf-hashes is given), then hands the remaining arguments to tracklab's Hydra main.

Usage (from the sn-gamestate folder, its .venv):
  python ../tools/run_tracklab.py --gbf-hashes hashes.json -- -cn soccernet dataset=video ...
The video path inside the Hydra overrides must not contain a drive letter: tracklab splits
"vid://<path>:<frame>" at the colon.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_reader import ExactVideoReader  # noqa: E402

import tracklab.utils.cv2 as tracklab_cv2  # noqa: E402


def allow_numpy_in_checkpoints() -> None:
    """PyTorch >= 2.6 unpickles only tensors by default (weights_only=True); the PRTReid
    checkpoint also stores numpy scalars. Allowlist exactly those numpy types instead of
    turning weights_only off (which would let a checkpoint run arbitrary code)."""
    import numpy
    import torch
    import yacs.config

    # torch.serialization.get_unsafe_globals_in_checkpoint lists exactly these three for
    # prtreid-soccernet-baseline.pth.tar (its stored training config and metrics)
    torch.serialization.add_safe_globals(
        [
            yacs.config.CfgNode,
            numpy.core.multiarray.scalar,
            numpy.dtype,
            numpy.dtypes.Float64DType,
            numpy.dtypes.Float32DType,
            numpy.dtypes.Int64DType,
        ]
    )


def patch() -> ExactVideoReader:
    allow_numpy_in_checkpoints()
    reader = ExactVideoReader()
    tracklab_cv2.video_reader = reader
    tracklab_cv2.cv2_load_image.cache_clear()
    return reader


def check(video: str, expected: dict[str, str]) -> None:
    """tracklab's loader (patched) gives the same pixels as gbf's decoder."""
    problems = []
    for key, want in expected.items():
        rgb = tracklab_cv2.cv2_load_image(f"vid://{video}:{int(key)}")
        got = hashlib.sha256(rgb.tobytes()).hexdigest()
        if got != want:
            problems.append(f"frame {key}: {got[:12]} != gbf {want[:12]}")
    tracklab_cv2.cv2_load_image.cache_clear()
    if problems:
        raise SystemExit("frame check failed: " + "; ".join(problems))
    print(f"frame check: {len(expected)} frames identical to gbf's decoder")


def main() -> None:
    args = sys.argv[1:]
    hashes = None
    if args[:1] == ["--gbf-hashes"]:
        hashes = json.loads(Path(args[1]).read_text(encoding="utf-8"))
        args = args[2:]
    if args[:1] == ["--"]:
        args = args[1:]
    patch()
    if hashes is not None:
        check(hashes["video"], hashes["frames"])
    sys.argv = ["tracklab", *args]
    from tracklab.main import main as tracklab_main

    tracklab_main()


if __name__ == "__main__":
    main()
