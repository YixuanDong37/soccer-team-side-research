"""COCO "sports ball" candidates per frame from Ultralytics YOLO11m, for gbf's ball stage.

Runs in the sn-gamestate environment (ultralytics is AGPL-3.0; it never enters gbf's core).
Frames are decoded by ffmpeg in decode order, which is gbf's canonical frame index for a CFR
file (ADR 0003), and passed to YOLO as BGR arrays. Writes, into --out:

- ball_raw.jsonl: one line per frame, {"frame": n, "cands": [[x1, y1, x2, y2, conf], ...]}
  in the video's pixels (docs/design/m1-highlights.md, "Inputs");
- ball-receipt.json: the licence ("license") and the restriction tags ("restrictions") gbf
  reads, plus the sha256 of the output, the weights and the video, and the settings.

Usage: python ball_detect.py --video V --weights yolo11m.pt --out DIR [--conf 0.05]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

SPORTS_BALL = 32  # COCO class index


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    width, height = (int(v) for v in out.split(",")[:2])
    return width, height


def frames(video: Path, width: int, height: int):
    """(index, RGB frame) in decode order."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0", "-fps_mode",
           "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]  # fmt: skip
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = width * height * 3
    index = 0
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield index, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        index += 1
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed with exit code {proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--restrictions", default="agpl_network_source,no_share")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    width, height = probe(args.video)
    model = YOLO(str(args.weights))
    started = time.perf_counter()
    lines: list[str] = []
    batch: list[tuple[int, np.ndarray]] = []

    def flush() -> None:
        results = model.predict(
            [frame[..., ::-1] for _, frame in batch],  # Ultralytics reads arrays as BGR
            classes=[SPORTS_BALL], conf=args.conf, imgsz=args.imgsz, verbose=False,
        )  # fmt: skip
        for (index, _), result in zip(batch, results):
            boxes = result.boxes
            cands = [
                [round(float(v), 2) for v in xyxy] + [round(float(c), 4)]
                for xyxy, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())
            ]
            lines.append(json.dumps({"frame": index, "cands": cands}, separators=(",", ":")))
        batch.clear()

    for index, frame in frames(args.video, width, height):
        batch.append((index, frame))
        if len(batch) == args.batch:
            flush()
    if batch:
        flush()

    raw = args.out / "ball_raw.jsonl"
    raw.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    receipt = {
        "tool": "ultralytics YOLO11m, COCO-pretrained weights, class 32 'sports ball'",
        "license": "AGPL-3.0-only",
        "restrictions": [t for t in args.restrictions.split(",") if t],
        "weights": {"name": args.weights.name, "sha256": sha256(args.weights)},
        "video": {"sha256": sha256(args.video), "frames": len(lines), "size": [width, height]},
        "settings": {"conf": args.conf, "imgsz": args.imgsz, "classes": [SPORTS_BALL]},
        "outputs": [{"name": raw.name, "sha256": sha256(raw), "bytes": raw.stat().st_size}],
        "wall_s": round(time.perf_counter() - started, 1),
    }
    (args.out / "ball-receipt.json").write_bytes(
        (json.dumps(receipt, indent=2) + "\n").encode("utf-8")  # LF only (Python 3.9)
    )
    with_cands = sum('"cands":[]' not in line for line in lines)
    print(f"{len(lines)} frames, {with_cands} with candidates, {receipt['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
