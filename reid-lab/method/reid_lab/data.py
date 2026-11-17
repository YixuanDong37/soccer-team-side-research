"""Label-free inputs of the re-identification lab: the tracking outputs of the 13 SN-GSR dev clips.

Methods import this module only. Nothing here reads an annotation: the labels live in score.py,
which methods must not import (except for clearly marked oracle runs and threshold fitting
inside a leave-one-clip-out loop).

    import sys; sys.path.insert(0, r"C:\\Research\\plugins\\tools")
    from reid_lab import data

Per clip (seq like "SNGS-162"):
- tracks(seq): gbf refine/tracks.parquet, one row per box (720p). track_id = gbf piece id
  (gbf split the plugin's tracks where a person left the picture and came back: piece id =
  100000 * k + plugin id, k = 0 for the first stretch; -1 = untracked row); source_track_id =
  the plugin's track id; role, team (left/right, plugin + gbf team check), jersey and
  jersey_conf (EasyOCR reads, rare), pitch_x/pitch_y in metres (pitch_usable, in_pitch), conf.
- embeddings(seq): unit-length PRTReid embedding (256) per tracks row, NaN where none.
- clean_rows(seq): rows whose box is away from the image edge, overlaps no other box and is at
  least 40 px tall (appearance_match.py's rule): the rows to describe a person by.
- calib(seq): gbf refine/calib.parquet, the camera per frame (K, R, C, pitch_usable ...).
- refine(seq): gbf refine/refine.json (teams, stitch audit, reentry_pieces ...).
- appearance(seq): appearance_match.py's output (candidate_pairs [earlier, later, cosine], ...).
- shot_cuts(seq): frames where a new shot starts (none in the 13 dev clips).
- pieces(seq): one row per piece: plugin id, first/last frame, rows, majority team and role,
  whether its first/last box touches the image edge, its first/last pitch position.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(r"C:\Research\gbf-data\gsr-runs")
DEV_CLIPS = ("SNGS-107", "SNGS-163", "SNGS-106", "SNGS-097", "SNGS-067", "SNGS-100", "SNGS-066",
             "SNGS-161", "SNGS-169", "SNGS-162", "SNGS-151", "SNGS-102", "SNGS-112")  # fmt: skip
FPS, WIDTH, HEIGHT = 25, 1280, 720
EDGE_PX = 10.0          # 720p; the check's 15 px at 1080p
MIN_BOX_H = 40.0        # appearance_match.py's MIN_BOX_H


def run_dir(seq: str) -> Path:
    if seq not in DEV_CLIPS:
        raise ValueError(f"{seq} is not a dev clip: the lab works on DEV_CLIPS only")
    return RUNS / seq


@lru_cache(maxsize=None)
def tracks(seq: str) -> pd.DataFrame:
    return pd.read_parquet(run_dir(seq) / "gbf/refine/tracks.parquet").reset_index(drop=True)


@lru_cache(maxsize=None)
def embeddings(seq: str) -> np.ndarray:
    t = tracks(seq)
    z = np.load(run_dir(seq) / "reid-embeddings.npz")
    milli = lambda v: int(np.rint(float(v) * 1000))  # noqa: E731
    lookup = {(int(f), int(k), *(milli(v) for v in b)): i for i, (f, k, b) in enumerate(zip(z["frame"], z["track"], z["box"]))}
    unit = z["emb"].astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    out = np.full((len(t), unit.shape[1]), np.nan)
    for r in t.itertuples():
        i = lookup.get((int(r.frame), int(r.source_track_id), milli(r.x1), milli(r.y1), milli(r.x2), milli(r.y2)))
        if i is not None:
            out[r.Index] = unit[i]
    return out


@lru_cache(maxsize=None)
def clean_rows(seq: str) -> np.ndarray:
    t = tracks(seq)
    x1, y1, x2, y2 = (t[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
    edge = (x1 <= 1) | (y1 <= 1) | (x2 >= WIDTH - 2) | (y2 >= HEIGHT - 2)
    overlap = np.zeros(len(t), bool)
    for _, idx in t.groupby("frame").indices.items():
        if len(idx) < 2:
            continue
        ix = np.minimum(x2[idx][:, None], x2[idx][None]) - np.maximum(x1[idx][:, None], x1[idx][None])
        iy = np.minimum(y2[idx][:, None], y2[idx][None]) - np.maximum(y1[idx][:, None], y1[idx][None])
        hit = (ix > 0) & (iy > 0)
        np.fill_diagonal(hit, False)
        overlap[idx] = hit.any(axis=1)
    has = ~np.isnan(embeddings(seq)[:, 0])
    return (~edge) & (~overlap) & ((y2 - y1) >= MIN_BOX_H) & has


@lru_cache(maxsize=None)
def calib(seq: str) -> pd.DataFrame:
    return pd.read_parquet(run_dir(seq) / "gbf/refine/calib.parquet")


@lru_cache(maxsize=None)
def refine(seq: str) -> dict:
    return json.loads((run_dir(seq) / "gbf/refine/refine.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def appearance(seq: str) -> dict:
    return json.loads((run_dir(seq) / "appearance-match.json").read_text(encoding="utf-8"))


def shot_cuts(seq: str) -> list[int]:
    return list(json.loads((run_dir(seq) / "gbf/cuts.json").read_text(encoding="utf-8")).get("cuts", []))


def _touches(r) -> bool:
    return bool(r.x1 <= EDGE_PX or r.y1 <= EDGE_PX or r.x2 >= WIDTH - 1 - EDGE_PX or r.y2 >= HEIGHT - 1 - EDGE_PX)


@lru_cache(maxsize=None)
def pieces(seq: str) -> pd.DataFrame:
    t = tracks(seq)
    t = t[t.track_id >= 0].sort_values(["track_id", "frame"])
    rows = []
    for pid, g in t.groupby("track_id"):
        first, last = g.iloc[0], g.iloc[-1]
        placed = g[g.pitch_usable & g.in_pitch & g.pitch_x.notna()]
        head, tail = placed.head(5), placed.tail(5)
        rows.append({
            "piece": int(pid), "plugin": int(first.source_track_id), "k": int(pid) // 100000,
            "first": int(first.frame), "last": int(last.frame), "rows": len(g),
            "team": g.team.mode().iloc[0] if g.team.notna().any() else None,
            "role": g.role.mode().iloc[0] if g.role.notna().any() else None,
            "starts_at_edge": _touches(first), "ends_at_edge": _touches(last),
            "start_xy": (float(head.pitch_x.median()), float(head.pitch_y.median())) if len(head) else None,
            "end_xy": (float(tail.pitch_x.median()), float(tail.pitch_y.median())) if len(tail) else None,
        })  # fmt: skip
    return pd.DataFrame(rows)


def loco_folds(clips=DEV_CLIPS):
    """Leave-one-clip-out: yields (fit_clips, held_out_clip)."""
    for c in clips:
        yield tuple(x for x in clips if x != c), c
