"""Shared helpers of the diag_cues diagnosis (LABEL-BASED: rule 5c, diagnosis only).

Nothing here produces links or cuts. It joins the label-free lab tables (reid_lab.data) with the
SN-GSR annotation so that each cue can be measured against the annotated identity, team, jersey,
role and pitch position.

    ann(seq)      annotation people rows (1080p): frame, ref_id, role, team, jersey, box, pitch_x/y
    matched(seq)  tracks rows matched to annotation rows (gsr.match, boxes x1.5, IoU >= 0.5):
                  one row per matched pair with trow (index into data.tracks(seq)) and the
                  annotation's attributes (suffix _ref)
    piece_ident(seq)  per piece: majority ref_id (needs >= 5 matched rows), purity, annotated
                  team / role / jersey of that person
    returns(seq)  the returns of reid_lab.score (same definition), with rows A and B
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402
from gbf.evaluate import gsr  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "cache"  # derived from labels: delete to rebuild
CACHE.mkdir(exist_ok=True)
TRAIN = score.TRAIN
SCALE = 1.5
W, H = data.WIDTH, data.HEIGHT
FPS = data.FPS
CLIPS = data.DEV_CLIPS


@lru_cache(maxsize=None)
def ann(seq: str) -> pd.DataFrame:
    doc = json.loads((TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    frame_of = {str(im["image_id"]): int(Path(im["file_name"]).stem) - 1 for im in doc["images"]}
    rows = []
    for a in doc["annotations"]:
        if a.get("supercategory") != "object":
            continue
        at = a.get("attributes") or {}
        if at.get("role") not in ("player", "goalkeeper", "referee"):
            continue
        b = a["bbox_image"]
        p = a.get("bbox_pitch") or {}
        rows.append({"frame": frame_of[str(a["image_id"])], "ref_id": int(a["track_id"]), "role": at.get("role"),
                     "team": at.get("team"), "jersey": at.get("jersey"),
                     "x1": float(b["x"]), "y1": float(b["y"]), "x2": float(b["x"]) + float(b["w"]), "y2": float(b["y"]) + float(b["h"]),
                     "pitch_x": p.get("x_bottom_middle", np.nan), "pitch_y": p.get("y_bottom_middle", np.nan)})  # fmt: skip
    df = pd.DataFrame(rows).sort_values(["frame", "ref_id"]).reset_index(drop=True)
    df["pitch_x"] = df["pitch_x"].astype(float)
    df["pitch_y"] = df["pitch_y"].astype(float)
    return df


@lru_cache(maxsize=None)
def reference(seq: str):
    return gsr.load_reference(TRAIN / seq / "Labels-GameState.json")


@lru_cache(maxsize=None)
def matched(seq: str) -> pd.DataFrame:
    """Matched (annotation row, tracks row) pairs; exactly score.clip's matching (cached on disk)."""
    cache = CACHE / f"matched-{seq}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    ref = reference(seq)
    t = data.tracks(seq)
    pred = t[["frame", "track_id", "x1", "y1", "x2", "y2"]].copy()
    for c in ("x1", "y1", "x2", "y2"):
        pred[c] = pred[c] * SCALE
    pairs, _ = gsr.match(ref, pred)
    a = ann(seq)
    # ann() and ref.people are both sorted by (frame, ref_id): same row order
    assert len(a) == len(ref.people) and (a.ref_id.to_numpy() == ref.people.ref_id.to_numpy()).all()
    rr = pairs["ref_row"].to_numpy(np.int64)
    out = pd.DataFrame({"frame": pairs["frame"].to_numpy(np.int64), "ref_id": pairs["ref_id"].to_numpy(np.int64),
                        "piece": pairs["track_id"].to_numpy(np.int64), "iou": pairs["iou"].to_numpy(float),
                        "trow": pairs["pred_row"].to_numpy(np.int64), "arow": rr})  # fmt: skip
    for c in ("role", "team", "jersey", "pitch_x", "pitch_y", "x1", "y1", "x2", "y2"):
        out[c + "_ref"] = a[c].to_numpy()[rr]
    # check against score.clip (the scorer's own matching)
    c = score.clip(seq)
    assert len(c.frames) == len(out) and (c.frames == out.frame.to_numpy()).all() and (c.refs == out.ref_id.to_numpy()).all()
    out.to_parquet(cache)
    return out


@lru_cache(maxsize=None)
def person_attrs(seq: str) -> pd.DataFrame:
    """Per annotated person: role, team, jersey (majority over its rows) and row count."""
    a = ann(seq)
    rows = []
    for rid, g in a.groupby("ref_id"):
        mode = lambda s: (s.dropna().mode().iloc[0] if s.notna().any() else None)  # noqa: E731
        rows.append({"ref_id": int(rid), "role": mode(g.role), "team": mode(g.team), "jersey": mode(g.jersey), "rows": len(g),
                     "team_values": int(g.team.dropna().nunique()), "jersey_values": int(g.jersey.dropna().nunique()),
                     "role_values": int(g.role.nunique())})  # fmt: skip
    return pd.DataFrame(rows).set_index("ref_id")


@lru_cache(maxsize=None)
def piece_ident(seq: str) -> pd.DataFrame:
    """Per piece (data.pieces order): annotated identity (majority person, >= 5 matched rows)."""
    m = matched(seq)
    p = data.pieces(seq).copy()
    pa = person_attrs(seq)
    ident, purity, nm = {}, {}, {}
    for pid, g in m[m.piece >= 0].groupby("piece"):
        vc = g.ref_id.value_counts()
        nm[int(pid)] = len(g)
        if vc.iloc[0] >= score.MIN_MATCHED:
            ident[int(pid)] = int(vc.index[0])
            purity[int(pid)] = float(vc.iloc[0] / len(g))
    p["matched_rows"] = p.piece.map(nm).fillna(0).astype(int)
    p["ref_id"] = p.piece.map(ident)
    p["purity"] = p.piece.map(purity)
    for c in ("role", "team", "jersey"):
        p[c + "_ref"] = [pa.at[int(r), c] if pd.notna(r) else None for r in p.ref_id]
    return p


@lru_cache(maxsize=None)
def returns(seq: str) -> pd.DataFrame:
    """score.clip's returns with the tracks rows A and B and their pieces (-1 when unmatched; cached)."""
    cache = CACHE / f"returns-{seq}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    c = score.clip(seq)
    m = matched(seq)
    rows = []
    for r in c.returns:
        a, b = r["a"], r["b"]
        rows.append({"seq": seq, **{k: r[k] for k in ("ref_id", "last", "first", "gap")},
                     "trow_a": int(m.trow.iat[a]) if a is not None else -1, "trow_b": int(m.trow.iat[b]) if b is not None else -1,
                     "piece_a": int(m.piece.iat[a]) if a is not None else -1, "piece_b": int(m.piece.iat[b]) if b is not None else -1,
                     "scored": a is not None and b is not None})  # fmt: skip
    out = pd.DataFrame(rows)
    out.to_parquet(cache)
    return out


def touch_sides(x1, y1, x2, y2, w=W, h=H, px=data.EDGE_PX):
    """Which image edges a box is within px of: dict of boolean arrays."""
    x1, y1, x2, y2 = (np.asarray(v, float) for v in (x1, y1, x2, y2))
    return {"left": x1 <= px, "top": y1 <= px, "right": x2 >= w - 1 - px, "bottom": y2 >= h - 1 - px}


def auc(pos, neg) -> float:
    """Area under the ROC curve (probability a positive scores above a negative; ties 0.5)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def prec_at_recall(pos, neg, recalls=(0.5, 0.8, 0.9)) -> dict:
    """Precision (and the threshold) at the threshold that keeps the given share of positives."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    out = {}
    for r in recalls:
        if len(pos) == 0:
            out[r] = (float("nan"), float("nan"))
            continue
        thr = np.sort(pos)[::-1][max(int(np.ceil(r * len(pos))) - 1, 0)]
        tp, fp = int((pos >= thr).sum()), int((neg >= thr).sum())
        out[r] = (tp / (tp + fp) if tp + fp else float("nan"), float(thr))
    return out


def qtable(x, qs=(0.5, 0.75, 0.9, 0.95, 0.99)) -> dict:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), **{f"p{int(q * 100)}": round(float(np.quantile(x, q)), 3) for q in qs}, "max": round(float(x.max()), 3)}  # fmt: skip


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)
