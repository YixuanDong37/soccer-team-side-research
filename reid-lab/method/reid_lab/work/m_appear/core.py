"""Label-free per-clip context shared by m_appear's code: rows, embeddings, pieces (optionally after cuts).

Reads reid_lab.data only. `cuts` follow reid_lab.score's convention: a cut (piece, frame) gives the
piece's rows from that frame on the id piece + 10**7 * (j + 1) for the j-th cut of that piece by frame,
so the links produced after cuts can be scored with the same cuts.
"""
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data  # noqa: E402

W, H, FPS = data.WIDTH, data.HEIGHT, data.FPS
SUB = 10**7


def norm_cuts(cuts):
    """Hashable, sorted form of a cut list."""
    return tuple(sorted({(int(p), int(f)) for p, f in (cuts or ())}))


def ids_after_cuts(tid, frame, cuts):
    out = tid.copy()
    by = {}
    for p, f in cuts:
        by.setdefault(p, []).append(f)
    for p, fs in by.items():
        for j, f in enumerate(sorted(fs)):
            out[(tid == p) & (frame >= f)] = p + SUB * (j + 1)
    return out


def _touches(t, i):
    return bool(t.x1[i] <= data.EDGE_PX or t.y1[i] <= data.EDGE_PX or t.x2[i] >= W - 1 - data.EDGE_PX or t.y2[i] >= H - 1 - data.EDGE_PX)


def pieces_table(t, tid):
    """data.pieces for a changed track_id column (same columns and rules)."""
    t = t.assign(track_id=tid)
    t = t[t.track_id >= 0].sort_values(["track_id", "frame"])
    rows = []
    for pid, g in t.groupby("track_id"):
        first, last = g.iloc[0], g.iloc[-1]
        placed = g[g.pitch_usable & g.in_pitch & g.pitch_x.notna()]
        head, tail = placed.head(5), placed.tail(5)
        rows.append({
            "piece": int(pid), "plugin": int(first.source_track_id), "k": (int(pid) % SUB) // 100000,
            "first": int(first.frame), "last": int(last.frame), "rows": len(g),
            "team": g.team.mode().iloc[0] if g.team.notna().any() else None,
            "role": g.role.mode().iloc[0] if g.role.notna().any() else None,
            "starts_at_edge": _touches(g, g.index[0]), "ends_at_edge": _touches(g, g.index[-1]),
            "start_xy": (float(head.pitch_x.median()), float(head.pitch_y.median())) if len(head) else None,
            "end_xy": (float(tail.pitch_x.median()), float(tail.pitch_y.median())) if len(tail) else None,
        })  # fmt: skip
    return pd.DataFrame(rows)


class Ctx:
    """Everything label-free about one clip, as arrays aligned with data.tracks(seq) rows."""

    def __init__(self, seq, cuts=()):
        self.seq = seq
        self.cuts = cuts
        t = data.tracks(seq)
        self.t = t
        self.E = data.embeddings(seq)
        self.has = ~np.isnan(self.E[:, 0])
        self.frame = t.frame.to_numpy()
        self.tid = ids_after_cuts(t.track_id.to_numpy(), self.frame, cuts) if cuts else t.track_id.to_numpy()
        x1, y1, x2, y2 = (t[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.h = y2 - y1
        self.w = x2 - x1
        self.edge_dist = np.minimum.reduce([x1, y1, W - 1 - x2, H - 1 - y2])   # px to the nearest image border
        cover = np.zeros(len(t))                                               # largest share covered by another box
        for _, idx in t.groupby("frame").indices.items():
            if len(idx) < 2:
                continue
            ix = np.minimum(x2[idx][:, None], x2[idx][None]) - np.maximum(x1[idx][:, None], x1[idx][None])
            iy = np.minimum(y2[idx][:, None], y2[idx][None]) - np.maximum(y1[idx][:, None], y1[idx][None])
            inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
            np.fill_diagonal(inter, 0.0)
            cover[idx] = inter.max(axis=1) / np.maximum(1e-9, (self.w * self.h)[idx])
        self.cover = cover
        self.clean = data.clean_rows(seq)
        te = {e["track_id"]: e for e in data.refine(seq)["teams"]["tracks"]}

        def verified(tid):
            e = te.get(tid % SUB)                       # a cut piece inherits the check of the piece it came from
            if not e or e["verdict"] != "consistent" or e.get("samples_against") or e.get("cleared"):
                return None
            return e["team"] if e.get("nearer_team") == e["team"] else None

        P = (pieces_table(t, self.tid) if cuts else data.pieces(seq)).copy()
        P["vteam"] = [verified(int(p)) for p in P.piece]
        rows_of = {}
        order = np.lexsort((self.frame, self.tid))
        tid_sorted = self.tid[order]
        for pid in P.piece:
            lo, hi = np.searchsorted(tid_sorted, pid, "left"), np.searchsorted(tid_sorted, pid, "right")
            rows_of[int(pid)] = order[lo:hi]
        self.rows_of = rows_of                           # piece -> row indices in frame order
        P["n_emb"] = [int(self.has[rows_of[int(p)]].sum()) for p in P.piece]
        P["n_clean"] = [int(self.clean[rows_of[int(p)]].sum()) for p in P.piece]
        P["med_h"] = [float(np.median(self.h[rows_of[int(p)]])) for p in P.piece]
        self.P = P.reset_index(drop=True)
        self.index = {int(p): i for i, p in enumerate(self.P.piece)}
        self.frames_of = {pid: set(self.frame[r].tolist()) for pid, r in rows_of.items()}

    def overlap_frames(self, a, b):
        return len(self.frames_of[a] & self.frames_of[b])


@lru_cache(maxsize=None)
def _ctx(seq, cuts):
    return Ctx(seq, cuts)


def ctx(seq, cuts=()):
    return _ctx(seq, norm_cuts(cuts))
