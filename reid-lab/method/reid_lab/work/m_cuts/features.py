"""m_cuts: label-free detector of in-piece identity switches (the plugin's track moves onto
another person without leaving the picture), expressed as cuts [piece, frame].

Nothing here reads an annotation or reid_lab.score. Inputs: reid_lab.data only.

    cuts(seq)                 -> [[piece, frame], ...] with the frozen model at P(switch) >= 0.7
                                 (frame = first frame of the new sub-piece; ids as in
                                 reid_lab/score.py: the j-th cut of a piece by frame starts
                                 piece + 10**7 * (j + 1))
    cuts(seq, 0.9)            -> the frozen model at another probability threshold
    cuts_heldout(seq, 0.7)    -> the same with the weights fitted WITHOUT this dev clip: what a
                                 leave-one-clip-out evaluation of a later step must call
    cuts(seq, params)         -> other weights / threshold (loco.py)
    cut_table(seq, params)    -> the chosen cuts with their features and probability
    candidates(seq)           -> every candidate boundary with its features (first level)
    remap_links(seq, links, cuts, how)  -> links between pieces -> links between sub-pieces
    subpiece_ids(cuts), sub_of(sub, piece, frame), subpieces(seq, cuts)

How it works
------------
Candidate boundaries of a piece (rows in frame order; a boundary lies between two consecutive rows
at frames f0 < f1, gap = f1 - f0 - 1 missing frames):
  "gap"  every boundary with gap >= GAP_MIN frames;
  "app"  inside a gap-free stretch: a local minimum (< COS_CAND) of the before/after cosine.
Features of a candidate inside the stretch [lo, hi) of the piece that is still in one part:
  cos   cosine between the mean re-id embeddings of the K_WIN clean rows before and after it
        (clean = data.clean_rows; with fewer than M_MIN clean rows on a side, all its rows);
  mm    cos - min(alt_d, alt_r): alt_d = the best cosine between the after side and the last rows
        of any other piece that was seen in the 10 s before f1 and is not seen right after f1 (a
        better origin of the after side); alt_r = the best cosine between the before side and the
        first rows of any other piece that appears in the 10 s after f0 and is not seen right
        before f0 (a better continuation of the before side). Clipped to +-MARGIN_CLIP;
  lgap  log(1 + gap);
  ld    log(1 + metres on the pitch between the last position before and the first after).
A logistic regression on these four gives the probability that the two sides are different people.
Greedy binary segmentation per piece: the most probable candidate is cut when its probability is
>= the threshold, the features of the remaining candidates are recomputed inside the two parts
(so a window never reaches across an earlier cut), and so on. Candidates between the same two
window rows are one decision (placed at the largest gap), and no cut is made within NMS_ROWS rows
of an earlier cut.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data  # noqa: E402

FPS = data.FPS
SUB = 10**7                 # score.py's id stride for sub-pieces after a cut

# Fixed settings: set by judgement while looking at the dev clips, never searched over (the
# M_CUTS_* environment variables exist only for sensitivity.py, which shows how little they matter).
def _setting(name: str, default: int) -> int:
    return int(os.environ.get("M_CUTS_" + name, default))


K_WIN = _setting("K_WIN", 20)           # rows per side for the before/after descriptor
M_MIN = 3                               # rows needed for a clean-row descriptor (else all rows of the side)
GAP_MIN = _setting("GAP_MIN", 2)        # a gap of at least this many frames is always a candidate
MIN_SIDE_ROWS = 5                       # an "app" candidate keeps at least this many rows on each side
COS_CAND = 0.90                         # appearance minima below this are candidates inside a gap-free stretch
ALT_FRAMES = _setting("ALT_FRAMES", 250)  # alternative origins / continuations within 10 s
ALT_GUARD = 5                           # ... that are not seen within this many frames on the other side
T_TAKE = 3                              # a donor ended / a receiver started within this many frames
NMS_ROWS = _setting("NMS_ROWS", 10)     # no second cut within this many rows of a cut
MARGIN_CLIP = 0.3

# Frozen detector: logistic regression fitted on all 13 dev clips (make_methods.py writes
# final_params.json and, per held-out clip, loco_params.json; the numbers to quote are the
# leave-one-clip-out ones in README.md).
PARAMS_FILE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/final_params.json"
LOCO_PARAMS_FILE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/loco_params.json"
DEFAULT_PROB = 0.7


class Piece:
    """Rows of one piece in frame order."""

    __slots__ = ("pid", "rows", "frame", "E", "clean", "has", "xy", "placed", "box", "team", "role", "conf")

    def __init__(self, pid, rows, t, E, clean):
        self.pid = int(pid)
        self.rows = rows
        self.frame = t.frame.to_numpy()[rows]
        self.E = np.nan_to_num(E[rows])
        self.has = ~np.isnan(E[rows, 0])
        self.clean = clean[rows] & self.has
        self.xy = t[["pitch_x", "pitch_y"]].to_numpy(float)[rows]
        self.placed = t.pitch_usable.to_numpy()[rows] & t.in_pitch.to_numpy()[rows] & ~np.isnan(self.xy[:, 0])
        self.box = t[["x1", "y1", "x2", "y2"]].to_numpy(float)[rows]
        self.conf = t.conf.to_numpy(float)[rows]
        team = t.team.to_numpy()[rows]
        role = t.role.to_numpy()[rows]
        self.team = team[0] if isinstance(team[0], str) else None
        self.role = role[0] if isinstance(role[0], str) else None

    def __len__(self):
        return len(self.rows)


@lru_cache(maxsize=None)
def piece_table(seq: str) -> dict[int, Piece]:
    t = data.tracks(seq)
    E = data.embeddings(seq)
    clean = data.clean_rows(seq)
    out = {}
    tt = t[t.track_id >= 0]
    order = np.lexsort((tt.frame.to_numpy(), tt.track_id.to_numpy()))
    idx = tt.index.to_numpy()[order]
    ids = tt.track_id.to_numpy()[order]
    starts = np.r_[0, np.where(np.diff(ids) != 0)[0] + 1, len(ids)]
    for a, b in zip(starts[:-1], starts[1:]):
        out[int(ids[a])] = Piece(ids[a], idx[a:b], t, E, clean)
    return out


def boundary_cos(p: Piece, k: int = K_WIN):
    """Before/after cosine at every boundary of the piece (nan where a side has < M_MIN rows),
    with clean rows; where that is undefined, with all rows that have an embedding."""
    n = len(p)
    out = np.full(max(n - 1, 0), np.nan)
    for mask in (p.clean, p.has):
        idx = np.where(mask)[0]
        if len(idx) < 2 * M_MIN:
            continue
        C = np.vstack([np.zeros((1, p.E.shape[1])), np.cumsum(p.E[idx], axis=0)])
        cnt = np.cumsum(mask)[:-1]                     # usable rows with index <= i
        b0 = np.maximum(0, cnt - k)
        a1 = np.minimum(len(idx), cnt + k)
        ok = (cnt - b0 >= M_MIN) & (a1 - cnt >= M_MIN) & np.isnan(out)
        mb = C[cnt[ok]] - C[b0[ok]]
        ma = C[a1[ok]] - C[cnt[ok]]
        out[ok] = (mb * ma).sum(axis=1) / (np.linalg.norm(mb, axis=1) * np.linalg.norm(ma, axis=1))
    return out


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def _pitch_side(p: Piece, i: int, side: int, n: int = 3, within: int = 12):
    """Median pitch position of up to n placed rows next to boundary i on one side, no further
    than `within` frames from the boundary row; (xy, frame of the nearest used row) or (None, None)."""
    if side < 0:
        idx = np.where(p.placed[: i + 1] & (p.frame[: i + 1] >= p.frame[i] - within))[0][-n:]
        near = idx[-1] if len(idx) else None
    else:
        idx = np.where(p.placed[i + 1:] & (p.frame[i + 1:] <= p.frame[i + 1] + within))[0][:n] + i + 1
        near = idx[0] if len(idx) else None
    if near is None:
        return None, None
    return np.median(p.xy[idx], axis=0), int(p.frame[near])


# ----------------------------------------------------------------------------- per-clip context
class _Ctx:
    """Per-clip lookups: usable-row cumulative sums per piece, pieces by first/last frame."""

    def __init__(self, seq: str):
        self.seq = seq
        self.P = piece_table(seq)
        self.cum = {}
        for pid, p in self.P.items():
            ent = []
            for mask in (p.clean, p.has):
                idx = np.where(mask)[0]
                C = np.vstack([np.zeros((1, p.E.shape[1])), np.cumsum(p.E[idx], axis=0)])
                ent.append((idx, C))
            self.cum[pid] = ent
        self.ends, self.starts = {}, {}
        for p in self.P.values():
            self.ends.setdefault(int(p.frame[-1]), []).append(p)
            self.starts.setdefault(int(p.frame[0]), []).append(p)
        t = data.tracks(seq)
        tt = t[t.track_id >= 0]
        self.foot = {}
        fu = ((tt.x1 + tt.x2) / 2).to_numpy(float)
        fv = tt.y2.to_numpy(float)
        tid = tt.track_id.to_numpy()
        for f, idx in tt.groupby("frame").indices.items():
            self.foot[int(f)] = (tid[idx], fu[idx], fv[idx])

    def desc(self, pid: int, lo: int, hi: int, take: str, k: int = K_WIN, small: bool = False):
        """Mean embedding of the k usable rows of rows[lo:hi] of a piece nearest its `take` end
        ("last" / "first"): clean rows when there are >= M_MIN, else rows with an embedding
        (>= M_MIN, or with small=True at least one).
        -> (unit vector | None, rows used, clean?, row index of the nearest used row | -1)"""
        for which, (idx, C) in enumerate((*self.cum[pid], self.cum[pid][1])):
            if which == 2 and not small:
                break
            a, b = np.searchsorted(idx, lo), np.searchsorted(idx, hi)
            if b - a >= (M_MIN if which < 2 else 1):
                if take == "last":
                    a = max(a, b - k)
                    near = int(idx[b - 1])
                else:
                    b = min(b, a + k)
                    near = int(idx[a])
                v = C[b] - C[a]
                return v / np.linalg.norm(v), int(b - a), which == 0, near
        return None, 0, False, -1

    def common_shift(self, f0: int, f1: int, skip: int):
        """Median image displacement (du, dv) between two frames of the foot points of the other
        pieces seen in both (camera motion); None with fewer than 3."""
        a, b = self.foot.get(f0), self.foot.get(f1)
        if a is None or b is None:
            return None
        ia = {int(t): j for j, t in enumerate(a[0])}
        du, dv = [], []
        for j, t in enumerate(b[0]):
            t = int(t)
            if t != skip and t in ia:
                du.append(b[1][j] - a[1][ia[t]])
                dv.append(b[2][j] - a[2][ia[t]])
        if len(du) < 3:
            return None
        return float(np.median(du)), float(np.median(dv))


@lru_cache(maxsize=None)
def ctx(seq: str) -> _Ctx:
    return _Ctx(seq)


# ----------------------------------------------------------------------------- candidates
def _split_in_plateau(p: Piece, lo: int, hi: int, mb, ma) -> int:
    """Boundary in [lo, hi] (boundary j lies between rows j and j+1) where the rows switch from
    looking like the before mean to looking like the after mean."""
    if hi <= lo:
        return lo
    rows = np.arange(lo + 1, hi + 1)          # rows strictly inside the plateau
    s = np.where(p.has[rows], p.E[rows] @ (mb - ma), 0.0)   # > 0: looks like before
    cum = np.r_[0.0, np.cumsum(s)]
    gain = cum - (s.sum() - cum)              # boundary lo + m keeps the first m plateau rows before
    return lo + int(np.argmax(gain))


def _boundaries(c: _Ctx, p: Piece) -> dict[int, str]:
    """{boundary index: kind} of the piece's candidate boundaries."""
    n = len(p)
    if n < 2:
        return {}
    gaps = np.diff(p.frame) - 1
    cand = {int(i): "gap" for i in np.where(gaps >= GAP_MIN)[0]}
    if n < 2 * MIN_SIDE_ROWS:
        return cand
    cs = boundary_cos(p)
    allowed = np.zeros(n - 1, bool)
    allowed[MIN_SIDE_ROWS - 1: n - MIN_SIDE_ROWS] = True
    low = np.where(~np.isnan(cs) & (cs < COS_CAND) & allowed)[0]
    if len(low):
        order = low[np.argsort(cs[low], kind="stable")]
        taken = []
        usable = p.clean if p.clean.sum() >= 2 * M_MIN else p.has
        ui = np.where(usable)[0]
        for i in order:
            if any(abs(i - j) <= K_WIN for j in taken):
                continue
            taken.append(int(i))
            # plateau: the boundaries between the same two usable rows share the value
            k = np.searchsorted(ui, i, side="right")          # usable rows <= i
            lo = int(ui[k - 1]) if k > 0 else 0
            hi = int(ui[k] - 1) if k < len(ui) else n - 2
            lo, hi = max(lo, MIN_SIDE_ROWS - 1), min(hi, n - MIN_SIDE_ROWS - 1)
            if hi < lo:
                continue
            if any(lo - K_WIN // 2 <= j <= hi + K_WIN // 2 for j, kind in cand.items() if kind == "gap"):
                continue                                       # a gap candidate covers it
            mb = c.desc(p.pid, 0, lo + 1, "last")[0]
            ma = c.desc(p.pid, hi + 1, n, "first")[0]
            j = _split_in_plateau(p, lo, hi, mb, ma) if mb is not None and ma is not None else (lo + hi) // 2
            cand[int(j)] = "app"
    return cand


def _static(c: _Ctx, p: Piece, i: int, kind: str) -> dict:
    """Everything about a candidate that does not depend on the other cuts of its piece."""
    f = p.frame
    f0, f1 = int(f[i]), int(f[i + 1])
    gap = f1 - f0 - 1
    xb, fb = _pitch_side(p, i, -1)
    xa, fa = _pitch_side(p, i, +1)
    if xb is not None and xa is not None:
        d, dt = float(np.hypot(*(xa - xb))), (fa - fb) / FPS
    else:
        d, dt = np.nan, np.nan
    # image jump, camera motion removed, in box heights (short gaps only)
    h = 0.5 * ((p.box[i, 3] - p.box[i, 1]) + (p.box[i + 1, 3] - p.box[i + 1, 1]))
    du = (p.box[i + 1, 0] + p.box[i + 1, 2]) / 2 - (p.box[i, 0] + p.box[i, 2]) / 2
    dv = p.box[i + 1, 3] - p.box[i, 3]
    cm = c.common_shift(f0, f1, p.pid) if gap <= 12 else None
    jump_h = float(np.hypot(du - cm[0], dv - cm[1]) / h) if cm is not None else np.nan
    # takeover: a piece that ended just before f1 where this piece continues (donor) / a piece
    # that started just after f0 where this piece stopped (receiver)
    d_iou, d_pid, r_iou, r_pid = 0.0, -1, 0.0, -1
    for T in range(1, T_TAKE + 1):
        for q in c.ends.get(f1 - T, []):
            v = iou(q.box[-1], p.box[i + 1]) if q.pid != p.pid else 0.0
            if v > d_iou:
                d_iou, d_pid = v, q.pid
        for q in c.starts.get(f0 + T, []):
            v = iou(q.box[0], p.box[i]) if q.pid != p.pid else 0.0
            if v > r_iou:
                r_iou, r_pid = v, q.pid
    # alternative origins of the after side / continuations of the before side
    D, Dp, R, Rp = [], [], [], []
    for q in c.P.values():
        if q.pid == p.pid:
            continue
        j1 = np.searchsorted(q.frame, f1)                      # rows of q before f1
        if j1 > 0 and q.frame[j1 - 1] >= f1 - ALT_FRAMES and not (j1 < len(q) and q.frame[j1] <= f1 + ALT_GUARD):
            v = c.desc(q.pid, 0, j1, "last")[0]
            if v is not None:
                D.append(v)
                Dp.append(q.pid)
        j0 = np.searchsorted(q.frame, f0, side="right")        # rows of q up to f0
        if j0 < len(q) and q.frame[j0] <= f0 + ALT_FRAMES and not (j0 > 0 and q.frame[j0 - 1] >= f0 - ALT_GUARD):
            v = c.desc(q.pid, j0, len(q), "first")[0]
            if v is not None:
                R.append(v)
                Rp.append(q.pid)
    dim = p.E.shape[1]
    return {"seq": c.seq, "piece": p.pid, "i": i, "f0": f0, "f1": f1, "gap": gap, "kind": kind,
            "d_m": d, "dt": dt, "jump_h": jump_h, "iou_self": iou(p.box[i], p.box[i + 1]),
            "d_iou": d_iou, "d_pid": d_pid, "r_iou": r_iou, "r_pid": r_pid,
            "team": p.team, "role": p.role, "box_h": float(h),
            "_D": np.array(D).reshape(-1, dim), "_Dp": Dp, "_R": np.array(R).reshape(-1, dim), "_Rp": Rp}  # fmt: skip


@lru_cache(maxsize=None)
def _static_candidates(seq: str) -> dict[int, list[dict]]:
    c = ctx(seq)
    out = {}
    for p in c.P.values():
        b = _boundaries(c, p)
        if b:
            out[p.pid] = [_static(c, p, i, kind) for i, kind in sorted(b.items())]
    return out


def _dynamic(c: _Ctx, p: Piece, s: dict, lo: int, hi: int) -> dict:
    """Features of a candidate inside the stretch rows[lo:hi] of its piece."""
    i = s["i"]
    B, nb, cb, near_b = c.desc(p.pid, lo, i + 1, "last", small=True)
    A, na, ca, near_a = c.desc(p.pid, i + 1, hi, "first", small=True)
    cos = float(B @ A) if B is not None and A is not None else np.nan
    alt_d, alt_r, alt_d_pid, alt_r_pid = np.nan, np.nan, -1, -1
    if A is not None and len(s["_D"]):
        v = s["_D"] @ A
        alt_d, alt_d_pid = float(v.max()), s["_Dp"][int(v.argmax())]
    if B is not None and len(s["_R"]):
        v = s["_R"] @ B
        alt_r, alt_r_pid = float(v.max()), s["_Rp"][int(v.argmax())]
    out = {k: v for k, v in s.items() if not k.startswith("_")}
    out.update({"cos": cos, "nb": nb, "na": na, "clean_b": cb, "clean_a": ca, "near_b": near_b, "near_a": near_a,
                "alt_d": alt_d, "alt_r": alt_r, "alt_d_pid": alt_d_pid, "alt_r_pid": alt_r_pid,
                "rows_before": i + 1 - lo, "rows_after": hi - i - 1})  # fmt: skip
    return out


@lru_cache(maxsize=None)
def candidates(seq: str) -> pd.DataFrame:
    """One row per candidate boundary of a piece of the clip, with its features computed on the
    whole piece (first level, before any cut)."""
    c = ctx(seq)
    rows = [_dynamic(c, c.P[pid], s, 0, len(c.P[pid])) for pid, ss in _static_candidates(seq).items() for s in ss]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- model
def design(cand: pd.DataFrame) -> pd.DataFrame:
    """Model inputs of candidates: cos (1 when undefined); mm = the larger of the two margins
    cos - alt_d and cos - alt_r, each clipped to +-MARGIN_CLIP, a missing alternative counting as
    +MARGIN_CLIP (also md, mr, mn = the two margins and the smaller one); lgap = log(1 + gap);
    ld = log(1 + pitch distance across the boundary) (0 when unknown); side = log(rows of the
    smaller window)."""
    cos = cand.cos.astype(float)
    md = (cos - cand.alt_d.astype(float)).clip(-MARGIN_CLIP, MARGIN_CLIP).fillna(MARGIN_CLIP)
    mr = (cos - cand.alt_r.astype(float)).clip(-MARGIN_CLIP, MARGIN_CLIP).fillna(MARGIN_CLIP)
    return pd.DataFrame({
        "cos": cos.fillna(1.0), "mm": np.maximum(md, mr), "mn": np.minimum(md, mr), "md": md, "mr": mr,
        "lgap": np.log1p(cand.gap.astype(float)), "ld": np.log1p(cand.d_m.astype(float).fillna(0.0)),
        "side": np.log(np.minimum(cand.nb, cand.na).clip(lower=1).astype(float)),
    }, index=cand.index)  # fmt: skip


def apply_model(par: dict, cand: pd.DataFrame) -> np.ndarray:
    """Logit of "the two sides are different people" for candidates (rows with features)."""
    X = design(cand)[par["cols"]].to_numpy(float)
    return ((X - np.array(par["mu"])) / np.array(par["sd"])) @ np.array(par["w"]) + par["b"]


@lru_cache(maxsize=None)
def _frozen() -> dict:
    import json
    with open(PARAMS_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def _resolve(params) -> dict:
    """None -> frozen model at DEFAULT_PROB; a float -> frozen model at that probability."""
    if params is None or isinstance(params, (int, float)):
        p0 = DEFAULT_PROB if params is None else float(params)
        return {**_frozen(), "threshold": float(np.log(p0 / (1 - p0)))}
    return params


# ----------------------------------------------------------------------------- segmentation
def _segment_piece(c: _Ctx, p: Piece, statics: list[dict], par: dict) -> list[dict]:
    """Greedy binary segmentation of one piece: returns the chosen candidates (their features at
    the moment they were cut, "logit" and the stretch [lo, hi) they split)."""
    th = par["threshold"]
    bounds = [0, len(p)]                     # row indices where stretches start (sorted)
    cut_rows: list[int] = []
    chosen = []
    while True:
        rows = []
        for s in statics:
            i = s["i"]
            if any(abs(i - j) <= NMS_ROWS for j in cut_rows):
                continue
            k = int(np.searchsorted(bounds, i, side="right"))
            rows.append(_dynamic(c, p, s, bounds[k - 1], bounds[k]))
            rows[-1]["lo"], rows[-1]["hi"] = bounds[k - 1], bounds[k]
        if not rows:
            break
        df = pd.DataFrame(rows)
        df["logit"] = apply_model(par, df)
        # candidates with the same before/after windows are one decision, placed at the largest gap
        key = ["lo", "near_b", "near_a"]
        df["_best"] = df.groupby(key).logit.transform("max")
        rep = df.sort_values(["gap", "i"], ascending=[False, True], kind="stable").drop_duplicates(key)
        rep = rep.sort_values(["_best", "gap"], ascending=False, kind="stable")
        top = rep.iloc[0]
        if not (top._best >= th):
            break
        d = top.to_dict()
        d["logit"] = float(d.pop("_best"))
        chosen.append(d)
        cut_rows.append(int(top.i))
        bounds = sorted(bounds + [int(top.i) + 1])
    return chosen


def cut_table(seq: str, params=None) -> pd.DataFrame:
    """The cuts of a clip with their features: piece, f0 (last frame before), f1 (first frame of
    the new sub-piece), logit, prob, ..."""
    par = _resolve(params)
    c = ctx(seq)
    rows = []
    for pid, statics in _static_candidates(seq).items():
        rows += _segment_piece(c, c.P[pid], statics, par)
    df = pd.DataFrame(rows)
    if len(df):
        df["prob"] = 1.0 / (1.0 + np.exp(-df.logit))
        df = df.sort_values(["piece", "f1"]).reset_index(drop=True)
    return df


def cuts(seq: str, params=None) -> list[list[int]]:
    """Label-free cuts of a clip: [[piece, frame], ...]; `frame` is the first frame of the new
    sub-piece. params: None (frozen model, probability >= DEFAULT_PROB), a probability in (0, 1)
    (frozen model at that operating point), or a full parameter dict (loco.py)."""
    df = cut_table(seq, params)
    return [[int(r.piece), int(r.f1)] for r in df.itertuples()] if len(df) else []


def heldout_params(seq: str, prob: float = DEFAULT_PROB) -> dict:
    """The weights fitted WITHOUT this clip (loco_params.json, written by make_methods.py): what a
    leave-one-clip-out evaluation of a later step must use for a dev clip."""
    import json
    with open(LOCO_PARAMS_FILE, encoding="utf-8") as fh:
        folds = json.load(fh)["folds"]
    return {**folds[seq], "threshold": float(np.log(prob / (1 - prob)))}


def cuts_heldout(seq: str, prob: float = DEFAULT_PROB) -> list[list[int]]:
    """cuts(seq) with the weights that were fitted on the other 12 dev clips."""
    return cuts(seq, heldout_params(seq, prob))


# ----------------------------------------------------------------------------- ids and links
def subpiece_ids(cuts) -> dict[int, list[tuple[int, int]]]:
    """{piece: [(first frame of the sub-piece, its id), ...]} as score.py numbers them: the rows
    before the first cut keep the piece id, the j-th cut by frame starts piece + SUB * (j + 1)."""
    by = {}
    for p, f in cuts or []:
        by.setdefault(int(p), set()).add(int(f))
    return {p: [(-1, p)] + [(f, p + SUB * (j + 1)) for j, f in enumerate(sorted(fs))] for p, fs in by.items()}


def sub_of(sub: dict, piece: int, frame: int) -> int:
    """Id of the sub-piece of `piece` that holds `frame`."""
    out = int(piece)
    for f, sid in sub.get(int(piece), ()):
        if frame >= f:
            out = sid
    return out


def subpieces(seq: str, cuts) -> dict[int, dict]:
    """{sub-piece id: {"piece", "lo", "hi" (row range in the piece), "first", "last" (frames)}}
    for every piece of the clip (uncut pieces have one entry, their own id)."""
    P = piece_table(seq)
    sub = subpiece_ids(cuts)
    out = {}
    for pid, p in P.items():
        parts = sub.get(pid, [(-1, pid)])
        starts = [int(np.searchsorted(p.frame, f)) if f >= 0 else 0 for f, _ in parts] + [len(p)]
        for (f, sid), lo, hi in zip(parts, starts[:-1], starts[1:]):
            if hi > lo:
                out[sid] = {"piece": pid, "lo": lo, "hi": hi, "first": int(p.frame[lo]), "last": int(p.frame[hi - 1])}
    return out


def remap_links(seq: str, links, cuts, how: str = "appearance"):
    """Links between pieces -> links between sub-pieces after cuts.
    "appearance": among the sub-pieces of the earlier piece and of the later piece that are in
        time order (the first ends before the second starts), the pair whose descriptors (mean
        embedding of all clean rows of the sub-piece) are most alike; sub-pieces without M_MIN rows
        with an embedding are not considered; when no pair qualifies, as "adjacent";
    "adjacent": the earlier piece's last sub-piece before the later piece starts and the later
        piece's sub-piece at its start (the two stretches next to the gap the link bridges);
    "drop": links that touch a cut piece are dropped."""
    sub = subpiece_ids(cuts)
    if not sub:
        return [[int(a), int(b)] for a, b in links or []]
    P = piece_table(seq)
    c = ctx(seq)
    parts = subpieces(seq, cuts)
    of_piece = {}
    for sid, s in parts.items():
        of_piece.setdefault(s["piece"], []).append(sid)
    out = []
    for a, b in links or []:
        a, b = int(a), int(b)
        if a not in sub and b not in sub:
            out.append([a, b])
            continue
        if how == "drop":
            continue
        if P[a].frame[0] > P[b].frame[0]:
            a, b = b, a
        fb = int(P[b].frame[0])
        fa = int(P[a].frame[P[a].frame < fb][-1]) if (P[a].frame < fb).any() else int(P[a].frame[0])
        pair = [sub_of(sub, a, fa), sub_of(sub, b, fb)]
        if how == "appearance":
            best = -2.0
            for sa in of_piece[a]:
                da = c.desc(a, parts[sa]["lo"], parts[sa]["hi"], "last", k=10**6)[0]
                if da is None:
                    continue
                for sb in of_piece[b]:
                    if parts[sa]["last"] >= parts[sb]["first"]:
                        continue
                    db = c.desc(b, parts[sb]["lo"], parts[sb]["hi"], "first", k=10**6)[0]
                    if db is not None and float(da @ db) > best:
                        best, pair = float(da @ db), [sa, sb]
        out.append(pair)
    return out
