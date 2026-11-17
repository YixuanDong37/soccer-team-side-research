"""LABEL-BASED helpers of m_cuts (reads the annotation through reid_lab.score).

Used only (a) to fit weights / thresholds inside leave-one-clip-out, (b) to count how many held-out
cuts sit on a true identity switch, (c) for diagnosis.  features.py (the method) never imports this.

True switch of a piece: its matched rows (score.clip: gbf box matched to an annotated person), in
frame order, form runs of one person; runs shorter than MIN_SEG matched rows are dropped; where two
consecutive remaining runs are different people the piece switched (last_a = last frame of the
first person, first_b = first frame of the second).

A candidate / cut at (piece, f0 | f1) is
  "T" when it separates a true switch leaving at most TOL rows of the piece on the wrong side, or
      when the majority person of the 10 matched rows before it differs from the 10 after it;
  "F" when the majority person before and after is the same (>= 3 matched rows on each side);
  "t" / "f" the same from only 1-2 matched rows on a side (weak; never used to fit or to count
      precision, reported separately);
  "U" otherwise (a side has no matched row).
"""

from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
from reid_lab import data, score  # noqa: E402
import features as F  # noqa: E402

MIN_SEG = 5
TOL = 5
MIN_LOC = 3     # matched rows needed on each side for the local before/after identity


def _runs(frames, refs, min_seg=MIN_SEG):
    rl = []
    for f, r in zip(frames, refs):
        if rl and rl[-1][0] == r:
            rl[-1][2] = f
            rl[-1][3] += 1
        else:
            rl.append([r, f, f, 1])
    while True:
        keep = [x for x in rl if x[3] >= min_seg]
        merged = []
        for x in keep:
            if merged and merged[-1][0] == x[0]:
                merged[-1][2] = x[2]
                merged[-1][3] += x[3]
            else:
                merged.append(list(x))
        if len(merged) == len(rl):
            return merged
        rl = merged


@lru_cache(maxsize=None)
def matched(seq: str):
    c = score.clip(seq)
    m = pd.DataFrame({"frame": c.frames, "ref": c.refs, "piece": c.piece})
    m = m[m.piece >= 0].sort_values(["piece", "frame"])
    return {int(pid): (g.frame.to_numpy(), g.ref.to_numpy()) for pid, g in m.groupby("piece")}


@lru_cache(maxsize=None)
def switches(seq: str) -> list[dict]:
    out = []
    for pid, (fr, rf) in matched(seq).items():
        rl = _runs(fr, rf)
        for a, b in zip(rl[:-1], rl[1:]):
            out.append({"piece": pid, "from_ref": int(a[0]), "to_ref": int(b[0]), "last_a": int(a[2]), "first_b": int(b[1]),
                        "rows_a": int(a[3]), "rows_b": int(b[3])})  # fmt: skip
    return out


def label(seq: str, cand: pd.DataFrame):
    """-> (labels "T"/"F"/"U", index of the true switch hit or -1) for rows with piece, f0, f1."""
    sws = switches(seq)
    mp = matched(seq)
    P = F.piece_table(seq)
    lab, which = [], []
    for r in cand.itertuples():
        pf = P[r.piece].frame
        hit = -1
        for k, s in enumerate(sws):
            if s["piece"] == r.piece and ((pf >= r.f1) & (pf <= s["last_a"])).sum() <= TOL and ((pf >= s["first_b"]) & (pf <= r.f0)).sum() <= TOL:
                hit = k
                break
        mf, mr = mp.get(r.piece, (np.zeros(0, int), np.zeros(0, int)))
        b, a = mr[mf <= r.f0][-10:], mr[mf >= r.f1][:10]
        diff = bool(np.bincount(a).argmax() != np.bincount(b).argmax()) if len(a) and len(b) else None
        loc = None if len(a) < MIN_LOC or len(b) < MIN_LOC else diff
        if hit >= 0 or loc is True:
            lab.append("T")
        elif loc is False:
            lab.append("F")
        else:                       # weak: a side has only 1-2 matched rows ("t"/"f"), or none ("U")
            lab.append("U" if diff is None else ("t" if diff else "f"))
        which.append(hit)
    return np.array(lab), np.array(which)
