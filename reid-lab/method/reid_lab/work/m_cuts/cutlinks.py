"""Label-free: links that a cut suggests.

When a piece switches from person X to person Y, Y usually had a piece of his own that stopped
there (the track took his boxes over) and X usually goes on in a new piece. For every cut:
  side "d" (origin of the after side):  best alternative origin (features.alt_d_pid) -> after sub-piece
  side "r" (continuation of the before side):  before sub-piece -> best alternative continuation
A candidate link carries cos (the cosine to that alternative) and margin (cos minus the cosine
between the two sides of the cut). It is dropped when the two sub-pieces are seen in the same frame
(one person is not in two places).

    link_candidates(seq, cut_df)                 -> DataFrame earlier, later, side, cos, margin
    links(seq, cut_df, margin, floor, sides)     -> [[earlier, later], ...]
    links_default(seq, prob, heldout)            -> (cuts, links) with the stored thresholds
cut_df = features.cut_table(seq, params).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import features as F  # noqa: E402


def link_candidates(seq: str, cut_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["seq", "earlier", "later", "side", "cos", "margin", "cut_piece", "cut_f1", "cut_prob"]
    if cut_df is None or not len(cut_df):
        return pd.DataFrame(columns=cols)
    cl = [[int(r.piece), int(r.f1)] for r in cut_df.itertuples()]
    sub = F.subpiece_ids(cl)
    parts = F.subpieces(seq, cl)
    P = F.piece_table(seq)

    def frames_of(sid):
        s = parts[sid]
        return P[s["piece"]].frame[s["lo"]: s["hi"]]

    rows = []
    for r in cut_df.itertuples():
        for side, pid, alt in (("d", int(r.alt_d_pid), r.alt_d), ("r", int(r.alt_r_pid), r.alt_r)):
            if pid < 0 or not np.isfinite(alt) or not np.isfinite(r.cos):
                continue
            q = P[pid]
            if side == "d":
                earlier = F.sub_of(sub, pid, int(q.frame[q.frame < r.f1][-1]))
                later = F.sub_of(sub, int(r.piece), int(r.f1))
            else:
                earlier = F.sub_of(sub, int(r.piece), int(r.f0))
                later = F.sub_of(sub, pid, int(q.frame[q.frame > r.f0][0]))
            if earlier == later or len(np.intersect1d(frames_of(earlier), frames_of(later))):
                continue
            rows.append({"seq": seq, "earlier": earlier, "later": later, "side": side, "cos": float(alt), "margin": float(alt - r.cos),
                         "cut_piece": int(r.piece), "cut_f1": int(r.f1), "cut_prob": float(r.prob)})  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def links(seq: str, cut_df: pd.DataFrame, margin: float, floor: float, sides=("d", "r")) -> list[list[int]]:
    c = link_candidates(seq, cut_df)
    c = c[(c.margin >= margin) & (c.cos >= floor) & c.side.isin(list(sides))]
    return [[int(a), int(b)] for a, b in sorted(set(zip(c.earlier, c.later)))]


PARAMS_FILE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/cutlink_params.json"


def links_default(seq: str, prob: float = F.DEFAULT_PROB, heldout: bool = False) -> tuple[list[list[int]], list[list[int]]]:
    """(cuts, cut links) of a clip with the stored thresholds (eval_cutlinks.py wrote them for
    prob 0.9 / 0.7 / 0.5): heldout=False -> weights and thresholds from all 13 dev clips;
    heldout=True -> both fitted without this clip (for leave-one-clip-out use on a dev clip)."""
    import json
    with open(PARAMS_FILE, encoding="utf-8") as fh:
        mf = json.load(fh)[str(prob)]["heldout" if heldout else "insample"][seq]
    ct = F.cut_table(seq, F.heldout_params(seq, prob) if heldout else prob)
    cl = [[int(r.piece), int(r.f1)] for r in ct.itertuples()] if len(ct) else []
    return cl, (links(seq, ct, mf[0], mf[1]) if mf is not None else [])
