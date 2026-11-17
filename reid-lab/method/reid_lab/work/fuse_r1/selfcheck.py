"""fuse_r1 / selfcheck.py -- LABEL-FREE check of how well the appearance evidence separates people
on a clip, to be run on any new footage before the fitted thresholds are trusted there.

Why: the plugin's re-id weights (prtreid-soccernet-baseline) were very probably trained on the
SoccerNet-GSR train split, and all 13 dev clips (and the 37 reserved ones) come from that split. If
so, the embeddings separate these players better than players they have never seen, and no
experiment on these clips can show by how much. This check needs no annotation, so it can compare
new footage with the dev clips on the same scale:

  certain different people   pairs of SOLID pieces of the same appearance team that are seen together
                             in >= 5 frames; cosine of their whitened team-centred prototypes (cos_w)
  certain same person        a long piece (>= 75 frames, >= 5 usable boxes in its first and in its last
                             third) against itself: cos_w of the prototype of its first third and of
                             its last third (the two thirds are at least 1 s apart)

Reported per clip: how many pairs of each kind, AUC (chance that a same-person pair scores above a
different-people pair), the 10% and 50% quantile of the same-person cosines, the 95% and 99%
quantile of the different-people cosines, and the share of different-people pairs at or above 0.5
(the "nobody" level of the linker: a pair that looks at least this alike can be joined when the
position evidence supports it).

Limits: the same-person pairs come from ONE tracklet, so they are easier than a return after a long
absence (drift between tracklets is invisible here); the different-people pairs are the co-visible
ones, not the ones that compete for a return. The check can therefore only raise an alarm (values
clearly worse than on the dev clips), it cannot certify the thresholds.

    python -I selfcheck.py            # the 13 dev clips with the frozen cuts -> out/selfcheck.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse_r1")
from reid_lab import data  # noqa: E402
import prep  # noqa: E402
from linker import _appear  # noqa: E402

HERE = Path(__file__).resolve().parent
MIN_SPAN, MIN_PART, COVISIBLE = 75, 5, 5


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="stable")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                      # average ranks on ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pairs(seq: str, cuts=()):
    """(same-person cosines, different-people cosines) of one clip, label-free."""
    af, core = _appear()
    cuts = prep.norm_cuts(cuts)
    c = core.ctx(seq, cuts)
    pr = af.protos(seq, cuts)
    ids = [p for p in af.candidate_ids(seq, pr, cuts) if c.P.rows[c.index[p]] >= prep.N_SOLID]
    tm = pr.team_mean
    E = np.where(c.has[:, None], c.E, 0.0)

    def centre(p):
        t = pr.ateam[p]
        return tm[t] if t in tm and tm[t] is not None else pr.glob

    neg = []
    for i, p in enumerate(ids):
        for q in ids[i + 1:]:
            if pr.ateam[p] == pr.ateam[q] and c.overlap_frames(p, q) >= COVISIBLE:
                neg.append(float(pr.w[p] @ pr.w[q]))
    pos = []
    for p in ids:
        rows = c.rows_of[p]
        f = c.frame[rows]
        if f[-1] - f[0] + 1 < MIN_SPAN:
            continue
        third = (f[-1] - f[0] + 1) / 3.0
        parts = []
        for sel in (f < f[0] + third, f > f[-1] - third):
            r = rows[sel]
            use = r[c.clean[r] & c.has[r]]
            if len(use) < MIN_PART:
                use = r[c.has[r]]
            parts.append(use)
        if min(len(x) for x in parts) < MIN_PART:
            continue
        a, b = (af.unit(pr.T @ (E[x].mean(axis=0) - centre(p))) for x in parts)
        pos.append(float(a @ b))
    return np.array(pos), np.array(neg)


def summary(pos: np.ndarray, neg: np.ndarray) -> dict:
    q = lambda v, p: round(float(np.quantile(v, p)), 3) if len(v) else None  # noqa: E731
    return {"same_n": int(len(pos)), "diff_n": int(len(neg)), "auc": round(auc(pos, neg), 4) if len(pos) and len(neg) else None,
            "same_q10": q(pos, 0.10), "same_q50": q(pos, 0.50), "diff_q95": q(neg, 0.95), "diff_q99": q(neg, 0.99),
            "diff_share_ge_0.5": round(float((neg >= 0.5).mean()), 4) if len(neg) else None}


def table(rows: dict) -> str:
    L = ["| clip | same-person pairs | different-people pairs | AUC | same 10% | same 50% | different 95% | different 99% | different >= 0.5 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, s in rows.items():
        L.append(f"| {k} | {s['same_n']} | {s['diff_n']} | {s['auc']} | {s['same_q10']} | {s['same_q50']} | {s['diff_q95']} | {s['diff_q99']} | {s['diff_share_ge_0.5']} |")
    return "\n".join(L)


def run(clips=data.DEV_CLIPS, fold_of=lambda s: None) -> dict:
    out, P, Ng = {}, [], []
    for s in clips:
        pos, neg = pairs(s, prep.cuts_for(s, fold_of(s)))
        out[s] = summary(pos, neg)
        P.append(pos)
        Ng.append(neg)
    out["all clips pooled"] = summary(np.concatenate(P), np.concatenate(Ng))
    return out


if __name__ == "__main__":
    res = run()
    text = ("# fuse_r1: label-free self-check of the appearance evidence (13 dev clips, frozen cuts)\n\n"
            "cos_w = cosine of whitened team-centred prototypes. Same-person pairs: first third against last third of one long piece. "
            "Different-people pairs: same-team solid pieces seen together in >= 5 frames.\n\n" + table(res) + "\n")
    (HERE / "out" / "selfcheck.md").write_text(text, encoding="utf-8")
    print(text)
