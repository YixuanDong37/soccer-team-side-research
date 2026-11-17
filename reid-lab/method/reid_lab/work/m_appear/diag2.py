"""DIAGNOSIS harness, vectorised (labels through fitlabels; never used to produce the links of a method).

For every clip: the pieces that are not referees and have a known identity; all ordered pairs a -> b
with a ending before b starts and team labels not different. A variant supplies a prototype per piece
(unit vector); the harness reports how well the cosine separates same-person pairs.
"""
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reid_lab import data  # noqa: E402

import fitlabels  # noqa: E402
from core import ctx  # noqa: E402


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ap(score, good):
    o = np.argsort(-score, kind="stable")
    g = good[o].astype(float)
    if g.sum() == 0:
        return np.nan
    prec = np.cumsum(g) / (np.arange(len(g)) + 1)
    return float((prec * g).sum() / g.sum())


_SETS = {}


def pair_set(seq):
    if seq in _SETS:
        return _SETS[seq]
    c = ctx(seq)
    P = c.P
    ident = fitlabels.piece_identity(seq)
    ids = [int(p) for p in P.piece if P.role[c.index[int(p)]] != "referee" and int(p) in ident]
    first = np.array([P["first"][c.index[p]] for p in ids])
    last = np.array([P["last"][c.index[p]] for p in ids])
    team = np.array([P.team[c.index[p]] or "" for p in ids])
    ref = np.array([ident[p][0] for p in ids])
    differ = (team[:, None] != team[None, :]) & (team[:, None] != "") & (team[None, :] != "")
    ok = (last[:, None] < first[None, :]) & ~differ
    same = ref[:, None] == ref[None, :]
    _SETS[seq] = (ids, ok, same)
    return _SETS[seq]


def evaluate(proto_by_seq, tag, clips=data.DEV_CLIPS, quiet=False, exclude=()):
    """proto_by_seq: seq -> {piece: unit vector}; pieces without one are left out."""
    pos, neg, ts, tg, th = [], [], [], [], []
    for seq in clips:
        if seq in exclude:
            continue
        ids, ok, same = pair_set(seq)
        pr = proto_by_seq[seq]
        dim = len(next(iter(pr.values())))
        have = np.array([p in pr for p in ids])
        M = np.stack([pr[p] if p in pr else np.zeros(dim) for p in ids])
        S = M @ M.T
        okk = ok & have[:, None] & have[None, :]
        pos.append(S[okk & same])
        neg.append(S[okk & ~same])
        Sm = np.where(okk, S, -np.inf)
        for j in range(len(ids)):
            if not okk[:, j].any():
                continue
            i = int(np.argmax(Sm[:, j]))
            ts.append(Sm[i, j])
            tg.append(same[i, j])
            th.append((okk[:, j] & same[:, j]).any())
    pos, neg = np.concatenate(pos), np.concatenate(neg)
    ts, tg, th = np.array(ts), np.array(tg), np.array(th)
    o = np.argsort(-ts)
    g = tg[o]
    fp = np.cumsum(~g)
    tp = np.cumsum(g)
    at = {k: int(tp[fp <= k].max()) if (fp <= k).any() else 0 for k in (0, 2, 5, 10, 20)}
    res = dict(tag=tag, pos=len(pos), neg=len(neg), auc=auc(pos, neg), partner=int(th.sum()), r1=float(tg[th].mean()),
               ap=ap(ts, tg), **{f"tp{k}": v for k, v in at.items()})
    if not quiet:
        print(f"{tag:58s} +{len(pos):4d} -{len(neg):5d} AUC {res['auc']:.4f} | partner {res['partner']:3d} R1 {res['r1']:.3f} "
              f"top1-AP {res['ap']:.4f} | TP@FP<=0/2/5/10/20: {at[0]:3d} {at[2]:3d} {at[5]:3d} {at[10]:3d} {at[20]:3d}")
    return res
