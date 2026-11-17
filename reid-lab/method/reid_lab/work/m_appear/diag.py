"""DIAGNOSIS helpers (use labels through fitlabels; never used to produce a method's links)."""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
import numpy as np, pandas as pd
from core import ctx
import fitlabels


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(score_by_seq, tag, clips=data.DEV_CLIPS, quiet=False):
    """score_by_seq: seq -> function(a, b) -> similarity or None (pair not scoreable).
    Pairs: a ends before b starts, neither is a referee, team labels not different, both identities known."""
    pos, neg, top = [], [], []
    for seq in clips:
        c = ctx(seq); P = c.P; ident = fitlabels.piece_identity(seq); f = score_by_seq[seq]
        ids = [int(p) for p in P.piece if P.role[c.index[int(p)]] != "referee" and int(p) in ident]
        for b in ids:
            B = P.iloc[c.index[b]]; best = None; hasg = False
            for a in ids:
                A = P.iloc[c.index[a]]
                if a == b or not (A["last"] < B["first"]):
                    continue
                if A.team is not None and B.team is not None and A.team != B.team:
                    continue
                s = f(a, b)
                if s is None:
                    continue
                g = ident[a][0] == ident[b][0]
                (pos if g else neg).append(s); hasg |= g
                if best is None or s > best[0]:
                    best = (s, g)
            if best is not None:
                top.append((best[0], best[1], hasg))
    top = sorted(top, reverse=True)
    r1 = float(np.mean([g for s, g, h in top if h])); nh = sum(h for s, g, h in top)
    tp = fp = 0; budgets = [0, 1, 2, 5, 10]; cur = {}
    for s, g, h in top:
        if g:
            tp += 1
        else:
            for bgt in budgets:
                if fp == bgt and bgt not in cur:
                    cur[bgt] = tp
            fp += 1
    for bgt in budgets:
        cur.setdefault(bgt, tp)
    res = dict(tag=tag, pos=len(pos), neg=len(neg), auc=auc(pos, neg), with_partner=nh, r1=r1, **{f"tp_fp{k}": v for k, v in cur.items()})
    if not quiet:
        print(f"{tag:50s} +{len(pos):4d} -{len(neg):6d} AUC {res['auc']:.4f} | partner {nh:3d} R1 {r1:.3f} | top1 TP@FP<=0/1/2/5/10: {cur[0]:3d} {cur[1]:3d} {cur[2]:3d} {cur[5]:3d} {cur[10]:3d}")
    return res


def dot_scorer(pr):
    return lambda a, b: float(pr[a] @ pr[b]) if a in pr and b in pr else None
