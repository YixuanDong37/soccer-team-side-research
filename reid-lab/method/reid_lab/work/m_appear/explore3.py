"""DIAGNOSIS (uses labels through fitlabels): which embedding transform / prototype separates people best."""
import sys, os, time; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
import numpy as np, pandas as pd, json
from core import ctx
import fitlabels

def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return np.nan
    allv = np.concatenate([pos, neg]); r = pd.Series(allv).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

def evaluate(protos_by_seq, tag, team_rule="label"):
    """protos_by_seq: seq -> {piece: unit vector}. Pairs: a before b, no overlap, not referee, team compatible."""
    pos, neg = [], []; top = []  # (score, is_genuine, has_genuine)
    for seq in data.DEV_CLIPS:
        c = ctx(seq); P = c.P; ident = fitlabels.piece_identity(seq); pr = protos_by_seq[seq]
        ids = [int(p) for p in P.piece if int(p) in pr and P.role[c.index[int(p)]] != "referee"]
        for b in ids:
            B = P.iloc[c.index[b]]; best = None; hasg = False
            for a in ids:
                A = P.iloc[c.index[a]]
                if a == b or not (A["last"] < B["first"]): continue
                if A.team is not None and B.team is not None and A.team != B.team: continue
                if a not in ident or b not in ident: continue
                s = float(pr[a] @ pr[b]); g = ident[a][0] == ident[b][0]
                (pos if g else neg).append(s); hasg |= g
                if best is None or s > best[0]: best = (s, g)
            if best is not None: top.append((best[0], best[1], hasg))
    top = sorted(top, reverse=True)
    r1 = np.mean([g for s, g, h in top if h]); nh = sum(h for s, g, h in top)
    # TP at FP budget over top-1 decisions
    out = {}; tp = fp = 0; budgets = [0, 1, 2, 5, 10]
    cur = {}
    for s, g, h in top:
        if g: tp += 1
        else:
            for bgt in budgets:
                if fp == bgt and bgt not in cur: cur[bgt] = tp
            fp += 1
    for bgt in budgets: cur.setdefault(bgt, tp)
    print(f"{tag:42s} pairs+ {len(pos):4d} - {len(neg):6d} AUC {auc(pos,neg):.4f} | later pieces with partner {nh:3d} R1 {r1:.3f} | top1 TP@FP<=0/1/2/5/10: {cur[0]:3d} {cur[1]:3d} {cur[2]:3d} {cur[5]:3d} {cur[10]:3d}")

def protos(seq, rows_sel, X, min_rows=1, fallback=True):
    c = ctx(seq); out = {}
    for pid, rows in c.rows_of.items():
        r = rows[rows_sel[rows] & c.has[rows]]
        if len(r) < min_rows:
            if not fallback: continue
            r = rows[c.has[rows]]
            if not len(r): continue
        out[pid] = unit(X[r].mean(axis=0))
    return out

t0 = time.time()
ALL = {}
for seq in data.DEV_CLIPS:
    c = ctx(seq); E = np.where(c.has[:, None], c.E, 0.0)
    ALL[seq] = (c, E)
print("ctx", time.time() - t0)

# 1. raw, clean rows (baseline, fallback to all rows when < 5 clean)
evaluate({s: protos(s, ALL[s][0].clean, ALL[s][1], 5) for s in ALL}, "raw clean>=5 (fallback all)")
evaluate({s: protos(s, ALL[s][0].clean, ALL[s][1], 5, False) for s in ALL}, "raw clean>=5 (no fallback)")
evaluate({s: protos(s, np.ones(len(ALL[s][1]), bool), ALL[s][1], 1) for s in ALL}, "raw all rows")
# 2. centering
def centered(seq, mode):
    c, E = ALL[seq]
    ok = c.has & (c.tid >= 0)
    if mode == "global":
        mu = E[ok & c.clean].mean(axis=0); X = unit(E - mu)
    elif mode == "team":
        X = E.copy(); team = c.t.team.to_numpy(); mu_all = E[ok & c.clean].mean(axis=0)
        done = np.zeros(len(E), bool)
        for tm in ("left", "right"):
            m = ok & (team == tm)
            if (m & c.clean).sum() > 50:
                X[m] = E[m] - E[m & c.clean].mean(axis=0); done |= m
        X[~done] = E[~done] - mu_all
        X = unit(X)
    return X
for mode in ("global", "team"):
    evaluate({s: protos(s, ALL[s][0].clean, centered(s, mode), 5) for s in ALL}, f"centered-{mode} clean>=5 (fallback)")
