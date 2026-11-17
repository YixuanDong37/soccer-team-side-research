"""DIAGNOSIS: embedding transforms (centering, within-piece whitening, LDA) and row selection."""
import sys, os, time; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
import numpy as np, pandas as pd, scipy.linalg as sl
from core import ctx
from diag import evaluate, unit, dot_scorer

CL = {s: ctx(s) for s in data.DEV_CLIPS}
E0 = {s: np.where(CL[s].has[:, None], CL[s].E, 0.0) for s in CL}

def team_center(seq, X=None, sel=None):
    c = CL[seq]; X = E0[seq] if X is None else X
    sel = c.clean if sel is None else sel
    ok = c.has & (c.tid >= 0)
    out = X - X[ok & sel].mean(axis=0)
    team = c.t.team.to_numpy()
    for tm in ("left", "right"):
        m = ok & (team == tm)
        if (m & sel).sum() > 50:
            out[m] = X[m] - X[m & sel].mean(axis=0)
    return out

def protos(seq, X, sel, min_rows=5, weight=None):
    c = CL[seq]; out = {}
    for pid, rows in c.rows_of.items():
        r = rows[sel[rows] & c.has[rows]]
        if len(r) < min_rows:
            r = rows[c.has[rows]]
            if not len(r): continue
        if weight is None:
            out[pid] = unit(X[r].mean(axis=0))
        else:
            w = weight[r]; out[pid] = unit((X[r] * w[:, None]).sum(axis=0))
    return out

def run(tag, fX, fsel=lambda c: c.clean, min_rows=5, weight=None):
    return evaluate({s: dot_scorer(protos(s, fX(s), fsel(CL[s]), min_rows, None if weight is None else weight(CL[s]))) for s in CL}, tag)

TC = {s: unit(team_center(s)) for s in CL}
run("team-centred, clean rows mean", lambda s: TC[s])
# centering on unnormalised? (centre, don't renormalise rows)
TCu = {s: team_center(s) for s in CL}
run("team-centred (rows not renormalised)", lambda s: TCu[s])

# within-piece whitening (label-free): Sw from rows around their piece mean
def within_cov(seq, X, sel):
    c = CL[seq]; D = []
    for pid, rows in c.rows_of.items():
        r = rows[sel[rows] & c.has[rows]]
        if len(r) >= 10:
            D.append(X[r] - X[r].mean(axis=0))
    D = np.concatenate(D); return D.T @ D / len(D)

def between_cov(seq, X, sel):
    c = CL[seq]; M = []; 
    for pid, rows in c.rows_of.items():
        r = rows[sel[rows] & c.has[rows]]
        if len(r) >= 10: M.append(X[r].mean(axis=0))
    M = np.array(M); M = M - M.mean(axis=0); return M.T @ M / len(M)

for lam in (0.03, 0.1, 0.3, 1.0, 3.0):
    def fX(s, lam=lam):
        X = TC[s]; Sw = within_cov(s, X, CL[s].clean)
        w, V = np.linalg.eigh(Sw); mean_w = w.mean()
        T = V @ np.diag(1.0 / np.sqrt(w + lam * mean_w)) @ V.T
        return unit(X @ T)
    run(f"team-centred + WCCN lam={lam}", fX)

for k in (16, 32, 64, 128):
    for lam in (0.3,):
        def fX(s, k=k, lam=lam):
            X = TC[s]; Sw = within_cov(s, X, CL[s].clean); Sb = between_cov(s, X, CL[s].clean)
            Sw = Sw + lam * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
            w, V = sl.eigh(Sb, Sw); V = V[:, ::-1][:, :k]
            return unit(X @ V)
        run(f"team-centred + LDA k={k} lam={lam}", fX)
# PCA dims on team-centred
for k in (32, 64, 128):
    def fX(s, k=k):
        X = TC[s]; c = CL[s]; ok = c.has & c.clean
        C = np.cov(X[ok].T); w, V = np.linalg.eigh(C); V = V[:, ::-1][:, :k]
        return unit(X @ V)
    run(f"team-centred + PCA k={k}", fX)
