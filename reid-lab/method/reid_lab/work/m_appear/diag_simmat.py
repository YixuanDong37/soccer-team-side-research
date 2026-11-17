"""DIAGNOSIS: similarity matrix between named pieces of a clip (raw and team-centred clean-row means), with identities."""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
import numpy as np, pandas as pd
from core import ctx
from diag import unit
import fitlabels
seq = sys.argv[1]; A = [int(x) for x in sys.argv[2].split(",")]; B = [int(x) for x in sys.argv[3].split(",")]
c = ctx(seq); ident = fitlabels.piece_identity(seq)
X = np.where(c.has[:, None], c.E, 0.0); ok = c.has & (c.tid >= 0); out = X - X[ok & c.clean].mean(axis=0); team = c.t.team.to_numpy()
for tm in ("left", "right"):
    m = ok & (team == tm)
    if (m & c.clean).sum() > 50: out[m] = X[m] - X[m & c.clean].mean(axis=0)
Xc = unit(out)
def pr(X, p):
    r = c.rows_of[p]; rr = r[c.clean[r]]
    if len(rr) < 5: rr = r[c.has[r]]
    return unit(X[rr].mean(axis=0))
P = c.P.set_index("piece")
for p in A + B:
    print(p, "frames", P.loc[p, "first"], P.loc[p, "last"], "rows", P.loc[p, "rows"], "clean", P.loc[p, "n_clean"], "team", P.loc[p, "team"], "ident", ident.get(p), "med_h", round(P.loc[p, "med_h"]))
for name, XX in (("raw", X), ("team-centred", Xc)):
    M = pd.DataFrame([[round(float(pr(XX, a) @ pr(XX, b)), 3) for b in B] for a in A], index=A, columns=B)
    print(name); print(M.to_string())
