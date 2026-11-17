"""DIAGNOSIS (labels for display only): why a pair of pieces is or is not linked: best candidates on both sides with identities."""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
import numpy as np, pandas as pd
from agglo import agglomerate
from features import pair_features
from core import ctx
import fitlabels
seq, kind, t_abs, margin, a, b = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
log = []
links = agglomerate(seq, kind=kind, t_abs=t_abs, margin=margin, mutual=True, log=log)
ident = fitlabels.piece_identity(seq); F = pair_features(seq); P = ctx(seq).P.set_index("piece")
def nm(p): return f"{p}[{ident.get(p, ('?',))[0]}|{P.loc[p,'first']}-{P.loc[p,'last']}|{P.loc[p,'n_clean']}c]"
for e in log:
    if a in e["g"] + e["h"] or b in e["g"] + e["h"]:
        print(f"merge s={e['s']:.3f} rivals {e['rival_g']:.2f}/{e['rival_h']:.2f}: {[nm(p) for p in e['g']]} + {[nm(p) for p in e['h']]}")
c = f"cos_{kind}"
print("later candidates of", nm(a)); x = F[F.a == a].sort_values(c, ascending=False).head(6)
print("  ", [(nm(int(p)), round(s, 3)) for p, s in zip(x.b, x[c])])
print("earlier candidates of", nm(b)); x = F[F.b == b].sort_values(c, ascending=False).head(6)
print("  ", [(nm(int(p)), round(s, 3)) for p, s in zip(x.a, x[c])])
for p in (a, b):
    co = [q for q in P.index if q != p and P.loc[q, "first"] <= P.loc[p, "last"] and P.loc[p, "first"] <= P.loc[q, "last"] and P.loc[q, "role"] != "referee"]
    print("co-visible with", p, ":", [nm(q) for q in co][:40])
