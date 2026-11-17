import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
seq = "SNGS-107"
t = data.tracks(seq)
print(t.dtypes)
print(t.head(3).T)
r = data.refine(seq)
print(r.keys())
print({k: (type(v).__name__, (list(v.keys()) if isinstance(v, dict) else len(v) if isinstance(v, list) else v)) for k, v in r.items()})
print(json.dumps(r["tracks"], indent=1)[:3000])
print(json.dumps(r["teams"]["tracks"][:2], indent=1))
print([k for k in r["teams"].keys()])
a = data.appearance(seq)
print(a.keys())
print(a["chosen"], len(a["candidate_pairs"]))
print(data.pieces(seq).head(20))
for s in data.DEV_CLIPS:
    rr = data.refine(s)
    p = data.pieces(s)
    print(s, "pieces", len(p), "reentry", len(rr["tracks"]["reentry_pieces"]), "k>0", int((p.k > 0).sum()), "cuts", data.shot_cuts(s), "cand", len(data.appearance(s)["candidate_pairs"]))
