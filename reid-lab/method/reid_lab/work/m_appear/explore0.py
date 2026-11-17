import sys; sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
import numpy as np, pandas as pd, json
seq = "SNGS-107"
t = data.tracks(seq)
print(t.dtypes)
print(t.head(3).T)
p = data.pieces(seq)
print(p.head(20))
print(len(p), p.role.value_counts(dropna=False).to_dict(), p.team.value_counts(dropna=False).to_dict())
r = data.refine(seq)
print(r.keys())
print({k: (type(v).__name__, (list(v.keys())[:20] if isinstance(v, dict) else None)) for k, v in r.items()})
print(json.dumps(r["teams"]["tracks"][:2], indent=1)[:1500])
a = data.appearance(seq)
print(a.keys())
print(a["chosen"], a["real_candidate_pairs"], a["eligible_pieces"], a["pieces"])
e = data.embeddings(seq)
print(e.shape, np.isnan(e[:,0]).mean(), data.clean_rows(seq).mean())
