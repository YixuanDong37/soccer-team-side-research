import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
seq = "SNGS-162"
t = data.tracks(seq)
print(t.dtypes)
print(t.head(3).T)
print(len(t), (t.track_id < 0).sum())
r = data.refine(seq)
print(r.keys())
for k, v in r.items():
    if isinstance(v, dict): print(k, list(v.keys())[:30])
    elif isinstance(v, list): print(k, "list", len(v))
    else: print(k, v)
print(json.dumps(r["teams"]["tracks"][:2], indent=1)[:1500])
print(json.dumps(r["tracks"], indent=1)[:2500])
a = data.appearance(seq)
print(a.keys())
print(a["params"], a["chosen"], a["real_candidate_pairs"], len(a["candidate_pairs"]), a["pieces"], a["eligible_pieces"])
print(data.pieces(seq).head(20))
c = data.calib(seq); print(c.dtypes); print(c.head(2).T)
ev = json.loads((data.RUNS/seq/"reentry-eval.json").read_text())
print(ev.keys())
print(json.dumps(ev, indent=1)[:3000])
