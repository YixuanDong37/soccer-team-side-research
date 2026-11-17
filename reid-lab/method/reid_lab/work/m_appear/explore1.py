import sys, time; sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score
import numpy as np, pandas as pd, json
t0=time.time()
for name in ("split","plugin","mutual"):
    r = score.evaluate(score.baseline_links(name))
    print(name, r["pooled"])
print(time.time()-t0)
# per clip overview
for seq in data.DEV_CLIPS:
    p = data.pieces(seq); c = score.clip(seq)
    a = data.appearance(seq)
    rets = c.returns
    gaps = [r["gap"] for r in rets]
    print(seq, "pieces", len(p), "k>0", int((p.k>0).sum()), "returns", len(rets), "scored", sum(r["a"] is not None and r["b"] is not None for r in rets),
          "gap>250:", sum(g>250 for g in gaps), "cand", len(a["candidate_pairs"]), "elig", a["eligible_pieces"], "team nan", int(p.team.isna().sum()), "roles", p.role.value_counts().to_dict())
