"""DIAGNOSIS / IN-SAMPLE: what would the plugin's own re-joins add to the appearance method?"""
import sys, os, json; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
import numpy as np, pandas as pd
from collections import Counter
from agglo import agglomerate
from features import pair_features
from exp2_agglo import show
mine = {s: {"links": agglomerate(s)} for s in data.DEV_CLIPS}   # w, 0.5, 0.25
plug = score.baseline_links("plugin")
cnt = Counter()
for s in data.DEV_CLIPS:
    a = score.score_clip(s, mine[s]["links"])["per_return"]; b = score.score_clip(s, plug[s]["links"])["per_return"]
    cnt.update(zip(a, b))
print("(mine, plugin) outcomes per return:", dict(cnt))
show("mine (w, 0.5, 0.25)", mine)
show("mine + all plugin re-joins", {s: {"links": mine[s]["links"] + plug[s]["links"]} for s in data.DEV_CLIPS})
# plugin re-joins vetted by appearance: keep a plugin join (a, b) when the pair's cosine / rival margins clear a lower bar
for kind in ("w",):
    for t in (0.3, 0.4, 0.5):
        for m in (-1.0, 0.0, 0.1, 0.15):
            out = {}
            for s in data.DEV_CLIPS:
                F = pair_features(s).set_index(["a", "b"]); extra = []
                for a, b in plug[s]["links"]:
                    if (a, b) not in F.index: continue
                    r = F.loc[(a, b)]
                    ua, ub = r[f"urival_a_{kind}"], r[f"urival_b_{kind}"]
                    if r[f"cos_{kind}"] >= t and (np.isnan(ua) or ua >= m) and (np.isnan(ub) or ub >= m): extra.append([a, b])
                out[s] = {"links": mine[s]["links"] + extra}
            show(f"mine + plugin joins with cos_{kind}>={t}, rival margin>={m}", out)
