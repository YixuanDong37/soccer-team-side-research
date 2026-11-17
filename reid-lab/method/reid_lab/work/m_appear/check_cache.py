"""Check: the current code reproduces the cached grid results (a few settings), and cuts leave untouched clips unchanged."""
import sys, os, json, random; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
from agglo import agglomerate
import loco
grid, results = loco.load("agglo")
random.seed(0); bad = 0
for k in random.sample(list(results), 6) + [json.dumps({"kind": "w", "margin": 0.25, "t_abs": 0.5}, sort_keys=True)]:
    p = json.loads(k)
    for s in data.DEV_CLIPS:
        links = agglomerate(s, kind=p["kind"], t_abs=p["t_abs"], margin=p["margin"])
        if sorted(map(tuple, links)) != sorted(map(tuple, results[k][s]["links"])): bad += 1; print("DIFF", k, s)
print("settings x clips that differ from the cache:", bad)
# cuts: cutting one piece in the middle gives sub-piece ids that score.py accepts
s = "SNGS-162"; cuts = [[53, 400]]
links = agglomerate(s, cuts=cuts)
print("with a cut:", score.score_clip(s, links, cuts)["returns"], "| without:", score.score_clip(s, agglomerate(s))["returns"])
print([l for l in links if max(l) >= 10**7])
