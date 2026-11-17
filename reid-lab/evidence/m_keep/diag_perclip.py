"""DIAGNOSIS: per-clip outcomes of the baselines."""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score
for name in ("split", "plugin", "mutual"):
    r = score.evaluate(score.baseline_links(name))
    print(name, r["pooled"])
    for s, v in r["per_clip"].items():
        print("  ", s, v["returns"], v["links"], v["impure_rows"])
