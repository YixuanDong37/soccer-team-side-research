"""DIAGNOSIS (labels): wrong returns and wrong links of a METHOD.json; per-clip counts."""
import sys, os, json; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
from diag_wrong import explain
m = json.load(open(sys.argv[1], encoding="utf-8"))
r = score.evaluate(m["clips"]); b = score.evaluate(score.baseline_links("split"))
print(r["pooled"])
for s, v in r["per_clip"].items():
    print(s, v["returns"], v["links"], "impure", v["impure_rows"], "(split", b["per_clip"][s]["impure_rows"], ")", (m.get("chosen_per_held_out_clip") or {}).get(s))
explain(m["clips"])
