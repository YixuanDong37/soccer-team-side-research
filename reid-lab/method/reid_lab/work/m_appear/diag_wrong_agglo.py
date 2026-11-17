import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
from agglo import agglomerate
from diag_wrong import explain
t_abs, margin = float(sys.argv[1]), float(sys.argv[2])
kw = dict(mutual=True)
if len(sys.argv) > 3: kw["kind"] = sys.argv[3]
m = {s: {"links": agglomerate(s, t_abs=t_abs, margin=margin, **kw)} for s in data.DEV_CLIPS}
r = score.evaluate(m); b = score.evaluate(score.baseline_links("split"))
print(r["pooled"])
for s, v in r["per_clip"].items(): print(s, v["returns"], v["links"], "impure", v["impure_rows"], "split impure", b["per_clip"][s]["impure_rows"])
explain(m)
