import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
from agglo import agglomerate
from diag_wrong import explain
for covis in (1, 5):
    m = {s: {"links": agglomerate(s, kind="w", t_abs=0.5, margin=0.3, covis=covis) if s == "SNGS-162" else []} for s in data.DEV_CLIPS}
    r = score.score_clip("SNGS-162", m["SNGS-162"]["links"])
    print("covis", covis, r["returns"], r["links"], r["impure_rows"])
    explain(m)
