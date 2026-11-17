"""IN-SAMPLE exploration: rival = shares >= covis frames; allow links between groups sharing <= max_shared frames."""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
from agglo import agglomerate
from exp2_agglo import show
for kind in ("tc", "w"):
    for covis, max_shared in ((1, 0), (5, 0), (10, 0), (5, 3)):
        for t_abs, margin in [(0.5, 0.1), (0.5, 0.2), (0.6, 0.2), (0.6, 0.3), (0.7, 0.2), (0.7, 0.3)]:
            m = {s: {"links": agglomerate(s, kind=kind, t_abs=t_abs, margin=margin, covis=covis, max_shared=max_shared)} for s in data.DEV_CLIPS}
            show(f"agglo {kind} covis={covis} max_shared={max_shared} t_abs={t_abs} margin={margin}", m)
