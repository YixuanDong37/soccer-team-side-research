"""IN-SAMPLE exploration: quality gate on clean rows (pieces with fewer clean rows are left alone)."""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
from agglo import agglomerate
from core import ctx
from exp2_agglo import show
for kind in ("tc", "w"):
    for min_clean in (0, 3, 5, 10, 20):
        ex = {s: set(int(p) for p, n in zip(ctx(s).P.piece, ctx(s).P.n_clean) if n < min_clean) for s in data.DEV_CLIPS}
        for t_abs, margin in [(0.5, 0.1), (0.5, 0.2), (0.6, 0.2), (0.6, 0.3), (0.7, 0.2), (0.7, 0.3)]:
            m = {s: {"links": agglomerate(s, kind=kind, t_abs=t_abs, margin=margin, mutual=True, exclude=ex[s])} for s in data.DEV_CLIPS}
            show(f"agglo mutual {kind} min_clean={min_clean} t_abs={t_abs} margin={margin}", m)
