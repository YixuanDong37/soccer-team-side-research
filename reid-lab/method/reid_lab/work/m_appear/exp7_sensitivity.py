"""IN-SAMPLE sensitivity of the frozen 'balanced' point (w / 0.5 / 0.25) to the two constants that were not tuned:
the WCCN shrinkage (0.3) and the minimum of clean rows (5). Not used to choose anything."""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
from agglo import agglomerate
from features import Protos
from exp2_agglo import show
for lam in (0.03, 0.1, 0.3, 1.0, 3.0):
    show(f"WCCN lambda={lam}", {s: {"links": agglomerate(s, pr=Protos(s, lam=lam))} for s in data.DEV_CLIPS})
for mc in (1, 3, 5, 10, 20):
    show(f"min clean rows={mc}", {s: {"links": agglomerate(s, pr=Protos(s, min_clean=mc))} for s in data.DEV_CLIPS})
