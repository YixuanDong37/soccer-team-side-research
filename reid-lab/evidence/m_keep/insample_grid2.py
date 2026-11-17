"""IN-SAMPLE exploration of the combined rule (not a held-out result)."""
import sys, json, itertools
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
import numpy as np
from reid_lab import data, score
import method

def ev(fn):
    m = {s: {"links": fn(s)} for s in data.DEV_CLIPS}
    p = score.evaluate(m)["pooled"]
    return f"R {p['right']:3d} W {p['wrong']:2d} L {p['lost']:3d} | links {p['links']:3d} r {p['links_right']:3d} w {p['links_wrong']:2d} u {p['links_unknown']:2d} | impure {p['impure_row_share']:.5f}"

t = 0.86
for rounds in (1, 10):
    for gap in (10.0, None):
        for lone in (False, True):
            for margin in (0.0, 0.01, 0.02, 0.03, 0.05):
                for bar in (-1.0, 0.90):
                    print("rounds", rounds, "gap", gap, "lone", lone, "m", margin, "bar", bar, ev(lambda s: method.combined_iter(s, t, margin, bar, rounds=rounds, max_gap_s=gap, lone=lone)))
