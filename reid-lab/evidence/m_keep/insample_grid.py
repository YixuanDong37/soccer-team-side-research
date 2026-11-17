"""IN-SAMPLE exploration (all 13 clips scored together; not a held-out result)."""
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

if "keep" in sys.argv:
    for rule in ("cos", "cos+reach", "cos+team", "imp", "near", "min"):
        grid = (0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96) if rule != "imp" else (-0.5, -0.3, -0.2, -0.12, -0.1, -0.08, -0.05, -0.03, -0.02, -0.01, -0.005)
        for t in grid:
            print(rule, t, ev(lambda s: method.keep_links(s, t, rule)))
if "comb" in sys.argv:
    for t in (0.85, 0.90):
        for source, kw in (("lab", {}), ("groups", {"max_gap_s": 10.0}), ("groups", {"max_gap_s": None}), ("groups", {"max_gap_s": 10.0, "team": "loose"}), ("groups", {"max_gap_s": None, "team": "loose"})):
            for margin in (0.0, 0.02, 0.03, 0.05, 0.08):
                for bar in (-1.0, 0.90, 0.93):
                    print(t, source, kw, "m", margin, "bar", bar, ev(lambda s: method.combined(s, t, margin, bar, source=source, **kw)))
