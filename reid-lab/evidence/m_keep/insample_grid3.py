"""IN-SAMPLE exploration: what does the keep step add to the combination? (not held-out)"""
import sys
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
from reid_lab import data, score
import method

def ev(fn):
    m = {s: {"links": fn(s)} for s in data.DEV_CLIPS}
    p = score.evaluate(m)["pooled"]
    return f"R {p['right']:3d} W {p['wrong']:2d} L {p['lost']:3d} | links {p['links']:3d} r {p['links_right']:3d} w {p['links_wrong']:2d} u {p['links_unknown']:2d} | impure {p['impure_row_share']:.5f}"

for gap in (10.0, None):
    for margin, bar in ((0.03, 0.9), (0.05, -1.0), (0.02, 0.9)):
        for t in (2.0, 0.95, 0.92, 0.90, 0.88, 0.86, 0.85, 0.82, 0.80, 0.75, -1.0):
            print("gap", gap, "m", margin, "bar", bar, "t_keep", t, ev(lambda s: method.combined_iter(s, t, margin, bar, rounds=1, max_gap_s=gap)))
