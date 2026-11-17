"""IN-SAMPLE (fixed parameters, nothing fitted): what the keep step adds to the appearance step
(margin 0.05, bar 0.90, gap <= 10 s)."""
import sys
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
from reid_lab import data, score
import method

def ev(fn):
    p = score.evaluate({s: {"links": fn(s)} for s in data.DEV_CLIPS})["pooled"]
    return f"right {p['right']:3d} wrong {p['wrong']:2d} lost {p['lost']:3d} | links {p['links']:3d}: right {p['links_right']:3d} wrong {p['links_wrong']:2d} unknown {p['links_unknown']:2d} | impure {p['impure_row_share']:.5f}"

def comb(t, n):
    def f(s):
        k = method.keep_links(s, t, "cos", n)
        return k + method.appearance_links(s, k, 0.05, 0.90, max_gap_s=10.0)
    return f

print("appearance step alone (no plugin join kept)      ", ev(comb(2.0, 1)))
print("keep strict (t 0.89, n_min 15) + appearance      ", ev(comb(0.89, 15)))
print("keep balanced (t 0.85) + appearance              ", ev(comb(0.85, 1)))
print("every plugin join kept + appearance              ", ev(comb(-1.0, 1)))
