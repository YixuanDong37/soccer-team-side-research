"""IN-SAMPLE exploration: keep rule with a minimum piece length (not held-out)."""
import sys
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
from reid_lab import data, score
import method

def ev(fn):
    p = score.evaluate({s: {"links": fn(s)} for s in data.DEV_CLIPS})["pooled"]
    return f"R {p['right']:3d} W {p['wrong']:2d} L {p['lost']:3d} | links {p['links']:3d} r {p['links_right']:3d} w {p['links_wrong']:2d} u {p['links_unknown']:2d} | impure {p['impure_row_share']:.5f}"

for n in (1, 3, 5, 10, 12, 15, 20, 25, 30):
    for t in (0.80, 0.85, 0.86, 0.90, 0.92, 0.95):
        print("n_min", n, "t", t, "keep:", ev(lambda s: method.keep_links(s, t, n_min=n)), " || comb(m .05, bar .9, 10s):",
              ev(lambda s: (lambda k: k + method.appearance_links(s, k, 0.05, 0.90, max_gap_s=10.0))(method.keep_links(s, t, n_min=n))))
