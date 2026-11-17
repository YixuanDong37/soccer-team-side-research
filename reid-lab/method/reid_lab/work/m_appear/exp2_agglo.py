"""IN-SAMPLE exploration (all 13 clips, labels only through score.evaluate): agglomerative linking variants.
Not a headline result. Usage: python -I exp2_agglo.py"""
import os
import sys
import time

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score  # noqa: E402

from agglo import agglomerate  # noqa: E402

BASE = None


def show(tag, method):
    global BASE
    if BASE is None:
        BASE = score.evaluate(score.baseline_links("split"))["pooled"]
    p = score.evaluate(method)["pooled"]
    print(f"{tag:66s} right {p['right']:3d} wrong {p['wrong']:2d} lost {p['lost']:3d} | links {p['links']:3d} r/w/u {p['links_right']:3d}/{p['links_wrong']:2d}/{p['links_unknown']:3d} "
          f"prec {p['link_precision'] if p['link_precision'] is not None else float('nan'):.3f} impure {p['impure_row_share']:.4f}", flush=True)
    return p


if __name__ == "__main__":
    t0 = time.time()
    grid = [(0.5, 0.2), (0.5, 0.3), (0.6, 0.2), (0.6, 0.3), (0.7, 0.1), (0.7, 0.2), (0.7, 0.3), (0.8, 0.1)]
    # as run during the exploration: team-centred cosine, no mutual-best requirement unless said
    base = dict(kind="tc", mutual=False, covis=1)
    variants = {
        "centroid": dict(),
        "centroid mutual": dict(mutual=True),
        "avg": dict(linkage="avg"),
        "min": dict(linkage="min"),
        "centroid w": dict(kind="w"),
        "centroid min_rows=5": dict(min_rows=5),
        "centroid min_rows=10": dict(min_rows=10),
        "centroid need_rival": dict(need_rival=True),
    }
    for name, kw in variants.items():
        for t_abs, margin in grid:
            m = {s: {"links": agglomerate(s, t_abs=t_abs, margin=margin, **dict(base, **kw))} for s in data.DEV_CLIPS}
            show(f"agglo {name} t_abs={t_abs} margin={margin}", m)
    print(round(time.time() - t0, 1), "s")
