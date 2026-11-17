"""m_appear's method with frozen parameters (label-free): links between gbf pieces from appearance.

    import sys; sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_appear")
    from method import links
    links("SNGS-162")                       # "balanced" operating point
    links("SNGS-162", "cautious", cuts=[[53, 400]])

    python -I method.py POINT OUT.json      # METHOD.json for reid_lab/score.py with the frozen parameters

The parameters below were fitted on all 13 dev clips (the settings loco.py picks when no clip is held
out), so scoring these files on the dev clips gives IN-SAMPLE numbers. The held-out numbers are those of
out/*.method.json, written by loco.py.
"""
import json
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data  # noqa: E402

from agglo import agglomerate  # noqa: E402

POINTS = {
    # name: agglo.agglomerate parameters                        chosen by loco.py on all 13 clips for
    "recall":   {"kind": "w", "t_abs": 0.40, "margin": 0.15},  # lam = 1 (not recommended: wrong identities rise on held-out clips)
    "balanced": {"kind": "w", "t_abs": 0.50, "margin": 0.25},  # lam = 2 and 3
    "cautious": {"kind": "w", "t_abs": 0.50, "margin": 0.30},  # lam = 5 and 10
    "strict":   {"kind": "w", "t_abs": 0.50, "margin": 0.35},  # one grid step beyond cautious (not selected by any lam)
}  # fmt: skip


def links(seq, point="balanced", cuts=()):
    """[[a, b], ...] between pieces of one clip (sub-piece ids of reid_lab.score when cuts are given)."""
    return agglomerate(seq, cuts=cuts, **POINTS[point])


if __name__ == "__main__":
    point, out = sys.argv[1], sys.argv[2]
    method = {"name": f"m_appear frozen '{point}' {POINTS[point]} (in-sample on the dev clips)",
              "clips": {s: {"links": links(s, point)} for s in data.DEV_CLIPS}}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(method, f)
    print("wrote", out)
