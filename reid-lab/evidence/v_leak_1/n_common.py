"""v_leak round 1: helpers shared by the nested analyses (no labels are read here; they work on scored cells).

A "cells" table of a fold maps (variant, margin, clip) -> {"off": scored, "on": scored} exactly as
fuse_r1's fit_loco.py builds it, so fuse_r1's rules.cell / rules.choose can be called on it unchanged.
"""
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse_r1")
from reid_lab import data  # noqa: E402
import rules  # noqa: E402   (fuse_r1)

MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)
POINTS = ("strict", "balanced", "recall")
CANDS = {"r1": [(lv, m) for lv in (False, True) for m in MARGINS], "r0": [(False, m) for m in MARGINS]}
for _v in ("n3", "n10", "c5"):
    CANDS[_v] = CANDS["r1"]


def fmt(p):
    return f"{p['right']} / {p['wrong']} / {p['lost']} | {p['links_right']} / {p['links_wrong']} / {p['links_unknown']} | {100 * p['impure_row_share']:.2f}%"


def short(p):
    return f"{p['right']} / {p['wrong']} / {p['lost']}"


def cand_txt(variant, cand):
    return f"{variant} {cand[1]:.2f} {'+leaves' if cand[0] else 'core'}"


def choose_multi(res, variants, fit_clips, point, rule="B", budget_fn=None, lam=None, trim=None, drop_named=None):
    """fuse_r1's rule applied to candidates of SEVERAL design variants at once: candidate =
    (variant, leaves, margin). Same definitions as rules.choose (feasibility within one variant and
    leaves setting, J / Jtrim, ties: larger margin, then leaves off, then the first variant listed).
    budget_fn(K, scored_fit) overrides the cap; lam overrides the penalty; trim overrides whether the
    worst fit clip is left out of J (default: rule B yes, others as rules.choose); drop_named: leave
    THIS clip out of J instead of the worst one (nothing is left out when it is not a fit clip)."""
    K = rules.CAPS[point]
    L = rules.LAMBDA[point] if lam is None else lam
    table = {}
    for v in variants:
        for cand in CANDS[v]:
            cs = [rules.cell(res, v, cand, s) for s in fit_clips]
            w = np.array([c["wrong"] + c["lw"] + c["ld"] for c in cs], float)
            r = np.array([c["right"] for c in cs], float)
            j_all = float((r - L * w).sum())
            d = int(np.argmax(w))
            if drop_named is not None:
                d = fit_clips.index(drop_named) if drop_named in fit_clips else None
            jt = j_all if d is None else j_all - float(r[d] - L * w[d])
            table[(v, *cand)] = {"right": int(r.sum()), "wrong": int(sum(c["wrong"] for c in cs)),
                                 "scored": int(sum(c["right"] + c["wrong"] + c["lost"] for c in cs)),
                                 "J": j_all, "Jtrim": jt, "dropped": None if d is None else fit_clips[d]}
    scored = next(iter(table.values()))["scored"]
    budget = int(np.floor(K * scored / rules.TOTAL_SCORED)) if budget_fn is None else int(budget_fn(K, scored))
    for (v, lv, m), t in table.items():
        t["feasible"] = all(table[k]["wrong"] <= budget for k in table if k[0] == v and k[1] == lv and k[2] >= m)
    order = {v: -i for i, v in enumerate(variants)}
    tie = lambda c: (c[2], not c[1], order[c[0]])  # noqa: E731
    feas = [c for c in table if table[c]["feasible"]]
    if rule == "R0":
        best = max(table, key=lambda c: (table[c]["J"], *tie(c)))
    elif rule == "Rcap":
        best = (max(feas, key=lambda c: (table[c]["right"], *tie(c))) if feas
                else max(table, key=lambda c: (-table[c]["wrong"], table[c]["right"], *tie(c))))
    else:
        use_trim = (rule == "B") if trim is None else trim
        key = "Jtrim" if use_trim else "J"
        best = (max(feas, key=lambda c: (table[c][key], *tie(c))) if feas
                else max(table, key=lambda c: (-table[c]["wrong"], table[c][key], *tie(c))))
    return best, {"budget": budget, "n_feasible": len(feas), **table[best]}


def pool_cells(cells):
    return rules.pool(cells)
