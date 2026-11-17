"""fuse_r1 / rules.py -- how an operating point's setting is chosen on the FIT clips of a fold.

Pure functions on already scored results (no labels are read here; the counts they work on come
from reid_lab.score on the fit clips, computed in fit_loco.py).

A candidate setting = (leaves on / off, margin). Rule B is the rule of PREREG.txt:
 (i)  hard constraint: wrong returns on the fit clips <= floor(K * scored_fit / 137) at the candidate's
      margin and at every stricter margin with the same leaves setting;
 (ii) among those the largest J = sum over the fit clips except d* of [right - lam * w], with
      w = wrong returns + links judged wrong + leaf links with different weak-label persons, and
      d* = the fit clip with the largest w (first in DEV_CLIPS order on ties);
 (iii) ties: the larger margin, then leaves off.
 No candidate meets (i): fewest wrong returns, then (ii), then (iii).
Alternative rules, reported next to it (never used to build the METHOD files):
 "R0"       round 0's rule: largest untrimmed J, no cap
 "Rcap"     cap only: most right among the candidates that meet (i)
 "Rnotrim"  rule B without leaving a clip out
"""
from __future__ import annotations

from collections import Counter

import numpy as np

CAPS = {"strict": 4, "balanced": 7, "recall": 14}
LAMBDA = {"strict": 10.0, "balanced": 3.0, "recall": 1.0}
RULES = ("B", "R0", "Rcap", "Rnotrim")
TOTAL_SCORED = 137.0


def cell(res: dict, variant: str, cand, clip: str) -> dict:
    """Counts of one (variant, candidate, clip) result."""
    leaves, m = cand
    r = res[(variant, m, clip)]
    x = r["on"] if leaves else r["off"]
    c = Counter(x["per_return"])
    return {"right": c["right"], "wrong": c["wrong"], "lost": c["lost"],
            "lr": x["lk"]["right"], "lw": x["lk"]["wrong"], "lu": x["lk"]["unknown"],
            "ld": x.get("leaf_disagree", 0), "impure": x["impure"], "rows": x["rows"]}


def pool(cells) -> dict:
    cells = list(cells)
    s = lambda k: int(sum(c[k] for c in cells))  # noqa: E731
    rows = s("rows")
    return {"right": s("right"), "wrong": s("wrong"), "lost": s("lost"), "links_right": s("lr"), "links_wrong": s("lw"),
            "links_unknown": s("lu"), "leaf_disagree": s("ld"), "impure_rows": s("impure"),
            "impure_row_share": round(s("impure") / rows, 5) if rows else None}


def choose(res: dict, variant: str, cands, fit_clips, point: str, rule: str = "B", caps=CAPS, lam=LAMBDA):
    """The candidate chosen for `point` on the fit clips, and a small record of why."""
    K, L = caps[point], lam[point]
    table = {}
    for cand in cands:
        cs = [cell(res, variant, cand, s) for s in fit_clips]
        w = np.array([c["wrong"] + c["lw"] + c["ld"] for c in cs], float)
        r = np.array([c["right"] for c in cs], float)
        j_all = float((r - L * w).sum())
        d = int(np.argmax(w))                      # first clip in order on ties
        j_trim = j_all - float(r[d] - L * w[d])
        table[cand] = {"right": int(r.sum()), "wrong": int(sum(c["wrong"] for c in cs)),
                       "scored": int(sum(c["right"] + c["wrong"] + c["lost"] for c in cs)),
                       "J": j_all, "Jtrim": j_trim, "dropped": fit_clips[d]}
    scored = next(iter(table.values()))["scored"]
    budget = int(np.floor(K * scored / TOTAL_SCORED))
    for (lv, m), t in table.items():
        t["feasible"] = all(table[(lv2, m2)]["wrong"] <= budget for (lv2, m2) in table if lv2 == lv and m2 >= m)
    tie = lambda cand: (cand[1], not cand[0])     # larger margin, then leaves off  # noqa: E731
    feas = [c for c in table if table[c]["feasible"]]
    if rule == "R0":
        best = max(table, key=lambda c: (table[c]["J"], *tie(c)))
    elif rule == "Rcap":
        best = (max(feas, key=lambda c: (table[c]["right"], *tie(c))) if feas
                else max(table, key=lambda c: (-table[c]["wrong"], table[c]["right"], *tie(c))))
    else:
        key = "Jtrim" if rule == "B" else "J"
        best = (max(feas, key=lambda c: (table[c][key], *tie(c))) if feas
                else max(table, key=lambda c: (-table[c]["wrong"], table[c][key], *tie(c))))
    return best, {"budget": budget, "n_feasible": len(feas), **table[best]}


def bootstrap(per_clip: dict, n: int = 5000, seed: int = 0) -> dict:
    """Clip bootstrap of the pooled held-out right / wrong shares (the clips are resampled)."""
    rng = np.random.default_rng(seed)
    clips = list(per_clip)
    R = np.array([per_clip[s]["right"] for s in clips], float)
    Wr = np.array([per_clip[s]["wrong"] for s in clips], float)
    N = np.array([per_clip[s]["right"] + per_clip[s]["wrong"] + per_clip[s]["lost"] for s in clips], float)
    idx = rng.integers(0, len(clips), size=(n, len(clips)))
    rs, ws = R[idx].sum(axis=1) / N[idx].sum(axis=1), Wr[idx].sum(axis=1) / N[idx].sum(axis=1)
    q = lambda v: [round(float(x), 3) for x in np.quantile(v, [0.05, 0.5, 0.95])]  # noqa: E731
    out = {"right_share_q05_q50_q95": q(rs), "wrong_share_q05_q50_q95": q(ws)}
    for cap in (4, 7, 14):
        out[f"P_wrong_share_le_{cap}_of_137"] = round(float((ws <= cap / TOTAL_SCORED + 1e-12).mean()), 3)
    return out
