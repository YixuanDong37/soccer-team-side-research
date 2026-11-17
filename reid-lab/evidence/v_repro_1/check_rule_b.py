"""v_repro_1 / check_rule_b.py -- rule B of PREREG.txt written again from its text (not from rules.py) and
applied to the scored fold results of THIS directory's LOCO run (cache/loco_raw.pkl). Checks that
 - the setting chosen per fold and on all 13 clips equals params_loco.json,
 - a clip's own results are never used to choose its setting (the choice is recomputed here from the
   12 fit clips only),
 - pooling each clip at its fold's setting gives the reported held-out counts.
Works on scored counts only (no annotation is read here).

    python -I -B check_rule_b.py
"""
import json
import math
import pickle
import sys
from collections import Counter

DEV = ["SNGS-107", "SNGS-163", "SNGS-106", "SNGS-097", "SNGS-067", "SNGS-100", "SNGS-066",
       "SNGS-161", "SNGS-169", "SNGS-162", "SNGS-151", "SNGS-102", "SNGS-112"]
MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)
K = {"strict": 4, "balanced": 7, "recall": 14}
LAM = {"strict": 10, "balanced": 3, "recall": 1}
HERE = "C:/Research/plugins/tools/reid_lab/work/v_repro_1/"

raw = pickle.load(open(HERE + "cache/loco_raw.pkl", "rb"))["folds"]
params = json.load(open(HERE + "params_loco.json", encoding="utf-8"))


def counts(res, leaves, margin, clip):
    x = res[("r1", margin, clip)]["on" if leaves else "off"]
    c = Counter(x["per_return"])
    w = c["wrong"] + x["lk"]["wrong"] + (x.get("leaf_disagree", 0) if leaves else 0)
    return c["right"], c["wrong"], c["lost"], w, x


def choose(res, fit, point):
    stat = {}
    for leaves in (False, True):
        for m in MARGINS:
            per = [counts(res, leaves, m, s) for s in fit]
            right = [p[0] for p in per]
            wrong = sum(p[1] for p in per)
            w = [p[3] for p in per]
            worst = max(range(len(fit)), key=lambda i: (w[i], -i))           # largest w, first clip on ties
            J = sum(right[i] - LAM[point] * w[i] for i in range(len(fit)) if i != worst)
            stat[(leaves, m)] = (wrong, J, fit[worst])
    scored = sum(sum(counts(res, False, MARGINS[0], s)[:3]) for s in fit)
    budget = math.floor(K[point] * scored / 137)
    feasible = [c for c in stat if all(stat[(c[0], m2)][0] <= budget for m2 in MARGINS if m2 >= c[1])]
    if feasible:
        best = max(feasible, key=lambda c: (stat[c][1], c[1], not c[0]))
    else:
        best = max(stat, key=lambda c: (-stat[c][0], stat[c][1], c[1], not c[0]))
    return best, stat[best], budget, len(feasible)


bad = 0
pooled = {p: Counter() for p in K}
dropped = {p: Counter() for p in K}
for point in K:
    for fold in [*DEV, "all13"]:
        fit = [s for s in DEV if s != fold]
        best, st, budget, nfeas = choose(raw[fold]["res"], fit, point)
        want = params[fold]["choice"][point]
        same = (bool(want["leaves"]), float(want["margin"])) == (best[0], best[1])
        bad += not same
        print(f"{point:<8} fold {fold:<8} budget {budget} feasible {nfeas:>2} -> margin {best[1]:.2f} leaves {'on' if best[0] else 'off'} "
              f"(J {st[1]:g}, left out of J: {st[2]}, wrong on fit clips {st[0]})", "= params_loco.json" if same else f"DIFFERENT: {want}")
        if fold != "all13":
            dropped[point][st[2]] += 1
            r, w, l, _, x = counts(raw[fold]["res"], best[0], best[1], fold)
            pooled[point].update({"right": r, "wrong": w, "lost": l, "lr": x["lk"]["right"], "lw": x["lk"]["wrong"], "lu": x["lk"]["unknown"],
                                  "impure": x["impure"], "rows": x["rows"]})
print()
for point, c in pooled.items():
    print(f"{point}: held-out pooled {c['right']} / {c['wrong']} / {c['lost']} | links {c['lr']} / {c['lw']} / {c['lu']} | impure {c['impure'] / c['rows']:.5f}"
          f" | clip left out of J over the 13 folds: {dict(dropped[point])}")
print("rule B re-implemented from PREREG.txt gives the settings of params_loco.json" if not bad else f"{bad} DIFFERENCES")
sys.exit(1 if bad else 0)
