"""v_robust_1 / r10_rules.py -- (works on fuse_r1's already scored LOCO results, cache/loco_raw.pkl;
no annotation is read here) integrity of the choice of the operating points and how much the
pooled held-out result depends on the rule's own numbers.

    python -I -B r10_rules.py     -> out/r10_rules.md, out/r10_rules.json

1. Re-derive every fold's setting with fuse_r1's rules.choose (rule B) from the raw results and
   compare with params_loco.json; re-pool the held-out cells and compare with the METHOD scores.
2. Rule B with its own numbers moved: lam x 0.9 / x 1.1 and wider; cap K - 1 / K + 1; the budget
   without the floor(K x scored_fit / 137) scaling; leaving out 0 / 1 / 2 fit clips in J; the error
   count w without the link terms; tie-break towards the smaller margin.
3. Which fit clip is left out of J in each fold, and what the strict point would be if the clip
   that is left out were NOT allowed to be SNGS-162.
4. Parameter transfer: every clip at the frozen settings under each of the 14 sets of numbers
   (13 folds + all13): is the held-out fold worse than the 12 folds that had the clip in their fit?
"""
import json
import pickle
import sys
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import numpy as np

import vr_common as vc
from vr_common import data

sys.path.insert(0, str(vc.FUSE))
import rules  # noqa: E402  fuse_r1's rule module (pure functions on scored counts)

assert str(vc.FUSE).replace("\\", "/").lower() in rules.__file__.replace("\\", "/").lower()
RAW = pickle.load(open(vc.FUSE_CACHE / "loco_raw.pkl", "rb"))["folds"]
MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)
CANDS = [(lv, m) for lv in (False, True) for m in MARGINS]
POINTS = ("strict", "balanced", "recall")
CLIPS = list(data.DEV_CLIPS)
ALL = "all13"
P = vc.params()
R1 = json.loads((vc.OUT / "r1_reproduce.json").read_text(encoding="utf-8"))


def cell(fold, cand, clip):
    return rules.cell(RAW[fold]["res"], "r1", cand, clip)


def my_choose(fold, point, K, lam, trim=1, scale_budget=True, w_terms=("wrong", "lw", "ld"), tie_small=False, forbid_drop=()):
    """Rule B re-implemented with its numbers as arguments (trim = how many fit clips with the largest w
    are left out of J; forbid_drop: clips that may not be left out)."""
    fit = [s for s in CLIPS if s != fold]
    table = {}
    for cand in CANDS:
        cs = [cell(fold, cand, s) for s in fit]
        w = np.array([sum(c[k] for k in w_terms) for c in cs], float)
        r = np.array([c["right"] for c in cs], float)
        j = r - lam * w
        order = sorted(range(len(fit)), key=lambda i: (-w[i], i))          # largest w first, first clip on ties
        order = [i for i in order if fit[i] not in forbid_drop]
        drop = order[:trim]
        table[cand] = {"wrong": int(sum(c["wrong"] for c in cs)), "right": int(r.sum()), "scored": int(sum(c["right"] + c["wrong"] + c["lost"] for c in cs)),
                       "J": float(j.sum() - j[drop].sum()), "dropped": [fit[i] for i in drop]}
    scored = next(iter(table.values()))["scored"]
    budget = int(np.floor(K * scored / 137.0)) if scale_budget else K
    for (lv, m), t in table.items():
        t["feasible"] = all(table[(lv2, m2)]["wrong"] <= budget for (lv2, m2) in table if lv2 == lv and m2 >= m)
    tie = (lambda c: (-c[1], not c[0])) if tie_small else (lambda c: (c[1], not c[0]))
    feas = [c for c in table if table[c]["feasible"]]
    best = (max(feas, key=lambda c: (table[c]["J"], *tie(c))) if feas else max(table, key=lambda c: (-table[c]["wrong"], table[c]["J"], *tie(c))))
    return best, table[best]


def pooled(chosen):
    cells = [cell(c, chosen[c], c) for c in CLIPS]
    p = rules.pool(cells)
    c162 = cells[CLIPS.index("SNGS-162")]
    return p, f"{c162['right']} / {c162['wrong']} / {c162['lost']}"


def txt(cand):
    return f"{cand[1]:.2f} {'+leaves' if cand[0] else 'core'}"


def row(name, point, **kw):
    K = kw.pop("K", rules.CAPS[point])
    lam = kw.pop("lam", rules.LAMBDA[point])
    chosen = {c: my_choose(c, point, K, lam, **kw)[0] for c in CLIPS}
    allc = my_choose(ALL, point, K, lam, **kw)[0]
    p, c162 = pooled(chosen)
    cnt = Counter(txt(v) for v in chosen.values())
    J.setdefault("variants", {}).setdefault(point, {})[name] = {"heldout": p, "chosen": {c: txt(v) for c, v in chosen.items()}, "all13": txt(allc),
                                                                 "folds_equal_all13": int(sum(v == allc for v in chosen.values()))}
    return (f"| {name} | {p['right']} / {p['wrong']} / {p['lost']} | {p['links_right']} / {p['links_wrong']} / {p['links_unknown']} | {100 * p['impure_row_share']:.2f}% | "
            f"{c162} | {', '.join(f'{n}x {k}' for k, n in sorted(cnt.items()))} | {txt(allc)} | {sum(v == allc for v in chosen.values())} / 13 | "
            f"{'OVER' if p['wrong'] > rules.CAPS[point] else 'ok'} |")


J = {}
L = ["# r10: the choice of the operating points (rule B) and how much it depends on the rule's own numbers", "",
     "Works on fuse_r1's scored LOCO results (cache/loco_raw.pkl). Nothing is re-linked.", ""]
# ---------------------------------------------------------------- 1. integrity
L += ["## 1. Integrity", ""]
bad = []
for point in POINTS:
    for c in [*CLIPS, ALL]:
        fit = [s for s in CLIPS if s != c]
        got, _ = rules.choose(RAW[c]["res"], "r1", CANDS, fit, point, "B")
        mine, _ = my_choose(c, point, rules.CAPS[point], rules.LAMBDA[point])
        want = (P[c]["choice"][point]["leaves"], P[c]["choice"][point]["margin"])
        if got != want or mine != want:
            bad.append((point, c, got, mine, want))
    chosen = {c: (P[c]["choice"][point]["leaves"], P[c]["choice"][point]["margin"]) for c in CLIPS}
    p, _ = pooled(chosen)
    m = R1["methods"][point]["pooled"]
    same = all(p[k] == m[k] for k in ("right", "wrong", "lost", "links_right", "links_wrong", "links_unknown")) and abs(p["impure_row_share"] - m["impure_row_share"]) < 2e-5
    L.append(f"- {point}: pooled held-out cells of loco_raw {p['right']} / {p['wrong']} / {p['lost']} | {p['links_right']} / {p['links_wrong']} / {p['links_unknown']} "
             f"= score of METHOD_{point}.json: {same}")
    J.setdefault("integrity", {})[point] = same
L.append(f"- settings re-derived with fuse_r1's rules.choose and with my own re-implementation for 3 points x 14 folds: {42 - len(bad)} of 42 equal params_loco.json"
         + (f"; differences: {bad}" if bad else ""))
J["integrity"]["choice_mismatches"] = [str(b) for b in bad]
budgets = {point: sorted({int(np.floor(rules.CAPS[point] * sum(sum(cell(c, (False, 0.6), s)[k] for k in ("right", "wrong", "lost")) for s in CLIPS if s != c) / 137.0)) for c in CLIPS})
           for point in POINTS}
L += [f"- the wrong-return budget of constraint (i) on the 12 fit clips, floor(K x scored_fit / 137), over the 13 folds: {budgets} (K = 4 / 7 / 14)", ""]
J["budgets"] = budgets

# ---------------------------------------------------------------- 2. the rule's numbers moved
L += ["## 2. Rule B with its own numbers moved (held-out, pooled; nothing re-linked)", "",
      "| variant | right / wrong / lost | links r / w / u | impure | SNGS-162 | settings chosen by the folds | all 13 | folds = all 13 | limit |",
      "|---|---|---|---|---|---|---|---|---|"]
for point in POINTS:
    lam0, K0 = rules.LAMBDA[point], rules.CAPS[point]
    L.append(f"| **{point}** (K {K0}, lam {lam0:g}) | | | | | | | | |")
    L.append(row("rule B as fixed", point))
    L.append(row(f"lam x 0.9 = {0.9 * lam0:g}", point, lam=0.9 * lam0))
    L.append(row(f"lam x 1.1 = {1.1 * lam0:g}", point, lam=1.1 * lam0))
    for f in (0.5, 0.7, 1.5, 2.0):
        L.append(row(f"lam x {f:g} = {f * lam0:g}", point, lam=f * lam0))
    L.append(row(f"cap K - 1 = {K0 - 1}", point, K=K0 - 1))
    L.append(row(f"cap K + 1 = {K0 + 1}", point, K=K0 + 1))
    L.append(row("budget not scaled to the fit clips (K itself)", point, scale_budget=False))
    L.append(row("no fit clip left out of J", point, trim=0))
    L.append(row("two fit clips left out of J", point, trim=2))
    L.append(row("w = wrong returns only (no link terms)", point, w_terms=("wrong",)))
    L.append(row("w = wrong returns + links judged wrong (no weak-label leaf term)", point, w_terms=("wrong", "lw")))
    L.append(row("ties towards the smaller margin", point, tie_small=True))
    L.append(row("SNGS-162 may not be the clip left out", point, forbid_drop=("SNGS-162",)))
L.append("")

# ---------------------------------------------------------------- 3. the clip left out
L += ["## 3. Which fit clip rule B leaves out of J at the chosen setting", ""]
for point in POINTS:
    d = Counter()
    for c in CLIPS:
        _, t = my_choose(c, point, rules.CAPS[point], rules.LAMBDA[point])
        d[t["dropped"][0]] += 1
    L.append(f"- {point}: " + ", ".join(f"{k} in {v} folds" for k, v in d.most_common()))
    J.setdefault("dropped", {})[point] = dict(d)
L.append("")

# ---------------------------------------------------------------- 4. parameter transfer
L += ["## 4. Parameter transfer at the frozen settings: a clip under the numbers of its own fold (held-out) against the 12 folds that had it in their fit", "",
      "| setting | held-out pooled right / wrong | mean over the 12 in-fit folds, summed over clips: right / wrong | held-out minus in-fit: right / wrong | all-13 numbers (in-sample) right / wrong | "
      "pooled over all 13 clips under each single fold's numbers: right min-max, wrong min-max |", "|---|---|---|---|---|---|"]
for point in POINTS:
    ch = P[ALL]["choice"][point]
    cand = (ch["leaves"], ch["margin"])
    held = [cell(c, cand, c) for c in CLIPS]
    infit_r = sum(np.mean([cell(f, cand, c)["right"] for f in CLIPS if f != c]) for c in CLIPS)
    infit_w = sum(np.mean([cell(f, cand, c)["wrong"] for f in CLIPS if f != c]) for c in CLIPS)
    hr, hw = sum(x["right"] for x in held), sum(x["wrong"] for x in held)
    ar, aw = sum(cell(ALL, cand, c)["right"] for c in CLIPS), sum(cell(ALL, cand, c)["wrong"] for c in CLIPS)
    per_fold = [(sum(cell(f, cand, c)["right"] for c in CLIPS), sum(cell(f, cand, c)["wrong"] for c in CLIPS)) for f in CLIPS]
    J.setdefault("transfer", {})[point] = {"setting": txt(cand), "heldout": [hr, hw], "infit_mean": [round(float(infit_r), 2), round(float(infit_w), 2)],
                                           "all13": [ar, aw], "per_fold_numbers": per_fold}
    L.append(f"| {point}: {txt(cand)} | {hr} / {hw} | {infit_r:.1f} / {infit_w:.1f} | {hr - infit_r:+.1f} / {hw - infit_w:+.1f} | {ar} / {aw} | "
             f"{min(x[0] for x in per_fold)}-{max(x[0] for x in per_fold)}, {min(x[1] for x in per_fold)}-{max(x[1] for x in per_fold)} |")
L += ["", "Pair-model weights over the 13 folds (cos_w, sink_lo, plug, bias) and the bonus per unit of sink_lo / plug in cosine units:", ""]
W = np.array([[*P[c]["pair_model"]["w"], P[c]["pair_model"]["b"]] for c in CLIPS])
L.append(f"- cos_w {W[:, 0].min():.2f} to {W[:, 0].max():.2f}; sink_lo {W[:, 1].min():.3f} to {W[:, 1].max():.3f}; plug {W[:, 2].min():.2f} to {W[:, 2].max():.2f}; "
         f"bias {W[:, 3].min():.2f} to {W[:, 3].max():.2f}; all13: {np.round([*P[ALL]['pair_model']['w'], P[ALL]['pair_model']['b']], 3).tolist()}")
L.append(f"- sink_lo / cos_w: {(W[:, 1] / W[:, 0]).min():.4f} to {(W[:, 1] / W[:, 0]).max():.4f} (largest possible position bonus = ratio x {np.log(0.98 / 0.02) + np.log(0.999 / 0.001):.2f}: "
         f"{(W[:, 1] / W[:, 0]).min() * (np.log(0.98 / 0.02) + np.log(0.999 / 0.001)):.2f} to {(W[:, 1] / W[:, 0]).max() * (np.log(0.98 / 0.02) + np.log(0.999 / 0.001)):.2f}); "
         f"plug / cos_w: {(W[:, 2] / W[:, 0]).min():.3f} to {(W[:, 2] / W[:, 0]).max():.3f}")
J["pair_model_ranges"] = {"cos_w": [float(W[:, 0].min()), float(W[:, 0].max())], "sink_lo": [float(W[:, 1].min()), float(W[:, 1].max())],
                          "plug": [float(W[:, 2].min()), float(W[:, 2].max())]}
(vc.OUT / "r10_rules.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(vc.OUT / "r10_rules.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
print("\n".join(L))
