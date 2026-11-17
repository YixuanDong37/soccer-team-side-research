"""v_leak round 1: how much of fuse_r1's held-out result rests on choices that were made with the pooled
held-out results of these 13 clips in view (the rule for the operating points, the tiers, the 5-box
boundary, "a rival shares >= 1 frame")?  Each choice is repeated INSIDE the folds.

Works on scored cells only (labels were used to score; the held-out clip's cell of a fold is read only
after the choice for that fold has been made on its 12 fit clips).

    python -I -B n1_nested.py blind     # cells of the blindfolded re-run (out/blind_folds.pkl, from s4_compare.py)   -> out/nested.md, out/nested.json
    python -I -B n1_nested.py fuse_r1   # cells of fuse_r1/cache/loco_raw.pkl (variants r1 and r0 only)               -> out/nested_fuse_r1.md

Sections
 A  the budget constant: rule B's cap floor(K * scored_fit / 137) uses 137, the scored returns of all 13
    clips (held-out clip included). Alternatives that do not: K * 12 / 13, K - 1, K.
 B  the rule family, each rule held-out (fuse_r1 reports these) and the rule chosen INSIDE each fold by
    an inner leave-one-clip-out over the 12 fit clips (same cells: the fold's weights are in-sample for
    the inner clips, only the choice of the setting is nested).
 C  design variants offered to rule B inside the fold next to the method: r0 (no tiers), n3 / n10 (solid
    from 3 / 10 boxes), c5 (rival shares >= 5 frames).
 D  penalty lam and the trimming of rule B, one at a time (how wide is the family the rule was taken from).
 E  rule B against the same rule with the left-out clip named (SNGS-162).
 F  fixed settings: the 12 other clips against SNGS-162 alone, per design variant.
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path("C:/Research/plugins/tools/reid_lab/work/v_leak_1")
sys.path.insert(0, str(HERE))
from n_common import CANDS, POINTS, cand_txt, choose_multi, data, rules, short  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "blind"
if src == "blind":
    folds = pickle.load(open(HERE / "out/blind_folds.pkl", "rb"))
    VARS = ("r1", "r0", "n3", "n10", "c5")
    tag = ""
else:
    folds = pickle.load(open("C:/Research/plugins/tools/reid_lab/work/fuse_r1/cache/loco_raw.pkl", "rb"))["folds"]
    VARS = ("r1", "r0")
    tag = "_fuse_r1"
CLIPS = list(data.DEV_CLIPS)
fit_of = {c: [s for s in CLIPS if s != c] for c in CLIPS}
OUT = {}
L = [f"# v_leak round 1: choices repeated inside the folds (cells: {'blindfolded re-run' if src == 'blind' else 'fuse_r1/cache/loco_raw.pkl'})", "",
     "Cells: right / wrong / lost of the 137 scored returns, pooled over the 13 held-out clips. Limits: strict 4, balanced 7, recall 14 wrong.", ""]


def heldout(choice):
    """choice: {clip: (variant, leaves, margin)} -> pooled held-out result and per-clip cells."""
    cells = {c: rules.cell(folds[c]["res"], choice[c][0], (choice[c][1], choice[c][2]), c) for c in CLIPS}
    return rules.pool(cells.values()), cells


def describe(choice):
    cnt = Counter(cand_txt(v[0], (v[1], v[2])) for v in choice.values())
    return ", ".join(f"{n}x {k}" for k, n in sorted(cnt.items(), key=lambda kv: -kv[1]))


def run(point, variants=("r1",), rule="B", **kw):
    choice = {c: choose_multi(folds[c]["res"], variants, fit_of[c], point, rule, **kw)[0] for c in CLIPS}
    p, cells = heldout(choice)
    return p, choice, cells


# ---------------------------------------------------------------- check: choose_multi with one variant is fuse_r1's rules.choose
n_checked = n_diff = 0
for v in ("r1", "r0"):
    for rule in rules.RULES:
        for point in POINTS:
            for c in CLIPS:
                mine, _ = choose_multi(folds[c]["res"], (v,), fit_of[c], point, rule)
                theirs, _ = rules.choose(folds[c]["res"], v, CANDS[v], fit_of[c], point, rule)
                n_checked += 1
                n_diff += (mine[1], mine[2]) != tuple(theirs)
L += [f"Check: n_common.choose_multi with a single variant against fuse_r1's rules.choose (2 linkers x 4 rules x 3 points x 13 folds = {n_checked} choices): {n_diff} differ.", ""]
if n_diff:
    raise SystemExit("choose_multi does not reproduce rules.choose")

# ---------------------------------------------------------------- A: the budget constant
L += ["## A. The cap of rule B uses the number of scored returns of all 13 clips (137)", "",
      "floor(K x scored_fit / 137) = floor(K x (1 - scored_heldout / 137)): the cap of a fold depends on how many scored returns its held-out clip has.",
      "Caps per fold under fuse_r1's formula, and the held-out result with caps that do not use the held-out clip:", "",
      "| cap | strict | balanced | recall |", "|---|---|---|---|"]
budgets = {"fuse_r1: floor(K x scored_fit / 137)": None, "floor(K x 12 / 13)": lambda K, s: K * 12 // 13, "K - 1": lambda K, s: K - 1, "K": lambda K, s: K,
           "no cap at all (part (i) of rule B removed)": lambda K, s: 10**9}
OUT["A"] = {}
for name, fn in budgets.items():
    row = []
    for point in POINTS:
        p, choice, _ = run(point, budget_fn=fn)
        caps = Counter(choose_multi(folds[c]["res"], ("r1",), fit_of[c], point, "B", budget_fn=fn)[1]["budget"] for c in CLIPS)
        base = {c: choose_multi(folds[c]["res"], ("r1",), fit_of[c], point, "B")[0] for c in CLIPS}
        same = sum(choice[c] == base[c] for c in CLIPS)
        capt = "none" if max(caps) > 10**6 else str(dict(sorted(caps.items())))
        row.append(f"{short(p)} (caps {capt}; {same} of 13 folds choose the same setting as rule B)")
        OUT["A"].setdefault(name, {})[point] = {"pooled": p, "choice": {c: list(v) for c, v in choice.items()}}
    L.append(f"| {name} | " + " | ".join(row) + " |")
L.append("")

# ---------------------------------------------------------------- B: the rule, nested
L += ["## B. The rule for the operating points", "",
      "fuse_r1 fixed rule B in PREREG.txt knowing round 0's held-out tables (there 'J without the worst fit clip' had given 88 / 4 / 45 post hoc).",
      "Held-out result of each rule of the family PREREG.txt names, and of the rule CHOSEN INSIDE each fold:", "",
      "| rule | strict | balanced | recall |", "|---|---|---|---|"]
OUT["B"] = {}
for rule in rules.RULES:
    row = []
    for point in POINTS:
        p, choice, _ = run(point, rule=rule)
        row.append(f"{short(p)}")
        OUT["B"].setdefault(rule, {})[point] = {"pooled": p, "choice": {c: list(v) for c, v in choice.items()}}
    L.append(f"| {rule} | " + " | ".join(row) + " |")


def inner_rule(c, point, criterion):
    """The rule with the best inner leave-one-clip-out result on the 12 fit clips of fold c."""
    res, fit = folds[c]["res"], fit_of[c]
    K = rules.CAPS[point]
    scored_fit = sum(sum(rules.cell(res, "cuts", (False, None), s)[k] for k in ("right", "wrong", "lost")) for s in fit)
    budget = int(K * scored_fit // rules.TOTAL_SCORED)
    table = {}
    for rule in rules.RULES:
        cells = []
        for d in fit:
            inner_fit = [s for s in fit if s != d]
            best, _ = choose_multi(res, ("r1",), inner_fit, point, rule)
            cells.append(rules.cell(res, "r1", (best[1], best[2]), d))
        table[rule] = rules.pool(cells)
    order = {r: -i for i, r in enumerate(rules.RULES)}          # ties: the rule listed first (B first)
    if criterion == "cap":        # most right among the rules whose inner held-out wrong returns keep the cap
        feas = [r for r in table if table[r]["wrong"] <= budget]
        best = (max(feas, key=lambda r: (table[r]["right"], order[r])) if feas
                else max(table, key=lambda r: (-table[r]["wrong"], table[r]["right"], order[r])))
    else:                         # largest right - lam * (wrong returns + wrong links + leaf disagreements)
        lam = rules.LAMBDA[point]
        best = max(table, key=lambda r: (table[r]["right"] - lam * (table[r]["wrong"] + table[r]["links_wrong"] + table[r]["leaf_disagree"]), order[r]))
    return best, table


for crit, label in (("cap", "rule chosen inside the fold: most right with the inner held-out wrong returns within the cap"),
                    ("J", "rule chosen inside the fold: largest inner held-out right - lam x errors")):
    row = []
    for point in POINTS:
        picks, choice = {}, {}
        for c in CLIPS:
            picks[c], _ = inner_rule(c, point, crit)
            choice[c] = choose_multi(folds[c]["res"], ("r1",), fit_of[c], point, picks[c])[0]
        p, _ = heldout(choice)
        cnt = Counter(picks.values())
        row.append(f"{short(p)} ({', '.join(f'{n}x {k}' for k, n in cnt.most_common())})")
        OUT["B"].setdefault(f"nested_{crit}", {})[point] = {"pooled": p, "rule_per_fold": picks, "choice": {c: list(v) for c, v in choice.items()}}
    L.append(f"| {label} | " + " | ".join(row) + " |")
L += ["", "Inner results of the four rules in the fold that holds SNGS-162 out and in the fold that holds SNGS-107 out (pooled over the 12 inner held-out clips):", ""]
for c in ("SNGS-162", "SNGS-107"):
    for point in POINTS:
        _, table = inner_rule(c, point, "cap")
        L.append(f"- {c}, {point}: " + "; ".join(f"{r} {short(t)}" for r, t in table.items()))
L.append("")

# ---------------------------------------------------------------- C: design variants offered to the rule
if len(VARS) > 2:
    sets = {"r1 only (the method)": ("r1",), "r1 or r0 (tiers or not)": ("r1", "r0"), "solid from 5 / 3 / 10 boxes": ("r1", "n3", "n10"),
            "rival shares >= 1 or >= 5 frames": ("r1", "c5"), "all five": ("r1", "r0", "n3", "n10", "c5")}
else:
    sets = {"r1 only (the method)": ("r1",), "r1 or r0 (tiers or not)": ("r1", "r0")}
L += ["## C. Design choices offered to rule B inside each fold", "",
      "r0 = round 0's linker (no tiers); n3 / n10 = solid from 3 / 10 boxes; c5 = a rival must share >= 5 frames with the partner (m_appear fixed '>= 1' after looking at all 13 clips).",
      "Rule B picks the (variant, leaves, margin) on the 12 fit clips; ties go to the method's own variant.", "",
      "| offered | strict | balanced | recall |", "|---|---|---|---|"]
OUT["C"] = {}
for name, vs in sets.items():
    row = []
    for point in POINTS:
        p, choice, cells = run(point, variants=vs)
        row.append(f"{short(p)} ({describe(choice)})")
        OUT["C"].setdefault(name, {})[point] = {"pooled": p, "choice": {c: list(v) for c, v in choice.items()},
                                                 "per_clip": {c: [x["right"], x["wrong"], x["lost"]] for c, x in cells.items()}}
    L.append(f"| {name} | " + " | ".join(row) + " |")
L += ["", "Each variant alone under rule B (held-out), and at the method's frozen settings (0.60 core / 0.40 core / 0.20 +leaves; held-out weights, setting not held-out):", "",
      "| variant | strict | balanced | recall | frozen strict | frozen balanced | frozen recall |", "|---|---|---|---|---|---|---|"]
frozen = {"strict": (False, 0.60), "balanced": (False, 0.40), "recall": (True, 0.20)}
OUT["C_alone"] = {}
for v in VARS:
    row = []
    for point in POINTS:
        p, choice, _ = run(point, variants=(v,))
        row.append(f"{short(p)} ({describe(choice)})")
        OUT["C_alone"].setdefault(v, {})[point] = p
    for point in POINTS:
        lv, m = frozen[point]
        lv = lv and v != "r0"
        p, _ = heldout({c: (v, lv, m) for c in CLIPS})
        row.append(short(p))
        OUT["C_alone"][v][f"frozen_{point}"] = p
    L.append(f"| {v} | " + " | ".join(row) + " |")
L.append("")

# ---------------------------------------------------------------- D: lam and trimming
L += ["## D. Rule B with another penalty or without the trimming (the method's variant r1; held-out)", "",
      "| lam | with the worst fit clip left out of J (rule B) | no clip left out |", "|---|---|---|"]
OUT["D"] = {}
for lam in (1, 2, 3, 5, 7, 10, 15, 20):
    row = []
    for trim in (True, False):
        cellsrow = []
        for point in POINTS:
            p, choice, _ = run(point, lam=float(lam), trim=trim)
            cellsrow.append(f"{point} {short(p)}")
            OUT["D"].setdefault(str(lam), {}).setdefault("trim" if trim else "notrim", {})[point] = p
        row.append("; ".join(cellsrow))
    L.append(f"| {lam} | " + " | ".join(row) + " |")
L += ["", "(The cap K of a point stays 4 / 7 / 14; fuse_r1 uses lam 10 / 3 / 1 for strict / balanced / recall.)", ""]

# ---------------------------------------------------------------- E: is "the worst fit clip" a name?
L += ["## E. Rule B leaves 'the fit clip with the most errors' out of J: is that SNGS-162 by another name?", "",
      "Rule B against the same rule with SNGS-162 named (left out of J whenever it is a fit clip; nothing left out in the fold that holds it out):", "",
      "| point | rule B | 'J without SNGS-162' | folds with the same setting | clip rule B leaves out, per fold |", "|---|---|---|---|---|"]
OUT["E"] = {}
for point in POINTS:
    pb, cb, _ = run(point)
    pn, cn, _ = run(point, drop_named="SNGS-162")
    dropped = Counter(choose_multi(folds[c]["res"], ("r1",), fit_of[c], point, "B")[1]["dropped"] for c in CLIPS)
    same = sum(cb[c] == cn[c] for c in CLIPS)
    L.append(f"| {point} | {short(pb)} | {short(pn)} | {same} of 13 | {', '.join(f'{n}x {k}' for k, n in dropped.most_common())} |")
    OUT["E"][point] = {"B": pb, "named": pn, "same_folds": same, "dropped": dict(dropped)}
L.append("")

# ---------------------------------------------------------------- F: where the variants differ (fixed settings, held-out weights)
L += ["## F. Fixed settings, held-out weights: the 12 clips other than SNGS-162 against SNGS-162 alone", "",
      "right / wrong / lost; leaves off. (No setting is chosen here; this shows on which clips a design variant differs from the method.)", "",
      "| margin | " + " | ".join(f"{v}: 12 other clips | {v}: SNGS-162" for v in VARS) + " |", "|---|" + "---|---|" * len(VARS)]
OUT["F"] = {}
for m in (0.40, 0.50, 0.55, 0.60, 0.70, 0.80):
    row = []
    for v in VARS:
        cells = {c: rules.cell(folds[c]["res"], v, (False, m), c) for c in CLIPS}
        rest = rules.pool(x for c, x in cells.items() if c != "SNGS-162")
        x = cells["SNGS-162"]
        row += [short(rest), f"{x['right']} / {x['wrong']} / {x['lost']}"]
        OUT["F"].setdefault(str(m), {})[v] = {"rest": [rest["right"], rest["wrong"], rest["lost"]], "SNGS-162": [x["right"], x["wrong"], x["lost"]]}
    L.append(f"| {m:.2f} | " + " | ".join(row) + " |")
L.append("")

(HERE / "out").mkdir(exist_ok=True)
(HERE / f"out/nested{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(HERE / f"out/nested{tag}.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8")
print("\n".join(L))
