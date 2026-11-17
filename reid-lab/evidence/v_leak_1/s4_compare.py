"""v_leak round 1, stage 4 (not blind; LABELS: scoring): compare the blindfolded re-run of every fold
(s3_blind_r1.py) with what fuse_r1 published, score the blind held-out links, and assemble the cells
that the nested analyses use.

    python -I -B s4_compare.py   -> out/compare.md, out/compare.json, out/METHOD_blind_<point>.json, out/blind_folds.pkl
"""
import json
import pickle
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path("C:/Research/plugins/tools/reid_lab/work/v_leak_1")
R1 = Path("C:/Research/plugins/tools/reid_lab/work/fuse_r1")
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, str(R1))
from reid_lab import data, score  # noqa: E402
import fit_loco  # noqa: E402      (fuse_r1: _scored)
import fuse_labels  # noqa: E402   (fuse_r1, labels: weak-label counts of the scored cells)
import rules  # noqa: E402

POINTS = ("strict", "balanced", "recall")
norm = lambda links: sorted(tuple(sorted(map(int, l))) for l in links)  # noqa: E731
normc = lambda cuts: sorted((int(p), int(f)) for p, f in cuts)  # noqa: E731

params = json.load(open(R1 / "params_loco.json", encoding="utf-8"))
theirs = {p: json.load(open(R1 / f"METHOD_{p}.json", encoding="utf-8"))["clips"] for p in POINTS}
raw = pickle.load(open(R1 / "cache/loco_raw.pkl", "rb"))["folds"]
rows, mine, folds = [], {p: {} for p in POINTS}, {}
for fold in data.DEV_CLIPS:
    b = pickle.load(open(HERE / f"out/blind/{fold}.pkl", "rb"))
    pm, tm = b["models"][5], params[fold]["pair_model"]
    dw = max(max(abs(x - y) for x, y in zip(pm["w"], tm["w"])), abs(pm["b"] - tm["b"]))
    dw0 = max(max(abs(x - y) for x, y in zip(b["models"]["r0"]["w"], raw[fold]["models"]["r0"]["w"])), abs(b["models"]["r0"]["b"] - raw[fold]["models"]["r0"]["b"]))
    ch = {p: (b["chosen"][("r1", "B", p)]["leaves"], b["chosen"][("r1", "B", p)]["margin"]) for p in POINTS}
    th = {p: (params[fold]["choice"][p]["leaves"], params[fold]["choice"][p]["margin"]) for p in POINTS}
    same_choice = all(ch[p] == th[p] for p in POINTS)
    # every rule, both linkers, against fuse_r1's report
    rep = json.load(open(R1 / "out/loco_report.json", encoding="utf-8"))["variants"]
    same_rules = all((b["chosen"][(v, r, p)]["leaves"], b["chosen"][(v, r, p)]["margin"]) ==
                     (rep[v]["rules"][r][p]["chosen_per_fold"][fold]["leaves"], rep[v]["rules"][r][p]["chosen_per_fold"][fold]["margin"])
                     for v in ("r1", "r0") for r in rules.RULES for p in POINTS)
    links_at = {}
    for p in POINTS:
        lv, m = ch[p]
        h = b["held"][("r1", m)]
        links_at[p] = h["core"] + (h["leaf"] if lv else [])
    same_links = all(norm(links_at[p]) == norm(theirs[p][fold]["links"]) for p in POINTS)
    same_cuts = all(normc(b["cuts"][fold]) == normc(theirs[p][fold]["cuts"]) for p in POINTS)
    all_r1 = all(norm(h["core"]) == norm(raw[fold]["res"][("r1", m, fold)]["core"]) and norm(h["leaf"]) == norm(raw[fold]["res"][("r1", m, fold)]["leaf"])
                 for (v, m), h in b["held"].items() if v == "r1")
    all_r0 = all(norm(h["core"]) == norm(raw[fold]["res"][("r0", m, fold)]["core"]) for (v, m), h in b["held"].items() if v == "r0")
    # fit-clip cells against fuse_r1's raw cells (same scored outcomes?)
    same_cells = all(b["cells"][k] == {kk: raw[fold]["res"][k][kk] for kk in b["cells"][k]} for k in b["cells"] if k[0] in ("r1", "r0", "cuts"))
    a = b["audit"]
    rows.append({"fold": fold, "pair_model_max_abs_diff": dw, "r0_pair_model_max_abs_diff": dw0, "choice_mine": ch, "choice_theirs": th, "same_choice_rule_B": same_choice,
                 "same_choice_all_rules_both_linkers": same_rules, "same_links_at_points": same_links, "same_cuts": same_cuts,
                 "same_heldout_links_32_candidates_r1": all_r1, "same_heldout_links_16_margins_r0": all_r0, "same_fit_cells": same_cells,
                 "linker2c_covis1_equals_linker2": b["same_as_linker2"],
                 "label_files_opened": len(a["label_files"]), "heldout_label_opened": any(fold in p for p in a["label_files"]) or any(fold in p for p in a["reentry_eval"]),
                 "fitted_files_of_other_agents_read": a["work_derived_foreign"], "blocked": a["blocked"], "seconds": b["seconds"]})
    for p in POINTS:
        mine[p][fold] = {"links": links_at[p], "cuts": b["cuts"][fold]}
    # cells of the fold with the held-out clip scored here (after every choice of the fold was made in the blind process)
    res = dict(b["cells"])
    cl = b["cuts"][fold]
    ident = fuse_labels.identities(fold, tuple(tuple(x) for x in normc(cl)))
    res[("cuts", None, fold)] = {"off": fit_loco._scored(score, fold, [], cl)}
    for (v, m), h in b["held"].items():
        if v == "r0":
            res[(v, m, fold)] = {"off": fit_loco._scored(score, fold, h["core"], cl, ident)}
        else:
            res[(v, m, fold)] = {"off": fit_loco._scored(score, fold, h["core"], cl, ident, []),
                                 "on": fit_loco._scored(score, fold, h["core"] + h["leaf"], cl, ident, h["leaf"])}
    folds[fold] = {"res": res, "models": b["models"], "chosen": b["chosen"]}
    print(fold, "compared", flush=True)
with open(HERE / "out/blind_folds.pkl", "wb") as fh:
    pickle.dump(folds, fh)

out = {"folds": rows, "points": {}}
for p in POINTS:
    json.dump({"name": f"v_leak_1 blindfolded re-run of fuse_r1, point {p} (LOCO held-out)", "clips": mine[p]}, open(HERE / f"out/METHOD_blind_{p}.json", "w"))
    out["points"][p] = {"mine": score.evaluate(mine[p])["pooled"], "theirs": score.evaluate(theirs[p])["pooled"]}
json.dump(out, open(HERE / "out/compare.json", "w"), indent=1, default=str)
L = ["# Blindfolded re-run of every LOCO fold of fuse_r1 against its published files", "",
     "| fold | pair model max abs diff (r1 / r0) | rule B choices equal | all 4 rules x 2 linkers equal | links at the 3 points equal | cuts equal | held-out links equal: 32 settings r1 / 16 r0 | fit-clip cells equal | "
     "fit-clip label files opened | held-out labels opened | fitted or cached files of other agents read | blocked attempts |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['fold']} | {r['pair_model_max_abs_diff']:.1e} / {r['r0_pair_model_max_abs_diff']:.1e} | {r['same_choice_rule_B']} | {r['same_choice_all_rules_both_linkers']} | {r['same_links_at_points']} | "
             f"{r['same_cuts']} | {r['same_heldout_links_32_candidates_r1']} / {r['same_heldout_links_16_margins_r0']} | {r['same_fit_cells']} | {r['label_files_opened']} | {r['heldout_label_opened']} | "
             f"{len(r['fitted_files_of_other_agents_read'])} | {len(r['blocked'])} |")
f = lambda d: f"{d['right']} / {d['wrong']} / {d['lost']} | {d['links_right']} / {d['links_wrong']} / {d['links_unknown']} | {100 * d['impure_row_share']:.2f}%"  # noqa: E731
L += ["", "| point | blind re-run (score.evaluate) | fuse_r1's METHOD file (score.evaluate) |", "|---|---|---|"]
for p in POINTS:
    L.append(f"| {p} | {f(out['points'][p]['mine'])} | {f(out['points'][p]['theirs'])} |")
(HERE / "out/compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
