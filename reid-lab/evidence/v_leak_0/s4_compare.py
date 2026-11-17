"""v_leak stage 4 (not blind; scoring): compare the blindfolded re-run (stages 1-3) with what fuse published,
and score the blind links with reid_lab/score.py.

    python -I -B s4_compare.py   -> out/compare.json, out/compare.md, out/METHOD_blind_<point>.json
"""
import json
import pickle
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402

HERE = "C:/Research/plugins/tools/reid_lab/work/v_leak_0/"
FUSE = "C:/Research/plugins/tools/reid_lab/work/fuse/"
POINTS = ("strict", "balanced", "recall")
norm = lambda links: sorted(tuple(sorted(map(int, l))) for l in links)  # noqa: E731

params = json.load(open(FUSE + "params_loco.json", encoding="utf-8"))
theirs = {p: json.load(open(FUSE + f"METHOD_{p}.json", encoding="utf-8"))["clips"] for p in POINTS}
raw = pickle.load(open(FUSE + "cache/loco_raw.pkl", "rb"))["folds"]
rows, mine, fixed = [], {p: {} for p in POINTS}, {}
for fold in data.DEV_CLIPS:
    b = json.load(open(HERE + f"out/blind/{fold}.json", encoding="utf-8"))
    pm, tm = b["pair_model"], params[fold]["pair_model"]
    dw = max(max(abs(x - y) for x, y in zip(pm["w"], tm["w"])), abs(pm["b"] - tm["b"]))
    same_margin = {p: b["margin"][p] == params[fold]["margin"][p] for p in POINTS}
    same_links = {p: norm(b["links"][str(b["margin"][p])]) == norm(theirs[p][fold]["links"]) for p in POINTS}
    same_cuts = norm(b["cuts"]) == norm(theirs["strict"][fold]["cuts"])
    all_m = {m: norm(links) == norm(raw[fold]["res"][("fuse", float(m), fold)]["links"]) for m, links in b["links"].items()}
    a = b["audit"]
    rows.append({"fold": fold, "pair_model_max_abs_diff": dw, "margins_mine": b["margin"], "margins_theirs": params[fold]["margin"], "same_margins": all(same_margin.values()),
                 "same_links_at_points": all(same_links.values()), "same_cuts": same_cuts, "same_heldout_links_all_16_margins_vs_loco_raw": all(all_m.values()),
                 "label_files_opened": len(a["label_files"]), "heldout_label_opened": any(fold in p for p in a["label_files"]) or any(fold in p for p in a["reentry_eval"]),
                 "fitted_files_of_other_agents_read": a["work_derived_foreign"], "blocked": a["blocked"]})
    for p in POINTS:
        mine[p][fold] = {"links": b["links"][str(b["margin"][p])], "cuts": b["cuts"]}
    for m, links in b["links"].items():
        fixed.setdefault(m, {})[fold] = {"links": links, "cuts": b["cuts"]}
res = {"folds": rows, "points": {}, "fixed_margin": {}}
for p in POINTS:
    json.dump({"name": f"v_leak blind re-run of fuse, point {p} (LOCO held-out)", "clips": mine[p]}, open(HERE + f"out/METHOD_blind_{p}.json", "w"))
    res["points"][p] = {"mine": score.evaluate(mine[p])["pooled"], "theirs": score.evaluate(theirs[p])["pooled"]}
for m, meth in fixed.items():
    res["fixed_margin"][m] = score.evaluate(meth)["pooled"]
json.dump(res, open(HERE + "out/compare.json", "w"), indent=1)
L = ["# Blindfolded re-run of every LOCO fold against fuse's published files", "",
     "| fold | pair model max abs diff | margins equal | links at the 3 points equal | cuts equal | held-out links equal at all 16 margins | fit-clip label files opened | held-out labels opened | fitted files of other agents read | blocked attempts |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['fold']} | {r['pair_model_max_abs_diff']:.1e} | {r['same_margins']} | {r['same_links_at_points']} | {r['same_cuts']} | {r['same_heldout_links_all_16_margins_vs_loco_raw']} | "
             f"{r['label_files_opened']} | {r['heldout_label_opened']} | {len(r['fitted_files_of_other_agents_read'])} | {len(r['blocked'])} |")
f = lambda d: f"{d['right']} / {d['wrong']} / {d['lost']} | {d['links_right']} / {d['links_wrong']} / {d['links_unknown']} | {100 * d['impure_row_share']:.2f}%"  # noqa: E731
L += ["", "| point | blind re-run (score.evaluate) | fuse's METHOD file |", "|---|---|---|"]
for p in POINTS:
    L.append(f"| {p} | {f(res['points'][p]['mine'])} | {f(res['points'][p]['theirs'])} |")
L += ["", "| fixed margin | blind re-run, held-out |", "|---|---|"]
for m, d in res["fixed_margin"].items():
    L.append(f"| {m} | {f(d)} |")
open(HERE + "out/compare.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
