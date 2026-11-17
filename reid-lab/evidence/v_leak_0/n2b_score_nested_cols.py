"""v_leak: pooled held-out result when the pair model's input set is chosen INSIDE each fold (n2_nested_cols.py)
instead of once on all 13 clips. Folds whose inner choice is fuse's own set use the blind re-run's links.

    python -I -B n2b_score_nested_cols.py   -> out/nested_cols.md, out/METHOD_nestedcols_<point>.json
"""
import json
import sys
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402

HERE = "C:/Research/plugins/tools/reid_lab/work/v_leak_0/"
POINTS = ("strict", "balanced", "recall")
f = lambda d: f"{d['right']} / {d['wrong']} / {d['lost']} | {d['links_right']} / {d['links_wrong']} / {d['links_unknown']} | {100 * d['impure_row_share']:.2f}%"  # noqa: E731
L = ["# Pair-model inputs chosen inside each fold (inner LOCO log-loss on the 12 fit clips)", "",
     "| fold | chosen inside the fold | rank of fuse's set (of 13) | log-loss gap (fuse's - best) | margins (strict / balanced / recall) |", "|---|---|---|---|---|"]
meth = {p: {} for p in POINTS}
base = {p: {} for p in POINTS}
fixed, fixed_base = {}, {}
for fold in data.DEV_CLIPS:
    n = json.load(open(HERE + f"out/nested_cols/{fold}.json", encoding="utf-8"))
    b = json.load(open(HERE + f"out/blind/{fold}.json", encoding="utf-8"))
    src = n if "links" in n else b
    L.append(f"| {fold} | {n['chosen']} | {n['rank_of_fuse_choice']} | {n['logloss_gap_fuse_minus_best']:.5f} | {src['margin']['strict']} / {src['margin']['balanced']} / {src['margin']['recall']} |")
    assert not n["audit"]["heldout_label_opened"] and not n["audit"]["blocked"] and not n["audit"]["foreign_fitted_files"]
    for p in POINTS:
        meth[p][fold] = {"links": src["links"][str(src["margin"][p])], "cuts": src["cuts"]}
        base[p][fold] = {"links": b["links"][str(b["margin"][p])], "cuts": b["cuts"]}
    for m in b["links"]:
        fixed.setdefault(m, {})[fold] = {"links": src["links"][m], "cuts": src["cuts"]}
        fixed_base.setdefault(m, {})[fold] = {"links": b["links"][m], "cuts": b["cuts"]}
L += ["", f"Chosen sets: {dict(Counter(json.load(open(HERE + f'out/nested_cols/{c}.json', encoding='utf-8'))['chosen'] for c in data.DEV_CLIPS))}", "",
      "| point | inputs chosen inside the fold (HELD-OUT) | fuse (inputs chosen once on all 13 clips; HELD-OUT) |", "|---|---|---|"]
for p in POINTS:
    json.dump({"name": f"v_leak: fuse with the pair-model inputs chosen inside each fold, point {p}", "clips": meth[p]}, open(HERE + f"out/METHOD_nestedcols_{p}.json", "w"))
    L.append(f"| {p} | {f(score.evaluate(meth[p])['pooled'])} | {f(score.evaluate(base[p])['pooled'])} |")
L += ["", "| fixed margin | inputs chosen inside the fold (HELD-OUT) | fuse (HELD-OUT) |", "|---|---|---|"]
for m in fixed:
    L.append(f"| {m} | {f(score.evaluate(fixed[m])['pooled'])} | {f(score.evaluate(fixed_base[m])['pooled'])} |")
open(HERE + "out/nested_cols.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
