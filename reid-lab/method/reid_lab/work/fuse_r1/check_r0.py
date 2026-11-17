"""fuse_r1 / check_r0.py -- regression check of this directory's copy of the pipeline: the variant
"r0" (round 0's linker run here from scratch: same cuts, tables, pair model, linker) must give the
same fixed-margin rows, held-out and in-sample, as fuse's own report (work/fuse/out/loco_report.json,
read-only), and the same pair-model weights as fuse's params_loco.json.
Reads only result files (no annotation)."""
import json
import sys

HERE = "C:/Research/plugins/tools/reid_lab/work/fuse_r1/"
FUSE = "C:/Research/plugins/tools/reid_lab/work/fuse/"
mine = json.load(open(HERE + "out/loco_report.json", encoding="utf-8"))["variants"]["r0"]
theirs = json.load(open(FUSE + "out/loco_report.json", encoding="utf-8"))["variants"]["fuse"]
keys = ("right", "wrong", "lost", "links_right", "links_wrong", "links_unknown", "impure_row_share")
bad = 0
for m in json.load(open(FUSE + "out/loco_report.json", encoding="utf-8"))["margins"]["fuse"]:
    for a, b, name in ((mine["fixed_heldout"], theirs["fixed_margin_heldout"], "held-out"),
                       (mine["fixed_all13_insample"], theirs["fixed_margin_all13_insample"], "in-sample")):
        x, y = a[f"off|{m}"], b[str(m)]
        same = all(x[k] == y[k] for k in keys)
        bad += not same
        print(f"margin {m:<5} {name:<9}", " / ".join(str(x[k]) for k in keys[:3]), "equal" if same else f"DIFFERENT from {[y[k] for k in keys]}")
pf = json.load(open(FUSE + "params_loco.json", encoding="utf-8"))
d = max(abs(u - v) for c in pf for u, v in zip(mine["pair_model"][c]["w"] + [mine["pair_model"][c]["b"]],
                                              pf[c]["pair_model"]["w"] + [pf[c]["pair_model"]["b"]]))
print("largest difference of a pair-model weight over the 14 fits:", d)
bad += d > 1e-9
print("r0 REPRODUCES fuse" if not bad else f"{bad} DIFFERENCES")
sys.exit(1 if bad else 0)
