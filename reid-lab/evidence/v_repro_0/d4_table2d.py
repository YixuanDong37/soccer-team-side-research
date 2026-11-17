"""DIAGNOSIS (labels, weak: >= 1 matched row; in-sample; no method result): joint table of
appearance (cos_w) and position (pos_sink) evidence: immediate-same vs different pairs. Also the
LOCO comparison of pair-model input sets on weak labels, split by short / long pieces."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
from reid_lab import data  # noqa: E402
import evidence  # noqa: E402
import fuse_labels as fitlabels  # noqa: E402
import pairmodel  # noqa: E402
import prep  # noqa: E402

rows = []
for seq in data.DEV_CLIPS:
    cuts = prep.cuts_for(seq, seq)
    num = seq.split("-")[1]
    pp = json.load(open(prep.WORK + f"m_space/params_cache/excl_{num}.json", encoding="utf-8"))
    rows.append(fitlabels.label_pairs(evidence.pair_table(seq, cuts, pp), seq, cuts, min_rows=1))
D = pd.concat(rows, ignore_index=True)
D["minrows"] = np.minimum(D.rows_a, D.rows_b)
D.to_pickle("C:/Research/plugins/tools/reid_lab/work/v_repro_0/cache/d4_pairs_weak.pkl")
K = D[D.known & (D.imm | ~D.same)]
cb = [-1, 0, .1, .2, .3, .4, .5, .6, .7, 1.01]
sb = [-.01, .001, .01, .05, .2, .4, .6, .8, 1.01]
for title, k in (("ALL", K), ("min rows >= 5", K[K.minrows >= 5]), ("min rows < 5", K[K.minrows < 5])):
    print(f"\n{title}: cells 'imm/diff'; rows cos_w, columns pos_sink (first column: sink <= 0.001 incl. not admissible)")
    print("cos_w \\ sink " + " ".join(f"{'<=' + str(sb[j + 1]):>9s}" for j in range(len(sb) - 1)))
    for i in range(len(cb) - 1):
        line = f"[{cb[i]:5.2f},{cb[i + 1]:4.2f}) "
        for j in range(len(sb) - 1):
            m = (k.cos_w >= cb[i]) & (k.cos_w < cb[i + 1]) & (k.pos_sink > sb[j]) & (k.pos_sink <= sb[j + 1])
            line += f"{int(k[m].imm.sum()):4d}/{int((~k[m].same).sum()):<4d} "
        print(line)
    kk = k[k.pos_unreach == 1]
    print("unreachable:", int(kk.imm.sum()), "imm /", int((~kk.same).sum()), "diff;  imm among them with cos_w >= .5:", int((kk.imm & (kk.cos_w >= .5)).sum()),
          " diff with cos_w >= .5:", int((~kk.same & (kk.cos_w >= .5)).sum()))

SETS = {
    "cos_w": ["cos_w"],
    "cos_w+unreach+sink": ["cos_w", "unreach", "sink"],
    "cos_w+unreach+sink_lo": ["cos_w", "unreach", "sink_lo"],
    "cos_w+unreach+sink+plug": ["cos_w", "unreach", "sink", "plug"],
    "cos_w+unreach+sink_lo+plug": ["cos_w", "unreach", "sink_lo", "plug"],
    "cos_w+unreach+sink_e/in+plug": ["cos_w", "unreach", "sink_edge", "sink_in", "plug"],
    "cos_w+cos_raw+unreach+sink_lo": ["cos_w", "cos_raw", "unreach", "sink_lo"],
    "cos_tc+unreach+sink_lo": ["cos_tc", "unreach", "sink_lo"],
    "cos_w+unreach+sink_lo+lgap": ["cos_w", "unreach", "sink_lo", "lgap"],
}


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def prec_at(y, s, rec):
    order = np.argsort(-s)
    tp = np.cumsum(y[order])
    k = int(np.searchsorted(tp, rec * y.sum()))
    return tp[k] / (k + 1)


print("\nLOCO, weak labels (fit and test on pairs with >= 1 matched row per piece):")
for name, cols in SETS.items():
    out = []
    for fit_clips, held in data.loco_folds():
        m = pairmodel.fit(D[D.seq.isin(fit_clips)], cols)
        h = D[(D.seq == held) & D.known & (D.imm | ~D.same)]
        out.append(pd.DataFrame({"z": pairmodel.logit(h, m), "y": h.imm.to_numpy(float), "short": (h.minrows < 5).to_numpy()}))
        last = m
    o = pd.concat(out)
    line = f"{name:30s}"
    for lab, q in (("all", o), ("long", o[~o.short]), ("short", o[o.short])):
        y, z = q.y.to_numpy(), q.z.to_numpy()
        ll = float((np.logaddexp(0, z) - y * z).mean())
        line += f" | {lab}: ll {ll:.4f} auc {auc(y, z):.4f} p@.5/.7/.85 {prec_at(y, z, .5):.3f}/{prec_at(y, z, .7):.3f}/{prec_at(y, z, .85):.3f}"
    print(line, " last-fold w", np.round(last["w"], 2).tolist(), "b", round(last["b"], 2))
