"""DIAGNOSIS (labels, weak: majority person of >= 1 matched row; no method result): what the
evidence looks like for pairs that involve a short piece (the pieces score.py cannot judge)."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
from reid_lab import data  # noqa: E402
import evidence  # noqa: E402
import fuse_labels as fitlabels  # noqa: E402
import prep  # noqa: E402

rows = []
for seq in data.DEV_CLIPS:
    cuts = prep.cuts_for(seq, seq)
    num = seq.split("-")[1]
    pp = json.load(open(prep.WORK + f"m_space/params_cache/excl_{num}.json", encoding="utf-8"))
    rows.append(fitlabels.label_pairs(evidence.pair_table(seq, cuts, pp), seq, cuts, min_rows=1))
D = pd.concat(rows, ignore_index=True)
D["minrows"] = np.minimum(D.rows_a, D.rows_b)
K = D[D.known]
print("weak-known pairs", len(K), "imm", int(K.imm.sum()))
for lo, hi in ((1, 2), (2, 5), (5, 15), (15, 50), (50, 10**6)):
    k = K[(K.minrows >= lo) & (K.minrows < hi)]
    t, f = k[k.imm], k[~k.same]
    if not len(t):
        continue
    print(f"min rows [{lo},{hi}) pairs {len(k):6d} imm {len(t):4d} diff {len(f):6d} | cos_w imm q10/q50/q90 {t.cos_w.quantile(.1):.2f}/{t.cos_w.median():.2f}/{t.cos_w.quantile(.9):.2f}"
          f" diff q50/q90/q99 {f.cos_w.median():.2f}/{f.cos_w.quantile(.9):.2f}/{f.cos_w.quantile(.99):.2f}"
          f" | imm with sink>=.4: {int((t.pos_sink >= .4).sum())}, diff with sink>=.4: {int((f.pos_sink >= .4).sum())}"
          f" | imm plug {int(t.plug.sum())} diff plug {int(f.plug.sum())}")
k = K[K.minrows < 5]
print("\nshort-piece pairs (min rows < 5), by evidence:")
for name, m in (("plug", k.plug == 1), ("sink>=.5", k.pos_sink >= .5), ("sink .3-.5", (k.pos_sink >= .3) & (k.pos_sink < .5)),
                ("plug & cos_raw>=.85", (k.plug == 1) & (k.cos_raw >= .85)), ("plug & cos_raw<.85", (k.plug == 1) & (k.cos_raw < .85)),
                ("plug & sink>=.2", (k.plug == 1) & (k.pos_sink >= .2)), ("cos_w>=.5", k.cos_w >= .5), ("cos_w>=.5 & sink>=.2", (k.cos_w >= .5) & (k.pos_sink >= .2)),
                ("cos_w>=.3 & sink>=.3", (k.cos_w >= .3) & (k.pos_sink >= .3)), ("sink>=.5 & cos_w<.2", (k.pos_sink >= .5) & (k.cos_w < .2)),
                ("sink>=.5 & kind>=1", (k.pos_sink >= .5) & (k.kind >= 1)), ("sink>=.5 & kind==0", (k.pos_sink >= .5) & (k.kind == 0))):
    kk = k[m]
    print(f"  {name:24s} n {len(kk):4d} imm {int(kk.imm.sum()):4d} same-not-imm {int((kk.same & ~kk.imm).sum()):3d} diff {int((~kk.same).sum()):4d}")
k = K[K.minrows >= 5]
print("\nlonger pairs (min rows >= 5), weak labels:")
for name, m in (("plug", k.plug == 1), ("sink>=.5", k.pos_sink >= .5), ("sink>=.5 & kind>=1", (k.pos_sink >= .5) & (k.kind >= 1)), ("sink>=.5 & kind==0", (k.pos_sink >= .5) & (k.kind == 0)),
                ("sink>=.5 & cos_w<.2", (k.pos_sink >= .5) & (k.cos_w < .2)), ("sink>=.5 & cos_w>=.2", (k.pos_sink >= .5) & (k.cos_w >= .2)),
                ("plug & cos_raw>=.85", (k.plug == 1) & (k.cos_raw >= .85)), ("plug & cos_raw<.85", (k.plug == 1) & (k.cos_raw < .85))):
    kk = k[m]
    print(f"  {name:24s} n {len(kk):4d} imm {int(kk.imm.sum()):4d} same-not-imm {int((kk.same & ~kk.imm).sum()):3d} diff {int((~kk.same).sum()):4d}")
