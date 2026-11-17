"""DIAGNOSIS (labels, in-sample, all 13 clips; no method result): how the evidence columns separate
same-person from different-person pairs on the cut pieces."""
import json
import sys

import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data  # noqa: E402
import evidence  # noqa: E402
import fuse_labels as fitlabels  # noqa: E402
import prep  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
rows = []
for seq in data.DEV_CLIPS:
    cuts = prep.cuts_for(seq, seq)
    num = seq.split("-")[1]
    pp = json.load(open(prep.WORK + f"m_space/params_cache/excl_{num}.json", encoding="utf-8"))
    df = fitlabels.label_pairs(evidence.pair_table(seq, cuts, pp), seq, cuts)
    rows.append(df)
D = pd.concat(rows, ignore_index=True)
D.to_pickle("C:/Research/plugins/tools/reid_lab/work/fuse/cache/d1_pairs.pkl")
K = D[D.known]
print("pairs", len(D), "known", len(K), "same", int(K.same.sum()), "imm", int(K.imm.sum()))


def tab(name, mask):
    k = K[mask]
    t = k[k.imm]
    f = k[~k.same]
    print(f"{name:28s} n={len(k):6d} imm={len(t):4d} diff={len(f):6d} | cos_w imm q10/50 {t.cos_w.quantile(.1):.2f}/{t.cos_w.median():.2f}"
          f" diff q50/q99 {f.cos_w.median():.2f}/{f.cos_w.quantile(.99):.2f}")


tab("all", K.a > -1)
tab("plug", K.plug == 1)
tab("pos_adm", K.pos_adm == 1)
tab("pos_unreach", K.pos_unreach == 1)
tab("pos_known==0", K.pos_known == 0)
tab("low", K.low == 1)
for lo, hi in ((0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
    m = (K.pos_sink >= lo) & (K.pos_sink < hi) & (K.pos_adm == 1)
    k = K[m]
    print(f"pos_sink [{lo:.2f},{hi:.2f})  n={len(k):6d}  imm={int(k.imm.sum()):4d} same_not_imm={int((k.same & ~k.imm).sum()):4d}"
          f" diff={int((~k.same).sum()):6d}  share imm {k.imm.mean():.3f}")
for kind in (0, 1, 2):
    for lo, hi in ((0.2, .4), (.4, .6), (.6, 1.01)):
        m = (K.pos_sink >= lo) & (K.pos_sink < hi) & (K.kind == kind)
        k = K[m]
        print(f"kind {kind} pos_sink [{lo:.2f},{hi:.2f})  n={len(k):5d} imm={int(k.imm.sum()):4d} same_not_imm={int((k.same & ~k.imm).sum()):3d}"
              f" diff={int((~k.same).sum()):5d}")
print("plug joins: known", int((K.plug == 1).sum()), "same", int(K[K.plug == 1].same.sum()), "| all", int((D.plug == 1).sum()))
k = K[K.plug == 1]
for lo, hi in ((-1, .0), (0, .2), (.2, .4), (.4, .6), (.6, 1.01)):
    m = (k.cos_w >= lo) & (k.cos_w < hi)
    print(f"  plug cos_w [{lo:.1f},{hi:.1f}) same {int(k[m].same.sum())} diff {int((~k[m].same).sum())}")
for lo, hi in ((-1, .8), (.8, .85), (.85, .9), (.9, .95), (.95, 1.01)):
    m = (k.cos_raw >= lo) & (k.cos_raw < hi)
    print(f"  plug cos_raw [{lo:.2f},{hi:.2f}) same {int(k[m].same.sum())} diff {int((~k[m].same).sum())}")
