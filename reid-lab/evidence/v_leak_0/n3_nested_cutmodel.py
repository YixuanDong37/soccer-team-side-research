"""v_leak: NESTED choice of m_cuts' feature set. m_cuts chose its inputs "among 9 by held-out log-loss" on all
13 clips, and fuse cuts with that model. Here, for every outer fold, the feature set is chosen by INNER
leave-one-clip-out log-loss over the 12 fit clips only (labels of the held-out clip are not used in its fold:
its rows are filtered out before anything is fitted), the chosen model is fitted on the 12 fit clips, and the
cuts it makes on the held-out clip at P >= 0.7 are compared with the cuts fuse used.

    python -I -B n3_nested_cutmodel.py   -> out/nested_cutmodel.md
"""
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
from reid_lab import data  # noqa: E402
import features as F  # noqa: E402
import loco  # noqa: E402

HERE = "C:/Research/plugins/tools/reid_lab/work/v_leak_0/"
USED = "lr(cos,mm,lgap,ld)"
tab = loco.table()
th = float(np.log(0.7 / 0.3))


def nll(par, rows):
    rows = rows[rows.label.isin(["T", "F"])]
    z = F.apply_model(par, rows)
    y = (rows.label == "T").to_numpy(float)
    return float(np.sum(np.logaddexp(0, z) - y * z)), len(rows)


L = ["# Nested choice of m_cuts' feature set (inner LOCO log-loss on the 12 fit clips of each fold)", "",
     "| fold | chosen inside the fold | inner log-loss: chosen / m_cuts' set | held-out cuts: nested model | fuse's cuts | in both | only nested | only fuse |", "|---|---|---|---|---|---|---|---|"]
tot = np.zeros(4, int)
picked = {}
for fit_clips, held in data.loco_folds():
    fit_tab = tab[tab.seq.isin(fit_clips)]          # the held-out clip's rows (and labels) are gone from here on
    inner = {}
    for name, cols in loco.MODELS.items():
        s, n = 0.0, 0
        for i in fit_clips:
            par = loco.fit_lr(cols, fit_tab[fit_tab.seq != i])
            a, b = nll(par, fit_tab[fit_tab.seq == i])
            s, n = s + a, n + b
        inner[name] = s / n
    best = min(inner, key=inner.get)
    picked[held] = best
    par = loco.fit_lr(loco.MODELS[best], fit_tab)
    mine = {tuple(x) for x in F.cuts(held, {**par, "threshold": th})}
    theirs = {tuple(x) for x in json.load(open(HERE + f"refit/cuts_{held}.json", encoding="utf-8"))["cuts"][held]}
    row = [len(mine), len(theirs), len(mine & theirs), len(mine - theirs), len(theirs - mine)]
    tot += np.array(row[1:])
    L.append(f"| {held} | {best} | {inner[best]:.4f} / {inner[USED]:.4f} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
L.append(f"| **all** | {sum(v == USED for v in picked.values())} of 13 folds choose m_cuts' set | | | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} |")
open(HERE + "out/nested_cutmodel.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
