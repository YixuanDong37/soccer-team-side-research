"""DIAGNOSIS (labels; in-sample on the fit clips of one fold; no method result): merges made at a
loose setting, tabulated by fused similarity S and by the smaller rival margin (weak labels)."""
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data  # noqa: E402
import evidence  # noqa: E402
import fuse_labels as fitlabels  # noqa: E402
import linker  # noqa: E402
import explore_engine as loco  # noqa: E402
import prep  # noqa: E402

fold = sys.argv[1] if len(sys.argv) > 1 else "SNGS-107"
variant = sys.argv[2] if len(sys.argv) > 2 else "pos"
t_abs = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
r = pickle.load(open(f"C:/Research/plugins/tools/reid_lab/work/fuse/cache/onefold_{fold}.pkl", "rb"))
fit = [s for s in data.DEV_CLIPS if s != fold]
pp = loco.pos_params(fold)
rows = []
model = r["models"][variant]
for s in fit:
    cuts = prep.norm_cuts(r["cuts"][s])
    tab = evidence.pair_table(s, cuts, pp)
    bonus = {k: v * scale for k, v in loco.bonus_of(tab, model).items()} if model else None
    log = []
    linker.agglomerate(s, cuts, bonus, t_abs=t_abs, margin=margin, log=log)
    ident = fitlabels.identities(s, cuts)

    def major(members):
        c = Counter()
        for p in members:
            if p in ident:
                c[ident[p][0]] += ident[p][2]
        return c.most_common(1)[0] if c else (None, 0)

    for e in log:
        (rg, ng), (rh, nh) = major(e["g"]), major(e["h"])
        verdict = "unknown" if rg is None or rh is None else ("right" if rg == rh else "wrong")
        mm = np.nanmin([e["rival_g"], e["rival_h"]]) if not (np.isnan(e["rival_g"]) and np.isnan(e["rival_h"])) else np.nan
        rows.append({"seq": s, "s": e["s"], "app": e["app"], "bonus": e["bonus"], "mm": mm, "verdict": verdict})
M = pd.DataFrame(rows)
print(f"{variant} x{scale} t_abs {t_abs} margin {margin}: merges {len(M)};", dict(M.verdict.value_counts()))


def table(xcol, xb, ycol, yb, nan_col=True):
    print(f"cells right/wrong; rows {xcol}, columns {ycol}" + (" (last column: no rival)" if nan_col else ""))
    print(" " * 14 + " ".join(f"{'<' + str(yb[j + 1]):>9s}" for j in range(len(yb) - 1)) + ("     none" if nan_col else ""))
    for i in range(len(xb) - 1):
        line = f"[{xb[i]:5.2f},{xb[i + 1]:4.2f}) "
        mx = (M[xcol] >= xb[i]) & (M[xcol] < xb[i + 1])
        for j in range(len(yb) - 1):
            m = mx & (M[ycol] >= yb[j]) & (M[ycol] < yb[j + 1])
            line += f"{int((M[m].verdict == 'right').sum()):4d}/{int((M[m].verdict == 'wrong').sum()):<4d} "
        if nan_col:
            m = mx & M[ycol].isna()
            line += f"{int((M[m].verdict == 'right').sum()):4d}/{int((M[m].verdict == 'wrong').sum()):<4d} "
        print(line)


table("s", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.5], "mm", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 3.0])
table("app", [-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01], "mm", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 3.0])
table("app", [-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01], "bonus", [-0.01, 0.001, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.5], nan_col=False)
print("without SNGS-162:")
M = M[M.seq != "SNGS-162"]
table("app", [-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01], "bonus", [-0.01, 0.001, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.5], nan_col=False)
table("s", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.5], "mm", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 3.0])
