"""DIAGNOSIS (labels; in-sample on the fit clips of one fold; no method result).
(1) Oracle on the cut pieces: link all sub-pieces of one known person -> right / wrong / lost.
(2) Every merge the linker makes at one setting, judged by the majority person of the two groups
    (weak labels, >= 1 matched row), with its appearance cosine and bonus."""
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data, score  # noqa: E402
import evidence  # noqa: E402
import fuse_labels as fitlabels  # noqa: E402
import linker  # noqa: E402
import explore_engine as loco  # noqa: E402
import prep  # noqa: E402

fold = sys.argv[1] if len(sys.argv) > 1 else "SNGS-107"
variant = sys.argv[2] if len(sys.argv) > 2 else "pos"
t_abs = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
r = pickle.load(open(f"C:/Research/plugins/tools/reid_lab/work/fuse/cache/onefold_{fold}.pkl", "rb"))
fit = [s for s in data.DEV_CLIPS if s != fold]
pp = loco.pos_params(fold)

# (1) oracle
tot = Counter()
for s in fit:
    cuts = prep.norm_cuts(r["cuts"][s])
    ident = fitlabels.identities(s, cuts)
    by = {}
    for pid, (ref, top, n) in ident.items():
        if top >= 5:
            by.setdefault(ref, []).append(pid)
    links = [[v[i], v[i + 1]] for v in by.values() for i in range(len(v) - 1)]
    res = score.score_clip(s, links, r["cuts"][s])
    tot.update(res["returns"])
    base = score.score_clip(s, [], r["cuts"][s])
    tot.update({"base_" + k: v for k, v in base["returns"].items()})
print("oracle on cut pieces (12 fit clips):", dict(tot))

# (2) merges
rows = []
model = r["models"][variant]
for s in fit:
    cuts = prep.norm_cuts(r["cuts"][s])
    tab = evidence.pair_table(s, cuts, pp)
    bonus = loco.bonus_of(tab, model) if model else None
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
        firm = ng >= 5 and nh >= 5
        rows.append({"seq": s, "s": e["s"], "app": e["app"], "bonus": e["bonus"], "rg": e["rival_g"], "rh": e["rival_h"],
                     "ng": len(e["g"]), "nh": len(e["h"]), "mg": ng, "mh": nh, "verdict": verdict, "firm": firm})
M = pd.DataFrame(rows)
print(f"\n{variant} t_abs {t_abs} margin {margin}: merges {len(M)};", dict(M.verdict.value_counts()), "| firm:", dict(M[M.firm].verdict.value_counts()))
ab = [-1, 0, .1, .2, .3, .4, .5, .6, .7, 1.01]
bb = [-.01, 0, .2, .4, .6, .8, 1.5]
print("cells right/wrong (weak labels); rows appearance cosine, columns bonus")
print("app \\ bonus  " + " ".join(f"{'<=' + str(bb[j + 1]):>9s}" for j in range(len(bb) - 1)))
for i in range(len(ab) - 1):
    line = f"[{ab[i]:5.2f},{ab[i + 1]:4.2f}) "
    for j in range(len(bb) - 1):
        m = (M.app >= ab[i]) & (M.app < ab[i + 1]) & (M.bonus > bb[j]) & (M.bonus <= bb[j + 1])
        line += f"{int((M[m].verdict == 'right').sum()):4d}/{int((M[m].verdict == 'wrong').sum()):<4d} "
    print(line)
pd.set_option("display.width", 250)
print("\nwrong merges:")
print(M[M.verdict == "wrong"].round(3).to_string(index=False))
