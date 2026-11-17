"""DIAGNOSIS (labels): leave-one-clip-out log-loss / AUC of pairwise fusion models with different
input sets, on immediate-same vs different pairs. Used to pick the input set of the pair model
(a design choice made on the dev clips; not a method result)."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data  # noqa: E402
import pairmodel  # noqa: E402

D = pd.read_pickle("C:/Research/plugins/tools/reid_lab/work/fuse/cache/d1_pairs.pkl")   # written by d1_look.py
SETS = {
    "cos_w": ["cos_w"],
    "cos_w+low": ["cos_w", "low", "cos_w_low"],
    "cos_w+unreach": ["cos_w", "unreach"],
    "cos_w+unreach+sink": ["cos_w", "unreach", "sink"],
    "cos_w+unreach+sink_lo": ["cos_w", "unreach", "sink_lo"],
    "cos_w+unreach+pos_p": ["cos_w", "unreach", "pos_p"],
    "cos_w+unreach+sink+plug": ["cos_w", "unreach", "sink", "plug"],
    "cos_w+unreach+sink_e/in+plug": ["cos_w", "unreach", "sink_edge", "sink_in", "plug"],
    "cos_w+unreach+sink+plug+low": ["cos_w", "unreach", "sink", "plug", "low", "cos_w_low"],
    "cos_w+unreach+sink+plug+lgap": ["cos_w", "unreach", "sink", "plug", "lgap"],
    "cos_w+cos_raw+unreach+sink+plug": ["cos_w", "cos_raw", "unreach", "sink", "plug"],
    "cos_tc+unreach+sink+plug": ["cos_tc", "unreach", "sink", "plug"],
    "unreach+sink+plug (no app)": ["unreach", "sink", "plug"],
}


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


for name, cols in SETS.items():
    nll, n, ys, ss, ws = 0.0, 0, [], [], []
    for fit_clips, held in data.loco_folds():
        m = pairmodel.fit(D[D.seq.isin(fit_clips)], cols)
        ws.append(m["w"])
        h = D[(D.seq == held) & D.known & (D.imm | ~D.same)]
        z = pairmodel.logit(h, m)
        y = h.imm.to_numpy(float)
        nll += float((np.logaddexp(0, z) - y * z).sum())
        n += len(y)
        ys.append(y)
        ss.append(z)
    y, s = np.concatenate(ys), np.concatenate(ss)
    # precision at the recall levels 0.5 / 0.7 / 0.85 of the pooled held-out scores
    order = np.argsort(-s)
    tp = np.cumsum(y[order])
    prec = {}
    for rec in (0.5, 0.7, 0.85):
        k = int(np.searchsorted(tp, rec * y.sum()))
        prec[rec] = tp[k] / (k + 1)
    w = np.array(ws)
    print(f"{name:34s} held-out logloss {nll / n:.4f}  AUC {auc(y, s):.4f}  prec@rec .5/.7/.85 {prec[0.5]:.3f}/{prec[0.7]:.3f}/{prec[0.85]:.3f}"
          f"  w mean {np.round(w.mean(axis=0), 2).tolist()} sd {np.round(w.std(axis=0), 2).tolist()}")
