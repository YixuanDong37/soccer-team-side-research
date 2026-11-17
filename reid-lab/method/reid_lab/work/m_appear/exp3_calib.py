"""EXPLORATION with labels inside LOCO: which pair features calibrate the same-person score best.

A logistic model is fitted on the candidate pairs of 12 clips (labels from fitlabels: both pieces with a
known identity) and applied to the held-out clip; the held-out scores of all 13 clips are pooled.
Reported: average precision and AUC over (a) all pairs, (b) the pairs that are the best earlier candidate
of their later piece under that score ("top-1 decisions").
"""
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reid_lab import data  # noqa: E402

import fitlabels  # noqa: E402
from diag2 import ap, auc  # noqa: E402
from features import pair_features  # noqa: E402


def fit_logistic(X, y, w=None, l2=1e-3, iters=50):
    """Newton / IRLS with a small ridge; X without intercept column. Returns (coef, intercept)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.ones(len(y)) if w is None else w
    beta = np.zeros(Xb.shape[1])
    R = l2 * np.eye(Xb.shape[1])
    R[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        g = Xb.T @ (w * (p - y)) + R @ beta
        Hm = (Xb * (w * p * (1 - p))[:, None]).T @ Xb + R
        step = np.linalg.solve(Hm + 1e-9 * np.eye(len(beta)), g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta[:-1], beta[-1]


def table():
    out = []
    for s in data.DEV_CLIPS:
        d = pair_features(s).copy()
        ident = fitlabels.piece_identity(s)
        d["seq"] = s
        d["y"] = [np.nan if (int(a) not in ident or int(b) not in ident) else float(ident[int(a)][0] == ident[int(b)][0]) for a, b in zip(d.a, d.b)]
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d["lmin_rows"] = np.log10(np.minimum(d.rows_a, d.rows_b))
    d["lmin_clean"] = np.log10(1 + np.minimum(d.clean_a, d.clean_b))
    d["low_any"] = (d.low_a | d.low_b).astype(float)
    d["lgap"] = np.log10(d.gap)
    d["lminh"] = np.log10(np.minimum(d.medh_a, d.medh_b))
    d["lhratio"] = np.abs(np.log(d.medh_a / d.medh_b))
    d["same_plugin_f"] = d.same_plugin.astype(float)
    for k in ("raw", "tc", "w"):
        d[f"impmax_{k}"] = d[[f"imp_a_{k}", f"imp_b_{k}"]].max(axis=1).fillna(d[f"cos_{k}"].median())
        d[f"overimp_{k}"] = d[f"cos_{k}"] - d[f"impmax_{k}"]
        d[f"zn_{k}f"] = d[f"zn_{k}"].fillna(d[f"zn_{k}"].median())
    return d


def loco_scores(d, feats, hard_only=False):
    s = np.full(len(d), np.nan)
    lab = d.y.notna().to_numpy()
    for fit, held in data.loco_folds():
        tr = d.seq.isin(fit).to_numpy() & lab
        if hard_only:
            tr &= ((d.rank_fwd_tc <= 3) | (d.rank_bwd_tc <= 3)).to_numpy()
        X = d.loc[tr, feats].to_numpy(float)
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        coef, b = fit_logistic((X - mu) / sd, d.y[tr].to_numpy())
        te = (d.seq == held).to_numpy()
        s[te] = ((d.loc[te, feats].to_numpy(float) - mu) / sd) @ coef + b
    return s


def report(d, s, tag):
    lab = d.y.notna().to_numpy()
    y = d.y.to_numpy()
    a_all = auc(s[lab & (y == 1)], s[lab & (y == 0)])
    ap_all = ap(s[lab], y[lab] == 1)
    # top-1 decisions: best earlier candidate of every later piece
    dd = d.assign(s=s)[lab]
    top = dd.loc[dd.groupby(["seq", "b"]).s.idxmax()]
    o = np.argsort(-top.s.to_numpy())
    g = (top.y.to_numpy() == 1)[o]
    fp, tp = np.cumsum(~g), np.cumsum(g)
    at = {k: int(tp[fp <= k].max()) if (fp <= k).any() else 0 for k in (2, 5, 10, 20)}
    # the same without SNGS-162 (suspected annotation swaps)
    t2 = top[top.seq != "SNGS-162"]
    o2 = np.argsort(-t2.s.to_numpy())
    g2 = (t2.y.to_numpy() == 1)[o2]
    fp2, tp2 = np.cumsum(~g2), np.cumsum(g2)
    at2 = {k: int(tp2[fp2 <= k].max()) if (fp2 <= k).any() else 0 for k in (0, 2, 5, 10)}
    print(f"{tag:60s} all: AUC {a_all:.4f} AP {ap_all:.4f} | top1: AP {ap(top.s.to_numpy(), top.y.to_numpy() == 1):.4f} TP@FP<=2/5/10/20 {at[2]:3d} {at[5]:3d} {at[10]:3d} {at[20]:3d}"
          f" | w/o 162: AP {ap(t2.s.to_numpy(), t2.y.to_numpy() == 1):.4f} TP@FP<=0/2/5/10 {at2[0]:3d} {at2[2]:3d} {at2[5]:3d} {at2[10]:3d}", flush=True)


if __name__ == "__main__":
    d = table()
    print("pairs", len(d), "labelled", int(d.y.notna().sum()), "positive", int((d.y == 1).sum()))
    for k in ("cos_raw", "cos_tc", "cos_w", "zn_tcf", "zn_wf", "zn_rawf", "overimp_tc", "overimp_w", "overimp_raw"):
        report(d, d[k].to_numpy(), f"single: {k}")
    sets = {
        "tc": ["cos_tc"],
        "tc+raw": ["cos_tc", "cos_raw"],
        "tc+w": ["cos_tc", "cos_w"],
        "tc+raw+w": ["cos_tc", "cos_raw", "cos_w"],
        "tc+lmin_rows": ["cos_tc", "lmin_rows"],
        "tc+lmin_clean": ["cos_tc", "lmin_clean"],
        "tc+low_any": ["cos_tc", "low_any"],
        "tc+lgap": ["cos_tc", "lgap"],
        "tc+lminh": ["cos_tc", "lminh"],
        "tc+lhratio": ["cos_tc", "lhratio"],
        "tc+same_plugin": ["cos_tc", "same_plugin_f"],
        "tc+zn_tc": ["cos_tc", "zn_tcf"],
        "tc+overimp_tc": ["cos_tc", "overimp_tc"],
        "tc+raw+lmin_clean": ["cos_tc", "cos_raw", "lmin_clean"],
        "tc+raw+w+lmin_clean+lgap": ["cos_tc", "cos_raw", "cos_w", "lmin_clean", "lgap"],
        "tc+raw+w+lmin_clean+lgap+same_plugin": ["cos_tc", "cos_raw", "cos_w", "lmin_clean", "lgap", "same_plugin_f"],
        "tc+raw+w+lmin_clean+lgap+lhratio+zn+overimp": ["cos_tc", "cos_raw", "cos_w", "lmin_clean", "lgap", "lhratio", "zn_tcf", "overimp_tc"],
    }
    for name, feats in sets.items():
        report(d, loco_scores(d, feats), f"LOCO logistic: {name}")
    for name in ("tc+raw", "tc+raw+w+lmin_clean+lgap"):
        report(d, loco_scores(d, sets[name], hard_only=True), f"LOCO logistic (fit on top-3 pairs): {name}")
