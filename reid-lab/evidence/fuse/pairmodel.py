"""fuse / pairmodel.py -- the pairwise fusion model: P(a and b are one person | evidence).

apply (label-free): design(df, cols) and logit(df, model).
fit (labels; fit clips of a LOCO fold only): fit(df_labelled, cols) -> model dict.

A plain logistic regression (L2 on the standardised weights, scipy L-BFGS). Training rows: pairs of
sub-pieces whose identities are both known (>= 5 matched rows, score.py's rule); positive = same
person and a is the latest earlier piece of that person before b ("immediate"); negative =
different people; same-person pairs that are not immediate are left out (their position evidence
is that of a pair with something in between, and the linker only reads position at the junctions
of a chain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

L2 = 1e-3
SINK_CLIP = 0.98


def design(df: pd.DataFrame, cols) -> np.ndarray:
    """Model inputs by name."""
    sink = df.pos_sink.to_numpy(float)
    lo = np.log(np.clip(sink, 1e-3, SINK_CLIP) / (1 - np.clip(sink, 1e-3, SINK_CLIP)))
    kind = df.kind.to_numpy(float)
    low = df.low.to_numpy(float)
    cw = df.cos_w.to_numpy(float)
    f = {
        "cos_w": cw,
        "cos_tc": df.cos_tc.to_numpy(float),
        "cos_raw": df.cos_raw.to_numpy(float),
        "low": low,
        "cos_w_low": cw * low,
        "unreach": df.pos_unreach.to_numpy(float),
        "sink": sink,
        "sink_edge": sink * (kind >= 1),
        "sink_in": sink * (kind == 0),
        "sink_lo": (lo - lo.min()) * 0 + (lo + np.log((1 - 1e-3) / 1e-3)),      # >= 0, 0 when sink <= 0.001
        "pos_p": np.minimum(df.pos_p_in.to_numpy(float), df.pos_p_out.to_numpy(float)),
        "plug": df.plug.to_numpy(float),
        "lgap": np.log1p(df.gap.to_numpy(float)),
        "kind1": (kind >= 1).astype(float),
    }
    return np.column_stack([f[c] for c in cols])


def fit(df: pd.DataFrame, cols, l2: float = L2) -> dict:
    use = df.known & (df.imm | ~df.same)
    d = df[use]
    X = design(d, cols)
    y = d.imm.to_numpy(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = (X - mu) / sd

    def nll(q):
        w, b = q[:-1], q[-1]
        z = Z @ w + b
        loss = np.logaddexp(0.0, z) - y * z
        p = 1.0 / (1.0 + np.exp(-z))
        g = Z.T @ (p - y) / len(y) + 2 * l2 * w
        return loss.mean() + l2 * float(w @ w), np.r_[g, (p - y).mean()]

    res = optimize.minimize(nll, np.zeros(len(cols) + 1), jac=True, method="L-BFGS-B")
    w = res.x[:-1] / sd
    b = float(res.x[-1] - (res.x[:-1] * mu / sd).sum())
    return {"cols": list(cols), "w": [float(v) for v in w], "b": b, "n": int(len(y)), "pos": int(y.sum()), "nll": float(res.fun)}


def logit(df: pd.DataFrame, model: dict) -> np.ndarray:
    return design(df, model["cols"]) @ np.asarray(model["w"]) + model["b"]
