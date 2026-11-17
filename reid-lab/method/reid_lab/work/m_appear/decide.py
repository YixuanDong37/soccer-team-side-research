"""Label-free decision rules: candidate pairs with a score -> links between pieces.

All rules take the pair table of one clip (features.pair_features) and return [[a, b], ...].
"""
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def mutual(df, kind="tc", t_abs=-1.0, margin=0.05, need_runner_up=True, margin_kind="marg"):
    """appearance_match.py's rule: mutual best, cosine >= t_abs, best beats the runner-up on both sides by margin.
    margin_kind "rival": only runner-ups seen at the same time as the partner count (others may be the same person)."""
    s = df[f"cos_{kind}"].to_numpy()
    keep = (df[f"rank_fwd_{kind}"].to_numpy() == 1) & (df[f"rank_bwd_{kind}"].to_numpy() == 1) & (s >= t_abs)
    mf, mb = df[f"{margin_kind}_fwd_{kind}"].to_numpy(), df[f"{margin_kind}_bwd_{kind}"].to_numpy()
    if need_runner_up:
        keep &= ~np.isnan(mf) & ~np.isnan(mb)
        keep &= (mf >= margin) & (mb >= margin)
    else:
        keep &= (np.isnan(mf) | (mf >= margin)) & (np.isnan(mb) | (mb >= margin))
    return df.loc[keep, ["a", "b"]].to_numpy().tolist()


def assign(df, score, t_abs):
    """One successor and one predecessor per piece (chains), maximising the sum of (score - t_abs) over links.
    score: array aligned with df. Returns the chosen pairs' positions in df."""
    s = np.asarray(score, float)
    use = np.nonzero(s > t_abs)[0]
    if not len(use):
        return np.zeros(0, int)
    a_ids, a_inv = np.unique(df.a.to_numpy()[use], return_inverse=True)
    b_ids, b_inv = np.unique(df.b.to_numpy()[use], return_inverse=True)
    na, nb = len(a_ids), len(b_ids)
    # square problem with a private "no link" column per a and row per b
    big = 1e6
    C = np.full((na + nb, nb + na), big)
    C[a_inv, b_inv] = -(s[use] - t_abs)
    C[np.arange(na), nb + np.arange(na)] = 0.0          # a without successor
    C[na + np.arange(nb), np.arange(nb)] = 0.0          # b without predecessor
    C[na:, nb:] = 0.0
    r, c = linear_sum_assignment(C)
    pos = {(int(i), int(j)): int(k) for i, j, k in zip(a_inv, b_inv, use)}
    out = [pos[(int(i), int(j))] for i, j in zip(r, c) if i < na and j < nb and (int(i), int(j)) in pos]
    return np.array(sorted(out), int)
