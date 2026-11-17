"""(5c) Appearance, exploratory: label-free normalisations of the embedding space on top of
team-centring (diagnosis; needs out/t5b_piece_pairs.parquet from t5b_appearance_pieces.py).

- teamcentred: subtract the mean embedding of the rows with the same gbf team label (clip mean
  for rows without a label), renormalise, average over the piece's rows.
- wccn(lam): additionally whiten with the within-piece covariance (rows minus their piece mean,
  pooled over the clip's pieces; pieces are the tracker's, so no labels), shrunk towards
  lam * mean eigenvalue. Directions that vary inside one track (pose, scale, blur) count less.
- top-k PCs of the within-piece variation projected out instead of whitened.
Parameters here (lam, k) are looked at in-sample on the 13 clips: a hint, not a fitted result.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import common as C  # noqa: E402

pd.set_option("display.width", 250)
data = C.data
PP = pd.read_parquet(C.OUT / "t5b_piece_pairs.parquet")


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


LAMS = (0.1, 0.3, 1.0, 3.0)
KS = (1, 3, 10)
cols = {**{f"wccn_{lam}": [] for lam in LAMS}, **{f"projout_{k}": [] for k in KS}, "teamcentred_check": [], "wccn_0.3_raw(no centring)": []}
keys = []
for seq in C.CLIPS:
    t = data.tracks(seq)
    emb = data.embeddings(seq)
    has = ~np.isnan(emb[:, 0])
    lab = t.team.to_numpy(object)
    tid = t.track_id.to_numpy()
    mu_clip = emb[has].mean(0)
    cen = emb - mu_clip
    for name in ("left", "right"):
        sel = has & (lab == name)
        if sel.any():
            cen[lab == name] = emb[lab == name] - emb[sel].mean(0)
    cen = unit(cen)
    use = has & (tid >= 0)
    ids, inv = np.unique(tid[use], return_inverse=True)

    def within(X):
        means = np.zeros((len(ids), X.shape[1]))
        np.add.at(means, inv, X[use])
        cnt = np.bincount(inv, minlength=len(ids))[:, None]
        means /= cnt
        resid = X[use] - means[inv]
        return means, resid.T @ resid / max(len(resid) - len(ids), 1)

    means_c, Sw = within(cen)
    ev, U = np.linalg.eigh(Sw)
    pos = {int(p): k for k, p in enumerate(ids)}
    protos = {"teamcentred_check": unit(means_c)}
    for lam in LAMS:
        Wm = U @ np.diag(1.0 / np.sqrt(ev.clip(min=0) + lam * ev.mean())) @ U.T
        protos[f"wccn_{lam}"] = unit(means_c @ Wm)
    for k in KS:
        top = U[:, -k:]
        protos[f"projout_{k}"] = unit(means_c - (means_c @ top) @ top.T)
    means_r, Sw_r = within(np.where(has[:, None], emb, 0.0))
    evr, Ur = np.linalg.eigh(Sw_r)
    Wr = Ur @ np.diag(1.0 / np.sqrt(evr.clip(min=0) + 0.3 * evr.mean())) @ Ur.T
    protos["wccn_0.3_raw(no centring)"] = unit(means_r @ Wr)
    g = PP[PP.seq == seq]
    ia = np.array([pos[int(a)] for a in g.a])
    ib = np.array([pos[int(b)] for b in g.b])
    for name, M in protos.items():
        cols[name].append(np.einsum("ij,ij->i", M[ia], M[ib]))
    keys.append(g.index.to_numpy())
order = np.concatenate(keys)
for name in cols:
    PP.loc[order, name] = np.concatenate(cols[name])

same, hard, oth = PP[PP.kind == "same"], PP[PP.kind == "hard"], PP[PP.kind == "other_team"]
tabs = []
for v in ["mean_all", "mean_all_teamcentred", "teamcentred_check", *[f"wccn_{lam}" for lam in LAMS], *[f"projout_{k}" for k in KS], "wccn_0.3_raw(no centring)"]:
    pr = C.prec_at_recall(same[v], hard[v])
    hv = hard[v].to_numpy()
    hit = n = 0
    for (_, b), g in PP[PP.team_lab_ok].groupby(["seq", "b"]):
        if (g.kind == "same").any():
            n += 1
            hit += g.loc[g[v].idxmax(), "kind"] == "same"
    per = [C.auc(same[same.seq == q][v], hard[hard.seq == q][v]) for q in C.CLIPS]
    tabs.append({"prototype": v, "AUC same vs hard": C.auc(same[v], hard[v]), "AUC same vs all": C.auc(same[v], pd.concat([hard[v], oth[v]])), "prec@rec0.5": pr[0.5][0],
                 "prec@rec0.8": pr[0.8][0], "prec@rec0.9": pr[0.9][0], "TPR@FPR1%": float((same[v] > np.quantile(hv, 0.99)).mean()),
                 "TPR@FPR0.1%": float((same[v] > np.quantile(hv, 0.999)).mean()), "rank-1": hit / n, "worst clip AUC": float(np.min(per))})  # fmt: skip
T = pd.DataFrame(tabs)
print(T.round(3).to_string())
T.to_csv(C.OUT / "t5c_wccn.csv", index=False)
PP.to_parquet(C.OUT / "t5c_piece_pairs.parquet")
