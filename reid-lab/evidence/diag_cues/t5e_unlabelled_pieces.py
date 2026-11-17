"""(5e) Team-centring when a piece has no team label (gbf clears the label of re-entry pieces it
could not check: 18 % of pieces with k >= 1; one piece of 21 of the 137 scored returns).

Three ways to centre the rows of an unlabelled piece before averaging (diagnosis; labels only
name the pairs):
  clip      subtract the clip's mean embedding (what t5b's mean_all_teamcentred does)
  nearest   give the piece the team whose mean embedding is nearer to the piece's raw mean, then
            subtract that team's mean (label-free team guess from appearance)
  partner   pair-specific: centre the unlabelled piece with the team mean of the piece it is
            compared with (when that one is labelled)
Reported on time-disjoint pairs of pieces with identity where at least one piece is unlabelled,
and, for reference, on pairs where both are labelled.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import common as C  # noqa: E402

pd.set_option("display.width", 250)
data = C.data


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


rows = []
guess = []
for seq in C.CLIPS:
    t = data.tracks(seq)
    emb = data.embeddings(seq)
    has = ~np.isnan(emb[:, 0])
    lab = t.team.to_numpy(object)
    mu = {"clip": emb[has].mean(0)}
    for name in ("left", "right"):
        sel = has & (lab == name)
        mu[name] = emb[sel].mean(0) if sel.any() else mu["clip"]
    P = C.piece_ident(seq)
    P = P[P.ref_id.notna() & P.role_ref.isin(["player", "goalkeeper"])].reset_index(drop=True)
    idx_of = {int(k): np.asarray(v) for k, v in t.groupby("track_id").indices.items() if k >= 0}
    E, raw = {}, {}
    for r in P.itertuples():
        idx = idx_of[r.piece]
        idx = idx[has[idx]]
        E[r.piece] = emb[idx]
        raw[r.piece] = unit(emb[idx].mean(0))

    def proto(piece, centre):
        return unit(unit(E[piece] - mu[centre]).mean(0))

    near = {}
    for r in P.itertuples():
        near[r.piece] = "left" if raw[r.piece] @ unit(mu["left"]) >= raw[r.piece] @ unit(mu["right"]) else "right"
        if pd.isna(r.team):
            # the annotation's team name may be swapped against gbf's in a clip: compare through labelled pieces
            guess.append({"seq": seq, "piece": r.piece, "nearest": near[r.piece], "team_ref": r.team_ref, "rows": r.rows})
        else:
            guess.append({"seq": seq, "piece": r.piece, "nearest": near[r.piece], "team_gbf": r.team, "rows": r.rows})
    first, last, rid, tm, tr = P["first"].to_numpy(), P["last"].to_numpy(), P.ref_id.to_numpy(), P.team.to_numpy(object), P.team_ref.to_numpy(object)
    pc = P.piece.to_numpy()
    for i in range(len(P)):
        for j in range(len(P)):
            if not last[i] < first[j]:
                continue
            li, lj = not pd.isna(tm[i]), not pd.isna(tm[j])
            kind = "same" if rid[i] == rid[j] else ("hard" if tr[i] == tr[j] else "other_team")
            ci = tm[i] if li else "clip"
            cj = tm[j] if lj else "clip"
            ni = tm[i] if li else near[pc[i]]
            nj = tm[j] if lj else near[pc[j]]
            pi_ = tm[i] if li else (tm[j] if lj else near[pc[i]])
            pj_ = tm[j] if lj else (tm[i] if li else near[pc[j]])
            rows.append({"seq": seq, "kind": kind, "labelled": int(li) + int(lj), "raw": float(raw[pc[i]] @ raw[pc[j]]),
                         "clip": float(proto(pc[i], ci) @ proto(pc[j], cj)), "nearest": float(proto(pc[i], ni) @ proto(pc[j], nj)),
                         "partner": float(proto(pc[i], pi_) @ proto(pc[j], pj_))})  # fmt: skip
D = pd.DataFrame(rows)
G = pd.DataFrame(guess)
lab_ok = G[G.team_gbf.notna()] if "team_gbf" in G else G.iloc[:0]
print("team guessed from the nearer team mean embedding, on pieces that DO have a gbf label: agrees with the label in",
      int((lab_ok.nearest == lab_ok.team_gbf).sum()), "of", len(lab_ok), f"({(lab_ok.nearest == lab_ok.team_gbf).mean():.3f})")  # fmt: skip
print("unlabelled pieces with identity:", int(G.team_gbf.isna().sum()) if "team_gbf" in G else 0)
tabs = []
for name, g in (("both pieces labelled", D[D.labelled == 2]), ("one or both unlabelled", D[D.labelled < 2])):
    s, hd = g[g.kind == "same"], g[g.kind == "hard"]
    for col in ("raw", "clip", "nearest", "partner"):
        pr = C.prec_at_recall(s[col], hd[col])
        tabs.append({"pairs": name, "centring of unlabelled pieces": col, "same": len(s), "hard": len(hd), "cos same p10": s[col].quantile(0.1), "cos same p50": s[col].median(),
                     "cos hard p50": hd[col].median(), "cos hard p99": hd[col].quantile(0.99), "AUC same vs hard": C.auc(s[col], hd[col]), "prec@rec0.8": pr[0.8][0]})  # fmt: skip
T = pd.DataFrame(tabs)
print(T.round(3).to_string())
T.to_csv(C.OUT / "t5e_unlabelled.csv", index=False)
