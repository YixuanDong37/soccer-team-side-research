"""(5b) Appearance, piece level: cosine of piece prototypes for same-person pairs against
different-person same-team pairs and different-team pairs, for many prototype choices
(diagnosis: the annotation labels the pairs; every prototype is built label-free).

Pairs: an earlier piece A and a later piece B of one clip (A's last frame < B's first frame),
both with a known identity (majority annotated person, >= 5 matched rows) of a player or
goalkeeper. same = one person; hard = different persons of one annotated team; other team.

Per prototype choice: AUC (same vs hard, same vs all different), precision at recall 0.5 / 0.8 /
0.9 in the pool same + hard (natural prevalence), TPR at 1 % and 0.1 % hard false positives, and
rank-1: for each later piece with an earlier piece of its person, is the most similar earlier
piece (same gbf team label or no label) that person.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import common as C  # noqa: E402

pd.set_option("display.width", 280)
pd.set_option("display.max_columns", 60)
data = C.data
rng = np.random.default_rng(0)
res: dict = {}


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def kmeans_protos(X, k=3, iters=8):
    """Spherical k-means prototypes (deterministic farthest-first start)."""
    if len(X) <= k:
        return unit(X)
    c = [unit(X.mean(0))]
    for _ in range(k - 1):
        s = np.max(np.stack([X @ ci for ci in c]), axis=0)
        c.append(X[int(np.argmin(s))])
    c = np.stack(c)
    for _ in range(iters):
        lab = np.argmax(X @ c.T, axis=1)
        c = np.stack([unit(X[lab == j].mean(0)) if (lab == j).any() else c[j] for j in range(k)])
    return c


def clip_tables(seq):
    t = data.tracks(seq)
    emb = data.embeddings(seq)
    has = ~np.isnan(emb[:, 0])
    clean = data.clean_rows(seq)
    x1, y1, x2, y2 = (t[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
    h = y2 - y1
    near = (x1 <= 10) | (y1 <= 10) | (x2 >= C.W - 11) | (y2 >= C.H - 11)
    cut = (x1 <= 1) | (y1 <= 1) | (x2 >= C.W - 2) | (y2 >= C.H - 2)
    overlap = np.zeros(len(t), bool)
    for _, idx in t.groupby("frame").indices.items():
        if len(idx) < 2:
            continue
        ix = np.minimum(x2[idx][:, None], x2[idx][None]) - np.maximum(x1[idx][:, None], x1[idx][None])
        iy = np.minimum(y2[idx][:, None], y2[idx][None]) - np.maximum(y1[idx][:, None], y1[idx][None])
        hit = (ix > 0) & (iy > 0)
        np.fill_diagonal(hit, False)
        overlap[idx] = hit.any(axis=1)
    return t, emb, has, clean, h, near, cut, overlap


SINGLE = ["mean_all", "mean_clean", "mean_noedge10", "mean_clean_h60", "hweight_all", "hweight_clean", "confweight_clean", "median_all", "median_clean",
          "tallest10_clean", "mean_clean_clipcentred", "mean_clean_teamcentred", "mean_all_teamcentred", "median_all_teamcentred",
          "mean_all_teamwhitened"]  # fmt: skip
MULTI = ["kmeans3_clean_max", "chunks25_clean_max", "chunks25_clean_top3", "rows_clean_q90", "rows_clean_q50", "rows_clean_max", "ends25_clean"]
NORMED = ["mean_clean_snorm", "mean_all_teamcentred_snorm"]
VARIANTS = SINGLE + MULTI + NORMED

pair_rows = []
piece_rows = []
subset_rows = []   # number-of-rows experiment
for seq in C.CLIPS:
    t, emb, has, clean, h, near, cut, overlap = clip_tables(seq)
    conf = t.conf.to_numpy(float)
    frames = t.frame.to_numpy()
    team_lab = t.team.to_numpy(object)
    P = C.piece_ident(seq)
    idx_of = {int(k): np.asarray(v) for k, v in t.groupby("track_id").indices.items() if k >= 0}
    mu_clip = emb[has].mean(0)
    mu_team = {}
    for lab in ("left", "right"):
        sel = has & (team_lab == lab)
        mu_team[lab] = emb[sel].mean(0) if sel.any() else mu_clip
    cen_clip = unit(emb - mu_clip)
    cen_team = emb.copy()
    for lab in ("left", "right"):
        sel = team_lab == lab
        cen_team[sel] = emb[sel] - mu_team[lab]
    other = ~np.isin(team_lab, ["left", "right"])
    cen_team[other] = emb[other] - mu_clip
    cen_team = unit(cen_team)
    # within-clip whitening of the team-centred rows (label-free): covariance shrunk towards its mean eigenvalue
    Xc = cen_team[has]
    cov = np.cov(Xc, rowvar=False)
    ev, U = np.linalg.eigh(cov)
    lam = 0.5 * ev.mean()
    Wm = U @ np.diag(1.0 / np.sqrt(ev.clip(min=0) + lam)) @ U.T
    white = unit((cen_team - Xc.mean(0)) @ Wm)

    pieces = []
    protos = {v: [] for v in SINGLE}
    multi = {v: [] for v in MULTI}
    for r in P.itertuples():
        idx = idx_of[r.piece]
        idx = idx[has[idx]]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(frames[idx])]
        cl = idx[clean[idx]]
        base = cl if len(cl) else idx                    # clean rows, else every row
        ne = idx[~near[idx]]
        ne = ne if len(ne) else idx
        ch60 = cl[h[cl] >= 60]
        ch60 = ch60 if len(ch60) else base
        tall = base[np.argsort(-h[base])[:10]]
        pieces.append({"seq": seq, "piece": r.piece, "first": r.first, "last": r.last, "rows": len(idx), "n_clean": len(cl), "ref_id": r.ref_id, "role_ref": r.role_ref,
                       "team_ref": r.team_ref, "team": r.team, "purity": r.purity, "h_med": float(np.median(h[base])), "near_share": float(near[idx].mean()),
                       "cut_share": float(cut[idx].mean()), "overlap_share": float(overlap[idx].mean()), "starts_at_edge": r.starts_at_edge, "ends_at_edge": r.ends_at_edge})  # fmt: skip
        protos["mean_all"].append(unit(emb[idx].mean(0)))
        protos["mean_clean"].append(unit(emb[base].mean(0)))
        protos["mean_noedge10"].append(unit(emb[ne].mean(0)))
        protos["mean_clean_h60"].append(unit(emb[ch60].mean(0)))
        protos["hweight_all"].append(unit((emb[idx] * h[idx, None]).sum(0)))
        protos["hweight_clean"].append(unit((emb[base] * h[base, None]).sum(0)))
        protos["confweight_clean"].append(unit((emb[base] * conf[base, None]).sum(0)))
        protos["median_all"].append(unit(np.median(emb[idx], axis=0)))
        protos["median_clean"].append(unit(np.median(emb[base], axis=0)))
        protos["tallest10_clean"].append(unit(emb[tall].mean(0)))
        protos["mean_clean_clipcentred"].append(unit(cen_clip[base].mean(0)))
        protos["mean_clean_teamcentred"].append(unit(cen_team[base].mean(0)))
        protos["mean_all_teamcentred"].append(unit(cen_team[idx].mean(0)))
        protos["median_all_teamcentred"].append(unit(np.median(cen_team[idx], axis=0)))
        protos["mean_all_teamwhitened"].append(unit(white[idx].mean(0)))
        multi["kmeans3_clean_max"].append(kmeans_protos(emb[base]))
        chunk = (frames[base] - frames[base][0]) // 25
        ck = np.stack([unit(emb[base[chunk == c]].mean(0)) for c in np.unique(chunk)])
        multi["chunks25_clean_max"].append(ck)
        multi["chunks25_clean_top3"].append(ck)
        sub = base[np.linspace(0, len(base) - 1, min(len(base), 40)).round().astype(int)]
        for v in ("rows_clean_q90", "rows_clean_q50", "rows_clean_max"):
            multi[v].append(emb[sub])
        multi["ends25_clean"].append((unit(emb[base[:25]].mean(0)), unit(emb[base[-25:]].mean(0))))  # (head, tail)
    PD = pd.DataFrame(pieces)
    piece_rows.append(PD)
    n = len(PD)
    S = {v: np.stack(protos[v]) @ np.stack(protos[v]).T for v in SINGLE}

    def agg_multi(v, i, j):
        A, B = multi[v][i], multi[v][j]
        if v == "ends25_clean":
            return float(A[1] @ B[0])  # tail of the earlier piece against head of the later one
        M = A @ B.T
        if v.endswith("_max"):
            return float(M.max())
        if v.endswith("_top3"):
            f = np.sort(M.ravel())[::-1]
            return float(f[: min(3, len(f))].mean())
        return float(np.quantile(M, 0.9 if v.endswith("q90") else 0.5))

    # s-norm of mean_clean: cohort of piece X = pieces seen together with X (surely other persons), same gbf team label when >= 3 such
    first, last = PD["first"].to_numpy(), PD["last"].to_numpy()
    covis = (first[:, None] <= last[None, :]) & (first[None, :] <= last[:, None])
    np.fill_diagonal(covis, False)
    tm = PD.team.to_numpy(object)
    same_lab = np.array([[tm[i] == tm[j] and tm[i] in ("left", "right") for j in range(n)] for i in range(n)])
    Sm = S["mean_clean"]
    St = S["mean_all_teamcentred"]
    mu, sd = np.zeros(n), np.ones(n)
    mut, sdt = np.zeros(n), np.ones(n)
    for i in range(n):
        co = covis[i] & same_lab[i]
        if co.sum() < 3:
            co = covis[i]
        if co.sum() >= 3:
            mu[i], sd[i] = Sm[i, co].mean(), max(Sm[i, co].std(), 1e-3)
            mut[i], sdt[i] = St[i, co].mean(), max(St[i, co].std(), 1e-3)
        else:
            mu[i], sd[i] = np.nan, np.nan
            mut[i], sdt[i] = np.nan, np.nan
    rid = PD.ref_id.to_numpy()
    known = PD.ref_id.notna().to_numpy() & PD.role_ref.isin(["player", "goalkeeper"]).to_numpy()
    tr = PD.team_ref.to_numpy(object)
    for i in range(n):
        for j in range(n):
            if not (last[i] < first[j]):
                continue
            if not (known[i] and known[j]):
                continue
            kind = "same" if rid[i] == rid[j] else ("hard" if tr[i] == tr[j] else "other_team")
            row = {"seq": seq, "a": int(PD.piece.iat[i]), "b": int(PD.piece.iat[j]), "kind": kind, "gap_s": (first[j] - last[i]) / C.FPS,
                   "n_clean_min": int(min(PD.n_clean.iat[i], PD.n_clean.iat[j])), "rows_min": int(min(PD.rows.iat[i], PD.rows.iat[j])),
                   "h_min": float(min(PD.h_med.iat[i], PD.h_med.iat[j])), "near_max": float(max(PD.near_share.iat[i], PD.near_share.iat[j])),
                   "overlap_max": float(max(PD.overlap_share.iat[i], PD.overlap_share.iat[j])), "purity_min": float(min(PD.purity.iat[i], PD.purity.iat[j])),
                   "team_lab_ok": bool(pd.isna(tm[i]) or pd.isna(tm[j]) or tm[i] == tm[j]),
                   "edge_pair": bool(PD.ends_at_edge.iat[i] or PD.starts_at_edge.iat[j])}  # fmt: skip
            for v in SINGLE:
                row[v] = float(S[v][i, j])
            for v in MULTI:
                row[v] = agg_multi(v, i, j)
            row["mean_clean_snorm"] = 0.5 * ((Sm[i, j] - mu[i]) / sd[i] + (Sm[i, j] - mu[j]) / sd[j])
            row["mean_all_teamcentred_snorm"] = 0.5 * ((St[i, j] - mut[i]) / sdt[i] + (St[i, j] - mut[j]) / sdt[j])
            pair_rows.append(row)
    # number of rows averaged: random subsets of clean rows for pieces with >= 30 clean rows
    big = [k for k in range(n) if PD.n_clean.iat[k] >= 30 and known[k]]
    for nsub in (1, 3, 10, 30):
        pr = []
        for k in big:
            idx = idx_of[int(PD.piece.iat[k])]
            idx = idx[has[idx] & clean[idx]]
            pr.append(unit(emb[rng.choice(idx, nsub, replace=False)].mean(0)))
        if len(big) < 2:
            continue
        Sx = np.stack(pr) @ np.stack(pr).T
        for a_, i in enumerate(big):
            for b_, j in enumerate(big):
                if last[i] < first[j]:
                    kind = "same" if rid[i] == rid[j] else ("hard" if tr[i] == tr[j] else "other_team")
                    subset_rows.append({"seq": seq, "n": nsub, "kind": kind, "cos": float(Sx[a_, b_]), "full": float(S["mean_clean"][i, j])})

PP = pd.DataFrame(pair_rows)
PCS = pd.concat(piece_rows, ignore_index=True)
PP.to_parquet(C.OUT / "t5b_piece_pairs.parquet")
PCS.to_csv(C.OUT / "t5b_pieces.csv", index=False)
print("pieces with an embedding:", len(PCS), "; with identity (player/GK):", int((PCS.ref_id.notna() & PCS.role_ref.isin(["player", "goalkeeper"])).sum()),
      "; with >= 5 clean rows:", int((PCS.n_clean >= 5).sum()), "; with no clean row:", int((PCS.n_clean == 0).sum()))  # fmt: skip
print("pairs:", PP.groupby("kind").size().to_dict())
same, hard, oth = PP[PP.kind == "same"], PP[PP.kind == "hard"], PP[PP.kind == "other_team"]


def metrics(v, s, hd, ot):
    pr = C.prec_at_recall(s[v], hd[v])
    hv = hd[v].dropna().to_numpy()
    return {"prototype": v, "AUC same vs hard": C.auc(s[v], hd[v]), "AUC same vs all": C.auc(s[v], pd.concat([hd[v], ot[v]])),
            "prec@rec0.5": pr[0.5][0], "prec@rec0.8": pr[0.8][0], "prec@rec0.9": pr[0.9][0],
            "TPR@FPR1%": float((s[v] > np.quantile(hv, 0.99)).mean()), "TPR@FPR0.1%": float((s[v] > np.quantile(hv, 0.999)).mean()),
            "same p10": float(s[v].quantile(0.1)), "same p50": float(s[v].median()), "hard p50": float(hd[v].median()), "hard p99": float(np.quantile(hv, 0.99))}  # fmt: skip


def rank1(v, pool):
    """For each later piece with an earlier piece of its person: is the most similar earlier piece (gbf team label equal or missing) that person?"""
    hit = n = 0
    mutual = 0
    for (seq, b), g in pool[pool.team_lab_ok].groupby(["seq", "b"]):
        if not (g.kind == "same").any():
            continue
        g = g.dropna(subset=[v])
        if not len(g) or not (g.kind == "same").any():
            continue
        n += 1
        hit += g.loc[g[v].idxmax(), "kind"] == "same"
    return hit / max(n, 1), n


print("\n=== prototype choices, all pairs ===")
tabs = []
for v in VARIANTS:
    m = metrics(v, same, hard, oth)
    m["rank-1"], m["rank-1 n"] = rank1(v, PP)
    tabs.append(m)
T = pd.DataFrame(tabs)
print(T.round(3).to_string())
T.to_csv(C.OUT / "t5b_prototypes_all_pairs.csv", index=False)
res["prototypes_all_pairs"] = T.round(4).to_dict("records")

print("\n=== prototype choices, pairs next to an image edge (A ends or B starts at the edge), gap >= 0.4 s ===")
E = PP[PP.edge_pair & (PP.gap_s >= 0.4)]
es, eh, eo = E[E.kind == "same"], E[E.kind == "hard"], E[E.kind == "other_team"]
print("pairs:", E.groupby("kind").size().to_dict())
tabs = []
for v in VARIANTS:
    m = metrics(v, es, eh, eo)
    m["rank-1"], m["rank-1 n"] = rank1(v, E)
    tabs.append(m)
TE = pd.DataFrame(tabs)
print(TE.round(3).to_string())
TE.to_csv(C.OUT / "t5b_prototypes_edge_pairs.csv", index=False)
res["prototypes_edge_pairs"] = TE.round(4).to_dict("records")

print("\n=== prototype choices, both pieces have >= 5 clean rows (appearance_match.py's eligibility) ===")
G = PP[PP.n_clean_min >= 5]
gs, gh, go = G[G.kind == "same"], G[G.kind == "hard"], G[G.kind == "other_team"]
print("pairs:", G.groupby("kind").size().to_dict())
tabs = []
for v in VARIANTS:
    m = metrics(v, gs, gh, go)
    m["rank-1"], m["rank-1 n"] = rank1(v, G)
    tabs.append(m)
TG = pd.DataFrame(tabs)
print(TG.round(3).to_string())
TG.to_csv(C.OUT / "t5b_prototypes_eligible_pairs.csv", index=False)
res["prototypes_eligible_pairs"] = TG.round(4).to_dict("records")

BASE = "mean_clean"
print(f"\n=== what the cosine depends on (prototype {BASE}; AUC is same vs hard inside the bin) ===")


def dep(name, col, bins, labels=None):
    out = []
    for k, (lo, hi) in enumerate(bins):
        s, hd = same[(same[col] >= lo) & (same[col] < hi)], hard[(hard[col] >= lo) & (hard[col] < hi)]
        pr = C.prec_at_recall(s[BASE], hd[BASE]) if len(s) else {0.8: (np.nan, np.nan)}
        out.append({"factor": name, "bin": labels[k] if labels else f"{lo}-{hi}", "same pairs": len(s), "hard pairs": len(hd), "cos same p10": s[BASE].quantile(0.1) if len(s) else np.nan,
                    "cos same p50": s[BASE].median() if len(s) else np.nan, "cos hard p50": hd[BASE].median() if len(hd) else np.nan,
                    "cos hard p99": hd[BASE].quantile(0.99) if len(hd) else np.nan, "AUC": C.auc(s[BASE], hd[BASE]), "prec@rec0.8": pr[0.8][0],
                    "AUC s-norm": C.auc(s["mean_clean_snorm"], hd["mean_clean_snorm"])})  # fmt: skip
    return out


D = []
D += dep("clean rows in the smaller piece", "n_clean_min", [(0, 1), (1, 5), (5, 20), (20, 100), (100, 10000)], ["0", "1-4", "5-19", "20-99", ">=100"])
D += dep("rows in the smaller piece", "rows_min", [(0, 5), (5, 20), (20, 100), (100, 10000)], ["<5", "5-19", "20-99", ">=100"])
D += dep("median box height of the smaller-boxed piece (px)", "h_min", [(0, 40), (40, 50), (50, 60), (60, 80), (80, 2000)])
D += dep("largest share of rows within 10 px of the border", "near_max", [(0, 1e-9), (1e-9, 0.2), (0.2, 0.5), (0.5, 1.01)], ["0", "0-0.2", "0.2-0.5", ">0.5"])
D += dep("largest share of rows overlapping another box", "overlap_max", [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)])
D += dep("time between the pieces (s)", "gap_s", [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 31)])
D += dep("smaller purity of the two pieces", "purity_min", [(0, 0.8), (0.8, 0.95), (0.95, 1.01)])
DT = pd.DataFrame(D)
print(DT.round(3).to_string())
DT.to_csv(C.OUT / "t5b_dependence.csv", index=False)
res["dependence"] = DT.round(4).to_dict("records")

print("\n=== number of clean rows averaged (random subsets; pieces with >= 30 clean rows) ===")
SB = pd.DataFrame(subset_rows)
tabs = []
for nsub, g in SB.groupby("n"):
    s, hd = g[g.kind == "same"], g[g.kind == "hard"]
    pr = C.prec_at_recall(s.cos, hd.cos)
    tabs.append({"rows averaged": nsub, "same pairs": len(s), "hard pairs": len(hd), "cos same p50": s.cos.median(), "cos hard p50": hd.cos.median(), "AUC": C.auc(s.cos, hd.cos),
                 "prec@rec0.8": pr[0.8][0], "TPR@FPR1%": float((s.cos > hd.cos.quantile(0.99)).mean())})  # fmt: skip
g = SB[SB.n == 30]
s, hd = g[g.kind == "same"], g[g.kind == "hard"]
pr = C.prec_at_recall(s.full, hd.full)
tabs.append({"rows averaged": "all clean rows", "same pairs": len(s), "hard pairs": len(hd), "cos same p50": s.full.median(), "cos hard p50": hd.full.median(), "AUC": C.auc(s.full, hd.full),
             "prec@rec0.8": pr[0.8][0], "TPR@FPR1%": float((s.full > hd.full.quantile(0.99)).mean())})  # fmt: skip
NT = pd.DataFrame(tabs)
print(NT.round(3).to_string())
NT.to_csv(C.OUT / "t5b_rows_averaged.csv", index=False)
res["rows_averaged"] = NT.round(4).to_dict("records")

print("\n=== per clip (mean_clean; and after s-norm) ===")
tabs = []
for seq in C.CLIPS:
    s, hd = same[same.seq == seq], hard[hard.seq == seq]
    tabs.append({"seq": seq, "same": len(s), "hard": len(hd), "cos same p10": s[BASE].quantile(0.1), "cos same p50": s[BASE].median(), "cos hard p50": hd[BASE].median(),
                 "cos hard p95": hd[BASE].quantile(0.95), "cos hard p99": hd[BASE].quantile(0.99), "AUC": C.auc(s[BASE], hd[BASE]),
                 "snorm same p10": s.mean_clean_snorm.quantile(0.1), "snorm hard p99": hd.mean_clean_snorm.quantile(0.99), "AUC s-norm": C.auc(s.mean_clean_snorm, hd.mean_clean_snorm),
                 "tc same p10": s.mean_all_teamcentred.quantile(0.1), "tc hard p99": hd.mean_all_teamcentred.quantile(0.99), "AUC team-centred": C.auc(s.mean_all_teamcentred, hd.mean_all_teamcentred)})  # fmt: skip
PC = pd.DataFrame(tabs)
print(PC.round(3).to_string())
PC.to_csv(C.OUT / "t5b_per_clip.csv", index=False)
res["per_clip"] = PC.round(4).to_dict("records")

print("\n=== one bar for all clips: pooled precision / recall of 'cosine >= bar' in the pool same + hard ===")
tabs = []
for v, bars in ((BASE, (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96)), ("mean_all_teamcentred", (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85)), ("mean_clean_snorm", (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0))):
    for b in bars:
        tp, fp = int((same[v] >= b).sum()), int((hard[v] >= b).sum())
        worst = min(((same[(same.seq == q)][v] >= b).sum() / max(((same[(same.seq == q)][v] >= b).sum() + (hard[(hard.seq == q)][v] >= b).sum()), 1)) for q in C.CLIPS)
        tabs.append({"score": v, "bar": b, "same kept": tp, "recall": tp / len(same), "hard kept": fp, "precision": tp / max(tp + fp, 1), "worst clip precision": float(worst)})
BT = pd.DataFrame(tabs)
print(BT.round(3).to_string())
BT.to_csv(C.OUT / "t5b_global_bar.csv", index=False)
res["global_bar"] = BT.round(4).to_dict("records")

print("\n=== the scored returns: cosine of (piece of row A, piece of row B) ===")
RET = pd.concat([C.returns(s) for s in C.CLIPS], ignore_index=True)
RS = RET[RET.scored]
key = {(r.seq, r.a, r.b): r for r in PP.itertuples()}
pk = {(r.seq, r.piece): r for r in PCS.itertuples()}
rows = []
for r in RS.itertuples():
    k = key.get((r.seq, r.piece_a, r.piece_b))
    a, b = pk.get((r.seq, r.piece_a)), pk.get((r.seq, r.piece_b))
    row = {"seq": r.seq, "ref_id": r.ref_id, "gap": r.gap, "a": r.piece_a, "b": r.piece_b, "a_is_person": bool(a is not None and a.ref_id == r.ref_id),
           "b_is_person": bool(b is not None and b.ref_id == r.ref_id), "a_clean": a.n_clean if a is not None else -1, "b_clean": b.n_clean if b is not None else -1,
           "a_rows": a.rows if a is not None else 0, "b_rows": b.rows if b is not None else 0, "in_pair_table": k is not None}  # fmt: skip
    if k is not None:
        row.update({"cos": getattr(k, BASE), "snorm": k.mean_clean_snorm, "kind": k.kind})
        cand = PP[(PP.seq == r.seq) & (PP.b == r.piece_b) & PP.team_lab_ok]
        row["rank_of_A"] = int((cand[BASE] > getattr(k, BASE)).sum()) + 1
        row["candidates"] = len(cand)
        srt = cand[BASE].sort_values(ascending=False).to_numpy()
        row["margin_to_next"] = float(getattr(k, BASE) - (srt[1] if row["rank_of_A"] == 1 and len(srt) > 1 else srt[0])) if len(srt) > 1 else np.nan
    rows.append(row)
RR = pd.DataFrame(rows)
RR.to_csv(C.OUT / "t5b_return_pairs.csv", index=False)
both = RR[RR.a_is_person & RR.b_is_person & RR.in_pair_table]
print("scored returns:", len(RR), "; both pieces are the person:", int((RR.a_is_person & RR.b_is_person).sum()), "; of those in the pair table:", len(both))
print("  clean rows of the smaller piece:", C.qtable(np.minimum(both.a_clean, both.b_clean), (0.1, 0.25, 0.5, 0.75)), "; < 5 clean rows:", int((np.minimum(both.a_clean, both.b_clean) < 5).sum()))
print("  cosine:", C.qtable(both.cos, (0.05, 0.1, 0.25, 0.5, 0.75)))
print("  rank of the true earlier piece among B's team-compatible earlier pieces: rank 1:", int((both.rank_of_A == 1).sum()), " rank 2:", int((both.rank_of_A == 2).sum()),
      " rank 3+:", int((both.rank_of_A >= 3).sum()), "; candidates per B p50:", float(both.candidates.median()))  # fmt: skip
r1 = both[both.rank_of_A == 1]
print("  rank-1 returns: margin to the runner-up:", C.qtable(r1.margin_to_next, (0.1, 0.25, 0.5, 0.75)), "; margin >= 0.05:", int((r1.margin_to_next >= 0.05).sum()), "; >= 0.02:", int((r1.margin_to_next >= 0.02).sum()))
print("  by gap: ", both.assign(g=pd.cut(both.gap / C.FPS, [0, 2, 5, 10, 40])).groupby("g", observed=True).agg(n=("cos", "size"), cos_p50=("cos", "median"), rank1=("rank_of_A", lambda s: float((s == 1).mean()))).round(3).to_dict("index"))
res["returns"] = {"scored": len(RR), "both_pieces_are_the_person": int((RR.a_is_person & RR.b_is_person).sum()), "cos": C.qtable(both.cos, (0.05, 0.1, 0.25, 0.5, 0.75)),
                  "rank1": int((both.rank_of_A == 1).sum()), "rank2": int((both.rank_of_A == 2).sum()), "rank3plus": int((both.rank_of_A >= 3).sum()),
                  "rank1_margin_ge_0.05": int((r1.margin_to_next >= 0.05).sum()), "rank1_margin_ge_0.02": int((r1.margin_to_next >= 0.02).sum()),
                  "smaller_piece_under_5_clean_rows": int((np.minimum(both.a_clean, both.b_clean) < 5).sum())}  # fmt: skip
(C.OUT / "t5b_appearance_pieces.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
