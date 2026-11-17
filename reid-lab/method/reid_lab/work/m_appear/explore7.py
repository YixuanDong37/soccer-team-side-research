"""DIAGNOSIS (labels through fitlabels; no method result): prototype and similarity variants that were asked for.

All in the team-centred space unless said otherwise. Reported per variant: pair AUC, R1 (best earlier
candidate is the same person, for later pieces that have one), average precision of the top-1 decisions,
true top-1 decisions accepted before the k-th false one; the same without SNGS-162 (suspected annotation swaps).
"""
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from reid_lab import data  # noqa: E402

from core import ctx  # noqa: E402
from diag2 import ap, auc, pair_set, unit  # noqa: E402
from features import Protos, protos  # noqa: E402


def evaluate_sim(sim_by_seq, tag):
    """sim_by_seq: seq -> (ids, S) with S[i, j] the similarity of pieces ids[i], ids[j] (NaN = no score)."""
    res = {}
    for name, clips in (("all", data.DEV_CLIPS), ("w/o 162", [s for s in data.DEV_CLIPS if s != "SNGS-162"])):
        pos, neg, ts, tg, th = [], [], [], [], []
        for seq in clips:
            ids, ok, same = pair_set(seq)
            sid, S0 = sim_by_seq[seq]
            where = {p: i for i, p in enumerate(sid)}
            idx = np.array([where.get(p, -1) for p in ids])
            S = np.where((idx[:, None] >= 0) & (idx[None, :] >= 0), S0[np.clip(idx, 0, None)][:, np.clip(idx, 0, None)], np.nan)
            okk = ok & ~np.isnan(S)
            pos.append(S[okk & same])
            neg.append(S[okk & ~same])
            Sm = np.where(okk, S, -np.inf)
            for j in range(len(ids)):
                if okk[:, j].any():
                    i = int(np.argmax(Sm[:, j]))
                    ts.append(Sm[i, j])
                    tg.append(same[i, j])
                    th.append((okk[:, j] & same[:, j]).any())
        pos, neg, ts, tg, th = np.concatenate(pos), np.concatenate(neg), np.array(ts), np.array(tg), np.array(th)
        o = np.argsort(-ts)
        g = tg[o]
        fp, tp = np.cumsum(~g), np.cumsum(g)
        at = [int(tp[fp <= k].max()) if (fp <= k).any() else 0 for k in (0, 2, 5, 10)]
        res[name] = f"AUC {auc(pos, neg):.4f} R1 {tg[th].mean():.3f} AP {ap(ts, tg):.4f} TP@FP<=0/2/5/10 {at[0]:3d} {at[1]:3d} {at[2]:3d} {at[3]:3d}"
    print(f"{tag:52s} {res['all']} | w/o 162: {res['w/o 162']}", flush=True)


def proto_sim(seq, pr, kind="tc"):
    ids = sorted(pr.get(kind))
    M = np.stack([pr.get(kind)[p] for p in ids])
    return ids, M @ M.T


def variant(row_sel_fn=None, min_clean=5, kind="tc", lam=0.3):
    out = {}
    for s in data.DEV_CLIPS:
        c = ctx(s)
        pr = Protos(s, min_clean=min_clean, row_sel=None if row_sel_fn is None else row_sel_fn(c), lam=lam)
        out[s] = proto_sim(s, pr, kind)
    return out


def away_from_ends(c, n):
    """Clean rows at least n frames after the piece's first and before its last frame."""
    keep = np.zeros(len(c.h), bool)
    for pid, rows in c.rows_of.items():
        f = c.frame[rows]
        keep[rows] = (f >= f.min() + n) & (f <= f.max() - n)
    return c.clean & keep


def robust_protos(seq, how, kind="tc"):
    """Trimmed mean / coordinate median of the clean rows instead of the mean."""
    c = ctx(seq)
    pr = protos(seq)
    E = np.where(c.has[:, None], c.E, 0.0)
    out = {}
    for p, rows in pr.used.items():
        X = E[rows]
        if how == "median":
            m = np.median(X, axis=0)
        else:  # trimmed: drop the 20 % of rows least similar to the mean, twice
            m = X.mean(axis=0)
            for _ in range(2):
                if len(X) >= 10:
                    sim = unit(X) @ unit(m)
                    X = X[sim >= np.quantile(sim, 0.2)]
                    m = X.mean(axis=0)
        team = pr.ateam[p]
        centre = pr.team_mean[team] if team in pr.team_mean and pr.team_mean[team] is not None else 0.0
        v = m - centre
        out[p] = unit(pr.T @ v) if kind == "w" else unit(v)
    ids = sorted(out)
    M = np.stack([out[p] for p in ids])
    return ids, M @ M.T


def multi_proto(seq, k, agg, kind="tc"):
    """k prototypes per piece (consecutive chunks of its clean rows in time); piece similarity = max or mean of the k x k cosines."""
    c = ctx(seq)
    pr = protos(seq)
    E = np.where(c.has[:, None], c.E, 0.0)
    ids = sorted(pr.used)
    reps = []
    for p in ids:
        rows = pr.used[p]
        chunks = np.array_split(rows, k) if len(rows) >= 5 * k else [rows]
        team = pr.ateam[p]
        centre = pr.team_mean[team] if team in pr.team_mean and pr.team_mean[team] is not None else 0.0
        vs = [E[ch].mean(axis=0) - centre for ch in chunks]
        reps.append(np.stack([unit(pr.T @ v) if kind == "w" else unit(v) for v in vs]))
    n = len(ids)
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            m = reps[i] @ reps[j].T
            S[i, j] = S[j, i] = m.max() if agg == "max" else m.mean()
    return ids, S


def rerank(seq, kind, k1, lam):
    """k-reciprocal re-ranking within the appearance team: similarity = (1 - lam) * Jaccard of the k-reciprocal
    neighbour sets + lam * cosine."""
    pr = protos(seq)
    ids, S = proto_sim(seq, pr, kind)
    team = np.array([pr.ateam[p] or "" for p in ids])
    n = len(ids)
    Sm = np.where(team[:, None] == team[None, :], S, -np.inf)
    np.fill_diagonal(Sm, np.inf)
    order = np.argsort(-Sm, axis=1)
    topk = [set(order[i, :k1 + 1].tolist()) for i in range(n)]
    recip = [{j for j in topk[i] if i in topk[j]} for i in range(n)]
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            u = len(recip[i] | recip[j])
            J[i, j] = len(recip[i] & recip[j]) / u if u else 0.0
    return ids, (1 - lam) * J + lam * S


def csls(seq, kind, k):
    """Similarity relative to each side's neighbourhood: s - (mean of a's k best same-team cosines + b's) / 2."""
    pr = protos(seq)
    ids, S = proto_sim(seq, pr, kind)
    team = np.array([pr.ateam[p] or "" for p in ids])
    Sm = np.where(team[:, None] == team[None, :], S, -np.inf)
    np.fill_diagonal(Sm, -np.inf)
    top = -np.sort(-Sm, axis=1)[:, :k]
    r = np.where(np.isfinite(top), top, np.nan)
    r = np.nanmean(r, axis=1)
    return ids, S - (r[:, None] + r[None, :]) / 2


if __name__ == "__main__":
    what = sys.argv[1:] or ["rows", "robust", "multi", "rerank", "wccn"]
    evaluate_sim(variant(kind="raw"), "raw, clean rows mean (appearance_match.py)")
    evaluate_sim(variant(), "team-centred, clean rows mean  [features.cos_tc]")
    evaluate_sim(variant(kind="w"), "team-centred + WCCN 0.3        [features.cos_w]")
    if "rows" in what:
        evaluate_sim(variant(lambda c: np.ones(len(c.h), bool), 1), "tc, all rows")
        evaluate_sim(variant(lambda c: c.clean & (c.edge_dist > 30)), "tc, clean & >30 px from the edge")
        evaluate_sim(variant(lambda c: (c.cover <= 0.3) & (c.edge_dist > 1) & (c.h >= 40)), "tc, cover<=0.3 instead of no overlap")
        evaluate_sim(variant(lambda c: c.clean & (c.h >= 60)), "tc, clean & >= 60 px tall")
        for n in (5, 12, 25):
            evaluate_sim(variant(lambda c, n=n: away_from_ends(c, n)), f"tc, clean rows >= {n} frames from the piece's ends")
    if "robust" in what:
        evaluate_sim({s: robust_protos(s, "median") for s in data.DEV_CLIPS}, "tc, coordinate median")
        evaluate_sim({s: robust_protos(s, "trimmed") for s in data.DEV_CLIPS}, "tc, trimmed mean (drop 20 % twice)")
    if "multi" in what:
        for k in (2, 3):
            for agg in ("max", "mean"):
                evaluate_sim({s: multi_proto(s, k, agg) for s in data.DEV_CLIPS}, f"tc, {k} prototypes per piece, {agg}")
    if "rerank" in what:
        for k1 in (3, 5):
            for lam in (0.3, 0.7):
                evaluate_sim({s: rerank(s, "tc", k1, lam) for s in data.DEV_CLIPS}, f"tc, k-reciprocal k1={k1} lam={lam}")
        for k in (3, 10):
            evaluate_sim({s: csls(s, "tc", k) for s in data.DEV_CLIPS}, f"tc, CSLS k={k}")
    if "wccn" in what:
        for lam in (0.03, 0.1, 1.0, 3.0):
            evaluate_sim(variant(kind="w", lam=lam), f"WCCN lambda={lam}")
