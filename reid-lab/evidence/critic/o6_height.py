"""critic / o6_height.py -- DIAGNOSIS (labels only for the same / different verdict): is the player's
BODY HEIGHT IN METRES (box height through the per-frame camera) a usable cue between team-mates?
Nobody tried it; the appearance model sees a resized crop and cannot know the absolute size.

Per clean row (data.clean_rows: away from the edge, no overlap, >= 40 px) with a usable ground point:
  h_m = 1.8 * box height / (pixel length of a 1.8 m vertical pole at the ground point)
Per piece: median h_m over its clean rows (pieces with >= 10 such rows).
Pairs of pieces of known identity (>= 5 matched rows), both player / goalkeeper of the same annotated
team: same person vs different people. Reported: spread, AUC of -|dh|, and the same among the
pairs the appearance model finds hard (raw prototype cosine >= 0.90).
"""
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/critic")
import common as C
from common import data

P = C.Tee(C.OUT / "o6_height.txt")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def piece_heights(seq):
    t = data.tracks(seq)
    cal = data.calib(seq).set_index("frame_index")
    clean = data.clean_rows(seq)
    ok = clean & t.pitch_usable.to_numpy(bool) & t.in_pitch.to_numpy(bool) & t.pitch_x.notna().to_numpy() & (t.track_id.to_numpy() >= 0)
    idx = np.nonzero(ok)[0]
    fr = t.frame.to_numpy()[idx]
    K = np.stack([np.asarray(cal.K.loc[f], float).reshape(3, 3) for f in fr])
    R = np.stack([np.asarray(cal.R.loc[f], float).reshape(3, 3) for f in fr])
    Cc = np.stack([np.asarray(cal.C.loc[f], float) for f in fr])
    X0 = np.column_stack([t.pitch_x.to_numpy()[idx], t.pitch_y.to_numpy()[idx], np.zeros(len(idx))])
    X1 = X0.copy()
    X1[:, 2] = -1.8                       # z points down in this world frame (camera z is negative)

    def proj(X):
        cam = np.einsum("nij,nj->ni", R, X - Cc)
        uvw = np.einsum("nij,nj->ni", K, cam)
        return uvw[:, :2] / uvw[:, 2:3]

    p0, p1 = proj(X0), proj(X1)
    foot_err = np.hypot(p0[:, 0] - t.foot_u.to_numpy()[idx], p0[:, 1] - t.foot_v.to_numpy()[idx])
    pole = p0[:, 1] - p1[:, 1]
    h = 1.8 * (t.y2.to_numpy()[idx] - t.y1.to_numpy()[idx]) / pole
    good = (pole > 5) & np.isfinite(h) & (h > 1.0) & (h < 2.6)
    tid = t.track_id.to_numpy()[idx]
    out = {}
    for p in np.unique(tid[good]):
        v = h[good & (tid == p)]
        if len(v) >= 10:
            out[int(p)] = (float(np.median(v)), float(np.subtract(*np.percentile(v, [75, 25]))), len(v))
    emb = data.embeddings(seq)
    proto = {}
    for p in out:
        rows = np.nonzero(clean & (t.track_id.to_numpy() == p))[0]
        m = emb[rows].mean(axis=0)
        proto[p] = m / np.linalg.norm(m)
    return out, proto, float(np.median(foot_err)), float(np.mean(good))


same, diff, same_hard, diff_hard, within = [], [], [], [], []
per_person_sd = []
for seq in data.DEV_CLIPS:
    lab = C.labels(seq)
    hts, proto, ferr, share = piece_heights(seq)
    maj = C.piece_majority(seq, (), 5)
    ps = [p for p in hts if p in maj and lab[maj[p]]["role"] in ("player", "goalkeeper")]
    within += [hts[p][1] for p in ps]
    by = {}
    for p in ps:
        by.setdefault(maj[p], []).append(hts[p][0])
    per_person_sd += [np.std(v) for v in by.values() if len(v) >= 2]
    ns = nd = 0
    for a, b in combinations(ps, 2):
        ra, rb = maj[a], maj[b]
        if lab[ra]["team"] != lab[rb]["team"]:
            continue
        d = abs(hts[a][0] - hts[b][0])
        cos = float(proto[a] @ proto[b])
        if ra == rb:
            same.append(d); ns += 1
            if cos >= 0.90:
                same_hard.append(d)
        else:
            diff.append(d); nd += 1
            if cos >= 0.90:
                diff_hard.append(d)
    P(f"{seq}: pieces with a height {len(ps)}, median foot reprojection error {ferr:.2f} px, rows kept {100 * share:.0f}%, same-person pairs {ns}, different {nd}")

same, diff = np.array(same), np.array(diff)
P(f"\nwithin-piece spread of the row heights (IQR), median over pieces: {np.median(within):.3f} m")
P(f"same person, piece to piece: sd of the piece medians, median over persons: {np.median(per_person_sd):.3f} m")
P(f"|dh| same person      (n {len(same)}): median {np.median(same):.3f} m, p90 {np.percentile(same, 90):.3f} m")
P(f"|dh| different people (n {len(diff)}, same team): median {np.median(diff):.3f} m, p90 {np.percentile(diff, 90):.3f} m")
P(f"AUC of -|dh| (same vs different, same team): {auc(-same, -diff):.3f}")
P(f"AUC among pairs with raw prototype cosine >= 0.90 (same {len(same_hard)}, different {len(diff_hard)}): {auc(-np.array(same_hard), -np.array(diff_hard)):.3f}")
for thr in (0.05, 0.08, 0.10, 0.15):
    P(f"veto |dh| > {thr:.2f} m: removes {100 * np.mean(diff > thr):.0f}% of different-people pairs, {100 * np.mean(same > thr):.1f}% of same-person pairs;"
      f" among hard pairs: {100 * np.mean(np.array(diff_hard) > thr):.0f}% / {100 * np.mean(np.array(same_hard) > thr):.1f}%")
