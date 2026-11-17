"""ORACLE / DIAGNOSIS ONLY. Variants of the collective-flow compensation: residual of true
partners and rank of the true earlier piece among role-compatible candidates."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import geom  # noqa: E402

pd.set_option("display.width", 250)
HALF = 6  # frames: velocity chord half length


def piece_velocities(seq):
    """Per tracked player row: velocity (m/s) from a central chord of +-HALF frames inside the piece."""
    t = data.tracks(seq)
    t = t[(t.track_id >= 0) & t.pitch_usable & t.pitch_x.notna()].sort_values(["track_id", "frame"])
    out = []
    for pid, g in t.groupby("track_id"):
        f = g.frame.to_numpy()
        xy = g[["pitch_x", "pitch_y"]].to_numpy()
        if len(f) < 2 * HALF + 1:
            continue
        idx = {int(ff): i for i, ff in enumerate(f)}
        for i, ff in enumerate(f):
            a, b = idx.get(int(ff) - HALF), idx.get(int(ff) + HALF)
            if a is None or b is None:
                continue
            v = (xy[b] - xy[a]) / (2 * HALF / geom.FPS)
            out.append((int(ff), int(pid), xy[i, 0], xy[i, 1], v[0], v[1], g.role.iloc[i], g.team.iloc[i]))
    return pd.DataFrame(out, columns=["frame", "piece", "x", "y", "vx", "vy", "role", "team"])


def flows(seq):
    v = piece_velocities(seq)
    v = v[(v.role == "player") & (np.hypot(v.vx, v.vy) < 12)]
    n = geom.cameras(seq)["n"]
    res = {}
    for name, fn in (("median", np.median), ("mean", np.mean)):
        F = np.zeros((n, 2))
        for f, g in v.groupby("frame"):
            if len(g) >= 3:
                F[int(f)] = fn(g[["vx", "vy"]].to_numpy(), axis=0)
        res[name] = np.cumsum(F, axis=0) / geom.FPS
    for team in ("left", "right"):
        F = np.zeros((n, 2))
        for f, g in v[v.team == team].groupby("frame"):
            if len(g) >= 2:
                F[int(f)] = np.median(g[["vx", "vy"]].to_numpy(), axis=0)
        res[team] = np.cumsum(F, axis=0) / geom.FPS
    return res


p = pd.read_pickle("oracle_pairs.pkl")
p["dx"] = p.b_start_x - p.a_end_x
p["dy"] = p.b_start_y - p.a_end_y
parts = []
for seq in data.DEV_CLIPS:
    S = flows(seq)
    q = p[p.seq == seq].copy()
    ta, tb = q.a_last.to_numpy(), q.b_first.to_numpy()
    for name in ("median", "mean"):
        sh = S[name][tb] - S[name][ta]
        q[f"sx_{name}"], q[f"sy_{name}"] = sh[:, 0], sh[:, 1]
    sh = np.where((q.a_team == "left").to_numpy()[:, None], S["left"][tb] - S["left"][ta],
                  np.where((q.a_team == "right").to_numpy()[:, None], S["right"][tb] - S["right"][ta], S["median"][tb] - S["median"][ta]))
    q["sx_team"], q["sy_team"] = sh[:, 0], sh[:, 1]
    parts.append(q)
p = pd.concat(parts, ignore_index=True)
variants = {"raw d": (0.0, "median")}
for name in ("median", "mean", "team"):
    for alpha in (0.5, 0.75, 1.0):
        variants[f"{name} a={alpha}"] = (alpha, name)
s = p[p.same].sort_values(["seq", "b_piece", "a_last"])
imm_idx = s.groupby(["seq", "b_piece"]).tail(1).index
for label, (alpha, name) in variants.items():
    r = np.hypot(p.dx - alpha * p[f"sx_{name}"], p.dy - alpha * p[f"sy_{name}"])
    p["_r"] = r
    imm = p.loc[imm_idx]
    line = []
    for lo, hi in ((0, 25), (25, 75), (75, 150), (150, 300), (300, 800)):
        q = imm[(imm.gap >= lo) & (imm.gap < hi)]
        line.append(f"{q._r.median():5.1f}/{q._r.quantile(.8):5.1f}")
    ranks = []
    ranks_long = []
    for (seq, bp), g in p.groupby(["seq", "b_piece"]):
        t = g[g.same]
        if not len(t):
            continue
        best_true = t._r.min()
        cand = g[(g.a_role == g.b_role)]
        rk = int((cand._r < best_true).sum()) + 1
        ranks.append(rk)
        if t.gap.min() >= 25:
            ranks_long.append(rk)
    r1, r2 = np.array(ranks), np.array(ranks_long)
    print(f"{label:14s} resid med/p80 by gap bin: {'  '.join(line)} | top1 {int((r1 == 1).sum())}/{len(r1)} top3 {int((r1 <= 3).sum())} | gap>=25: top1 {int((r2 == 1).sum())}/{len(r2)} top3 {int((r2 <= 3).sum())}")
