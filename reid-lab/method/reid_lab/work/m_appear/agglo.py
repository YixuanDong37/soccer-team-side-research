"""Label-free agglomerative re-linking of pieces with time constraints (m_appear's method).

Groups start as single pieces (not referees, with a prototype). Two groups are "allowed" to merge only
when they are of the same appearance team and share at most max_shared frames (default 0: a person is in
one place at a time, so every merged group stays non-overlapping). Repeatedly the most similar allowed
pair that passes the test is merged, and the group's prototype is recomputed from the pooled rows:

  similarity s(G, H): cosine of the prototypes; kind "w" (team-centred + whitened, default), "tc"
      (team-centred) or "raw". linkage "centroid" (default): prototypes of the pooled rows; "avg" / "min":
      row-count-weighted mean / minimum of the earlier similarities.
  test:  s >= t_abs,
         mutual: H is G's most similar allowed group and G is H's,
         s - (G's best similarity to an allowed group that is seen together with H) >= margin   [H's rivals]
         s - (H's best similarity to an allowed group that is seen together with G) >= margin   [G's rivals]
         "Seen together" = sharing at least covis frames (default 1). A group seen together with H is
         certainly not H's person, so it is a true rival for G; a group that is not could be another piece of
         H's person and must not block the link. A side without any rival passes (need_rival=True: abstains).

Returns links [[a, b], ...]: consecutive pieces (by first frame) of every merged group. With cuts, the ids
are reid_lab.score's sub-piece ids.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import ctx  # noqa: E402
from features import candidate_ids, protos, unit  # noqa: E402


def agglomerate(seq, kind="w", t_abs=0.5, margin=0.25, need_rival=False, min_rows=1, mutual=True, pr=None,
                linkage="centroid", log=None, exclude=None, covis=1, max_shared=0, cuts=()):
    c = ctx(seq, cuts)
    pr = pr or protos(seq, cuts)
    ids = candidate_ids(seq, pr, cuts)
    if exclude:
        ids = [p for p in ids if p not in exclude]
    n = len(ids)
    if n < 2:
        return []
    P = c.P
    nrow = np.array([len(pr.used[p]) for p in ids], float)
    msum = np.stack([pr.mean[p] * len(pr.used[p]) for p in ids])           # sum of the raw prototype rows per group
    team = np.array([pr.ateam[p] or "" for p in ids])
    tm = pr.team_mean
    centre = np.stack([tm[t] if t in tm and tm[t] is not None else pr.glob for t in team])
    first = np.array([P["first"][c.index[p]] for p in ids])
    last = np.array([P["last"][c.index[p]] for p in ids])
    frames = [c.frames_of[p] for p in ids]
    members = [[p] for p in ids]
    alive = np.ones(n, bool)
    N = np.zeros((n, n), int)                                              # frames shared by two groups
    for i in range(n):
        for j in range(i + 1, n):
            if first[i] <= last[j] and first[j] <= last[i]:
                N[i, j] = N[j, i] = len(frames[i] & frames[j])
    same_team = team[:, None] == team[None, :]

    def proto(i):
        v = msum[i] / nrow[i]
        if kind == "raw":
            return unit(v)
        v = v - centre[i]
        return unit(pr.T @ v) if kind == "w" else unit(v)

    X = np.stack([proto(i) for i in range(n)])
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    while True:
        seen = N >= covis                                                  # seen together: certainly two people
        can = same_team & (N <= max_shared) & alive[:, None] & alive[None, :]
        np.fill_diagonal(can, False)
        if min_rows > 1:
            small = nrow < min_rows
            can[small, :] = False
            can[:, small] = False
        Sc = np.where(can, S, -np.inf)
        # rival[g, h] = G's best allowed similarity among the groups seen together with H
        Om = np.where(seen & alive[None, :], 0.0, -np.inf)                 # [h, h']
        rival = (Sc[:, None, :] + Om[None, :, :]).max(axis=2)              # [g, h]
        with np.errstate(invalid="ignore"):
            mg = np.where(np.isfinite(rival), S - rival, np.nan)           # margin on g's side of the pair (g, h)
        mh = mg.T
        ok = can & (S >= t_abs)
        if need_rival:
            ok &= ~np.isnan(mg) & ~np.isnan(mh)
        ok &= (np.isnan(mg) | (mg >= margin)) & (np.isnan(mh) | (mh >= margin))
        if mutual:
            best = Sc.argmax(axis=1)
            isbest = np.zeros((n, n), bool)
            isbest[np.arange(n), best] = True
            ok &= isbest & isbest.T
        if not ok.any():
            break
        Sok = np.where(ok, S, -np.inf)
        g, h = np.unravel_index(int(Sok.argmax()), Sok.shape)
        g, h = int(g), int(h)
        if log is not None:
            log.append({"s": float(S[g, h]), "g": list(members[g]), "h": list(members[h]),
                        "rival_g": float(mg[g, h]), "rival_h": float(mh[g, h])})
        ng, nh = nrow[g], nrow[h]
        if linkage == "avg":
            v = (ng * S[g] + nh * S[h]) / (ng + nh)
        elif linkage == "min":
            v = np.minimum(S[g], S[h])
        msum[g] += msum[h]
        nrow[g] += nrow[h]
        frames[g] = frames[g] | frames[h]
        members[g] += members[h]
        alive[h] = False
        N[g] += N[h]
        N[:, g] += N[:, h]
        N[g, g] = 0
        if linkage == "centroid":
            X[g] = proto(g)
            v = X @ X[g]
        S[g, :] = v
        S[:, g] = v
        S[g, g] = -np.inf
        S[h, :] = -np.inf
        S[:, h] = -np.inf
    links = []
    for i in np.nonzero(alive)[0]:
        m = sorted(members[i], key=lambda p: P["first"][c.index[p]])
        links += [[int(m[k]), int(m[k + 1])] for k in range(len(m) - 1)]
    return links
