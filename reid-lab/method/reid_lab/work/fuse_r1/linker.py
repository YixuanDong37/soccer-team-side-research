"""fuse / linker.py -- LABEL-FREE: constrained agglomeration of (sub-)pieces on a fused similarity.

This is m_appear's agglomeration (work/m_appear/agglo.py, re-implemented here so that a second kind
of evidence can enter; with no bonus it returns exactly m_appear's links, see check_linker.py) with
one extension: the similarity of two groups is

    S(G, H) = cosine of their pooled, team-centred, whitened appearance prototypes      (m_appear)
            + the largest pair bonus at a junction of G and H                           (new)

A junction is a pair (p in G, q in H) of pieces that would be neighbours in time in the merged
chain; the pair bonus of a time-ordered pair of pieces comes from the position model (m_space) and,
optionally, from the plugin's own join, weighted by the pair model (pairmodel.py) and expressed in
cosine units. Position therefore only speaks where one piece would directly follow the other.

Rule (unchanged from m_appear): two groups may merge when they are of the same appearance team and
share no frame. Repeatedly merge the most similar allowed pair (G, H) that passes
    S >= t_abs;  H is G's most similar allowed group and G is H's (mutual best);
    S - (G's best S to an allowed group seen together with H) >= margin, and the same for H
    (a group seen together with H is certainly not H's person; a side without such a rival passes);
then pool the prototypes and continue. Links = consecutive pieces of every merged group.

Two additions for the fused similarity (both off by default, so the default is m_appear's rule):
  floor  no merge when the appearance cosine alone is below `floor`;
  t0     a "nobody" rival: the margin of each side is taken against max(best rival, t0), also when
         the side has no rival at all, so S >= t0 + margin always holds. One number (margin) then
         sets how sure a merge must be.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse_r1")
import prep  # noqa: E402


def _appear():
    af = prep._import("m_appear", "features")
    core = sys.modules["m_appear__core"]
    return af, core


def _junction_bonus(mg, mh, first, B):
    """Largest bonus over the junctions of two groups (member index lists sorted by first frame)."""
    seq = sorted([(first[i], 0, i) for i in mg] + [(first[i], 1, i) for i in mh])
    best = 0.0
    for (_, ga, ia), (_, gb, ib) in zip(seq[:-1], seq[1:]):
        if ga != gb:
            v = B[ia, ib]
            if v > best:
                best = v
    return best


def agglomerate(seq, cuts=(), bonus=None, kind="w", t_abs=0.5, margin=0.25, floor=None, t0=None, log=None):
    """Links [[a, b], ...] of one clip. bonus: {(a, b): value >= 0} for time-ordered pairs of
    sub-pieces (a ends before b starts), in cosine units; None = appearance only (m_appear)."""
    af, core = _appear()
    cuts = prep.norm_cuts(cuts)
    c = core.ctx(seq, cuts)
    pr = af.protos(seq, cuts)
    ids = af.candidate_ids(seq, pr, cuts)
    n = len(ids)
    if n < 2:
        return []
    P = c.P
    pos = {p: i for i, p in enumerate(ids)}
    nrow = np.array([len(pr.used[p]) for p in ids], float)
    msum = np.stack([pr.mean[p] * len(pr.used[p]) for p in ids])
    team = np.array([pr.ateam[p] or "" for p in ids])
    tm = pr.team_mean
    centre = np.stack([tm[t] if t in tm and tm[t] is not None else pr.glob for t in team])
    first = np.array([P["first"][c.index[p]] for p in ids])
    last = np.array([P["last"][c.index[p]] for p in ids])
    frames = [c.frames_of[p] for p in ids]
    members = [[i] for i in range(n)]
    alive = np.ones(n, bool)
    N = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if first[i] <= last[j] and first[j] <= last[i]:
                N[i, j] = N[j, i] = len(frames[i] & frames[j])
    same_team = team[:, None] == team[None, :]
    B = np.zeros((n, n))                      # piece-level bonus, [earlier, later]
    if bonus:
        for (a, b), v in bonus.items():
            i, j = pos.get(int(a)), pos.get(int(b))
            if i is not None and j is not None and v > 0:
                B[i, j] = v
    Bg = np.maximum(B, B.T)                   # group-level bonus (single pieces: the pair's own)

    def proto(i):
        v = msum[i] / nrow[i]
        if kind == "raw":
            return af.unit(v)
        v = v - centre[i]
        return af.unit(pr.T @ v) if kind == "w" else af.unit(v)

    X = np.stack([proto(i) for i in range(n)])
    A = X @ X.T                               # appearance similarity of groups
    np.fill_diagonal(A, -np.inf)
    while True:
        S = A + Bg
        seen = N >= 1
        can = same_team & (N <= 0) & alive[:, None] & alive[None, :]
        np.fill_diagonal(can, False)
        Sc = np.where(can, S, -np.inf)
        Om = np.where(seen & alive[None, :], 0.0, -np.inf)
        rival = (Sc[:, None, :] + Om[None, :, :]).max(axis=2)
        with np.errstate(invalid="ignore"):
            if t0 is None:
                mg = np.where(np.isfinite(rival), S - rival, np.nan)
            else:
                mg = S - np.maximum(rival, t0)             # rival is -inf where there is none
        mh = mg.T
        ok = can & (S >= t_abs)
        if floor is not None:
            ok &= A >= floor
        ok &= (np.isnan(mg) | (mg >= margin)) & (np.isnan(mh) | (mh >= margin))
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
            log.append({"s": float(S[g, h]), "app": float(A[g, h]), "bonus": float(Bg[g, h]),
                        "g": [int(ids[i]) for i in members[g]], "h": [int(ids[i]) for i in members[h]],
                        "rival_g": float(mg[g, h]), "rival_h": float(mh[g, h])})
        msum[g] += msum[h]
        nrow[g] += nrow[h]
        frames[g] = frames[g] | frames[h]
        members[g] = sorted(members[g] + members[h], key=lambda i: first[i])
        alive[h] = False
        N[g] += N[h]
        N[:, g] += N[:, h]
        N[g, g] = 0
        X[g] = proto(g)
        v = X @ X[g]
        A[g, :] = v
        A[:, g] = v
        A[g, g] = -np.inf
        A[h, :] = -np.inf
        A[:, h] = -np.inf
        if bonus:
            for x in np.nonzero(alive)[0]:
                if x != g:
                    Bg[g, x] = Bg[x, g] = _junction_bonus(members[g], members[x], first, B)
        Bg[h, :] = 0.0
        Bg[:, h] = 0.0
    links = []
    for i in np.nonzero(alive)[0]:
        m = members[i]
        links += [[int(ids[m[k]]), int(ids[m[k + 1]])] for k in range(len(m) - 1)]
    return links
