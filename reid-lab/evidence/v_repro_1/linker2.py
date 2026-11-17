"""fuse_r1 / linker2.py -- LABEL-FREE: the two-stage linker of fuse_r1.

Stage 1 (core) is fuse's constrained agglomeration (linker.py; with n_solid = 0 this function
returns exactly linker.agglomerate's links, see check_linker2.py) run on the SOLID pieces only:
(sub-)pieces with at least n_solid boxes. Short pieces take no part in it: they are neither
candidates nor rivals, and the position evidence handed in (bonus_core) was computed without them.

Stage 2 (leaves, optional): every SHORT piece may be attached to at most one stage-1 group (a
single solid piece is a group of one). A leaf never joins another short piece, never connects
two groups, and does not change the group's appearance prototype. The similarity of a leaf f and
a group G is
    S(f, G) = cosine of f's prototype and G's pooled prototype
            + the largest bonus_leaf at a junction of f with G's SOLID members in time order
and the decision rule is the same as in stage 1: same appearance team, no shared frame (leaves
already attached to G count), appearance cosine >= floor, f is G's best free leaf and G is f's
best group, and S beats by `margin` both t0 ("nobody") and the best rival on either side (for f:
its best S to a group seen together with G; for G: its best S to a leaf seen together with f).

Why two stages (round 0's verifier finding): in fuse a 1-4 box piece could join group G on thin
evidence and then, as the last member of G, pass a strong position bonus to the piece that
follows it, so that two long pieces of different people were joined through it. Here two solid
pieces are joined only on their own evidence.

    link(seq, cuts, bonus_core, bonus_leaf, margin, ...) -> (core_links, leaf_links)
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_1")
import prep  # noqa: E402
from linker import _appear, _junction_bonus  # noqa: E402


def _bonus_matrix(bonus, pos, n):
    B = np.zeros((n, n))                      # piece-level bonus, [earlier, later]
    if bonus:
        for (a, b), v in bonus.items():
            i, j = pos.get(int(a)), pos.get(int(b))
            if i is not None and j is not None and v > 0:
                B[i, j] = v
    return B


def _decide(A, Bg, N, can, alive, t_abs, margin, floor, t0):
    """One step of the rule: the index pair (g, h) to merge (or None), S and the margins."""
    n = len(A)
    S = A + Bg
    seen = N >= 1
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
        return None, S, mg
    Sok = np.where(ok, S, -np.inf)
    g, h = np.unravel_index(int(Sok.argmax()), Sok.shape)
    return (int(g), int(h)), S, mg


def link(seq, cuts=(), bonus_core=None, bonus_leaf=None, margin=0.5, floor=0.3, t0=0.5, t_abs=0.5,
         n_solid=prep.N_SOLID, leaves=True, margin_leaf=None, kind="w", log=None):
    """(core_links, leaf_links) of one clip; both lists of [earlier id, later id].
    bonus_core: {(a, b): value >= 0} for time-ordered pairs of SOLID sub-pieces (cosine units);
    bonus_leaf: the same for pairs with a short piece, from the evidence on all pieces."""
    af, core = _appear()
    cuts = prep.norm_cuts(cuts)
    c = core.ctx(seq, cuts)
    pr = af.protos(seq, cuts)
    ids = af.candidate_ids(seq, pr, cuts)
    n = len(ids)
    if n < 2:
        return [], []
    P = c.P
    pos = {p: i for i, p in enumerate(ids)}
    boxes = np.array([P["rows"][c.index[p]] for p in ids])
    solid = boxes >= n_solid
    nrow = np.array([len(pr.used[p]) for p in ids], float)
    msum = np.stack([pr.mean[p] * len(pr.used[p]) for p in ids])
    team = np.array([pr.ateam[p] or "" for p in ids])
    tm = pr.team_mean
    centre = np.stack([tm[t] if t in tm and tm[t] is not None else pr.glob for t in team])
    first = np.array([P["first"][c.index[p]] for p in ids])
    last = np.array([P["last"][c.index[p]] for p in ids])
    frames = [c.frames_of[p] for p in ids]
    members = [[i] for i in range(n)]         # solid members of a group, in time order
    N = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if first[i] <= last[j] and first[j] <= last[i]:
                N[i, j] = N[j, i] = len(frames[i] & frames[j])
    same_team = team[:, None] == team[None, :]

    def proto(i):
        v = msum[i] / nrow[i]
        if kind == "raw":
            return af.unit(v)
        v = v - centre[i]
        return af.unit(pr.T @ v) if kind == "w" else af.unit(v)

    X = np.stack([proto(i) for i in range(n)])

    # ------------------------------------------------------------------ stage 1: solid pieces
    alive = solid.copy()
    B = _bonus_matrix(bonus_core, pos, n)
    Bg = np.maximum(B, B.T)
    A = X @ X.T
    np.fill_diagonal(A, -np.inf)
    A[~alive, :] = -np.inf
    A[:, ~alive] = -np.inf
    while True:
        can = same_team & (N <= 0) & alive[:, None] & alive[None, :]
        np.fill_diagonal(can, False)
        pick, S, mg = _decide(A, Bg, N, can, alive, t_abs, margin, floor, t0)
        if pick is None:
            break
        g, h = pick
        if log is not None:
            log.append({"stage": 1, "s": float(S[g, h]), "app": float(A[g, h]), "bonus": float(Bg[g, h]),
                        "g": [int(ids[i]) for i in members[g]], "h": [int(ids[i]) for i in members[h]],
                        "rival_g": float(mg[g, h]), "rival_h": float(mg[h, g])})
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
        v[~alive] = -np.inf
        A[g, :] = v
        A[:, g] = v
        A[g, g] = -np.inf
        A[h, :] = -np.inf
        A[:, h] = -np.inf
        if bonus_core:
            for x in np.nonzero(alive)[0]:
                if x != g:
                    Bg[g, x] = Bg[x, g] = _junction_bonus(members[g], members[x], first, B)
        Bg[h, :] = 0.0
        Bg[:, h] = 0.0
    core_links = []
    for i in np.nonzero(alive)[0]:
        m = members[i]
        core_links += [[int(ids[m[k]]), int(ids[m[k + 1]])] for k in range(len(m) - 1)]

    # ------------------------------------------------------------------ stage 2: leaves
    leaf_links = []
    isleaf = ~solid
    if leaves and isleaf.any() and alive.any():
        groups = alive.copy()                 # representatives of the stage-1 groups
        alive2 = groups | isleaf
        B2 = _bonus_matrix(bonus_leaf, pos, n)
        A2 = X @ X.T
        np.fill_diagonal(A2, -np.inf)
        A2[~alive2, :] = -np.inf
        A2[:, ~alive2] = -np.inf
        Bg2 = np.zeros((n, n))
        if bonus_leaf:
            for f in np.nonzero(isleaf)[0]:
                for g in np.nonzero(groups)[0]:
                    Bg2[f, g] = Bg2[g, f] = _junction_bonus(members[g], [f], first, B2)
        cross = isleaf[:, None] != isleaf[None, :]
        m2 = margin if margin_leaf is None else margin_leaf
        while True:
            can = same_team & (N <= 0) & alive2[:, None] & alive2[None, :] & cross
            pick, S, mg = _decide(A2, Bg2, N, can, alive2, t_abs, m2, floor, t0)
            if pick is None:
                break
            g, f = pick if isleaf[pick[1]] else (pick[1], pick[0])
            # anchor: the solid member next to the leaf in time with the larger junction bonus
            mem = members[g]
            prev = [i for i in mem if first[i] < first[f]]
            nxt = [i for i in mem if first[i] > first[f]]
            cand = []
            if prev:
                cand.append((B2[prev[-1], f], -abs(int(first[f]) - int(last[prev[-1]])), prev[-1]))
            if nxt:
                cand.append((B2[f, nxt[0]], -abs(int(first[nxt[0]]) - int(last[f])), nxt[0]))
            anchor = max(cand)[2]
            a, b = (anchor, f) if first[anchor] < first[f] else (f, anchor)
            leaf_links.append([int(ids[a]), int(ids[b])])
            if log is not None:
                log.append({"stage": 2, "s": float(S[g, f]), "app": float(A2[g, f]), "bonus": float(Bg2[g, f]),
                            "g": [int(ids[i]) for i in mem], "h": [int(ids[f])],
                            "rival_g": float(mg[g, f]), "rival_h": float(mg[f, g])})
            alive2[f] = False
            frames[g] = frames[g] | frames[f]
            N[g] += N[f]
            N[:, g] += N[:, f]
            N[g, g] = 0
            A2[f, :] = -np.inf
            A2[:, f] = -np.inf
    return core_links, leaf_links
