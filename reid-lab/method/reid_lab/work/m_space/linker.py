"""Position (+ team / role) only linker of pieces: a conditional-logit model over the label-free
pair features (features.py) with competition between candidates.

Label-free part (apply): admissible(), posteriors(), links().
Label part (fit): fit() reads fitdata.labelled_pairs for the FIT clips only; loco_params() fits
on 12 clips for each held-out clip, and the weights are applied to it unchanged.

Model. For a later piece b and its admissible earlier pieces A(b) (a.last < b.first, same role,
teams not known to differ, b within reach of a under the speed bound):
    P(a | b) = exp(w . x_ab) / (exp(g_in . y_b) + sum_{a' in A(b)} exp(w . x_a'b))
and the same over the later pieces of an earlier piece a, with its own "nobody" term:
    P(b | a) = exp(w . x_ab) / (exp(g_out . y_a) + sum_{b' in B(a)} exp(w . x_ab'))
x_ab (pair evidence, any subset by name):
    lu   log(1 + u^2), u = residual / sigma(dt); residual = flow-compensated distance (r or r_cv)
    ls   log sigma(dt)      (sigma(dt) = s0 + s1 dt^g: typical residual of true partners)
    iv   min(inview_s, 3)   least in-picture seconds of a hidden path
    tu   1 when a team label is missing
    ee   1 when a ends at an image border and b starts at one
    ex   1 when exactly one of the two is at an image border
    fv   min(fwd_inview_s, 3)   seconds a's flow-carried end point is in the picture in the gap
    bv   min(back_inview_s, 3)  the same for b's start point carried backwards
    fa, ba   fv, bv along the locally advected paths (fwd_adv_inview_s, back_adv_inview_s)
    lu2, ls2 lu and ls times ee (own weights for border-to-border pairs; tried, not used)
y_b ("nobody came before b"): 1 ("one"), b starts at an image border ("edge"),
    log(1 + seconds since clip start, capped at TIME_CAP_S) ("time"), team census ("census":
    features.census, can a player of this team still be unseen?; tried, not used)
y_a ("nobody comes after a"): 1, a ends at an image border, log(1 + seconds to clip end), census
Scores: mode "min" = min(P(a | b), P(b | a)); mode "sinkhorn" = the doubly normalised
assignment (rows and columns each sum to one with their "nobody" slack). Pairs with a score
>= theta are linked, best first, each piece at most one predecessor and one successor; with
peel=True the two slots of an accepted link leave the competition and the rest is re-scored
(peel_sequence / cut_sequence give the same links for a whole range of thresholds at once).
Only links of kind() >= emit are given out; the others still compete and take their slots.
links_grouped() is the variant with competition between groups of pieces (tried, no gain).

The delivered configuration is run_position_linker.CONFIG: residual r_adv_cv, pair evidence
(lu, ls, tu, ee, ex, fa, ba), context (one, edge, time), border groups weighted 1 + 3 in the
fit, mode "sinkhorn", peel, emit 1 or 2.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import optimize

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import features  # noqa: E402

INVIEW_CAP = 3.0
PAIR_FEATURES = ("lu", "ls", "iv", "tu", "ee", "ex", "fv", "bv")
CTX_FEATURES = ("one", "edge", "time", "census")
L2 = 1e-3
TIME_CAP_S = 30.0   # the time context is capped at the length of the dev clips (no effect on them)


def admissible(p: pd.DataFrame) -> pd.DataFrame:
    """Pairs the linker considers at all (hard, label-free rules)."""
    ok = p.reach & (p.same_role == 1) & (p.same_team != 0) & p.r.notna()
    return p[ok].copy()


def sigma(dt: np.ndarray, s) -> np.ndarray:
    return s[0] + s[1] * np.power(np.maximum(dt, 0.0), s[2])


def design(p: pd.DataFrame, s, resid: str, names) -> np.ndarray:
    sg = sigma(p.dt.to_numpy(float), s)
    u = p[resid].to_numpy(float) / sg
    ea, eb = p.a_end_edge.to_numpy(bool), p.b_start_edge.to_numpy(bool)
    cols = {
        "lu": np.log1p(u * u),
        "ls": np.log(sg),
        "iv": np.minimum(p.inview_s.to_numpy(float), INVIEW_CAP),
        "tu": (p.same_team.to_numpy() == -1).astype(float),
        "ee": (ea & eb).astype(float),
        "ex": (ea ^ eb).astype(float),
        "fv": np.minimum(p.fwd_inview_s.to_numpy(float), INVIEW_CAP),
        "bv": np.minimum(p.back_inview_s.to_numpy(float), INVIEW_CAP),
        "fa": np.minimum(p.fwd_adv_inview_s.to_numpy(float), INVIEW_CAP),
        "ba": np.minimum(p.back_adv_inview_s.to_numpy(float), INVIEW_CAP),
    }
    cols["lu2"] = cols["lu"] * cols["ee"]
    cols["ls2"] = cols["ls"] * cols["ee"]
    return np.column_stack([cols[n] for n in names]) if names else np.zeros((len(p), 0))


def context(p: pd.DataFrame, side: str, names) -> np.ndarray:
    """y_b (side "in") or y_a (side "out") per pair row."""
    if side == "in":
        edge = p.b_start_edge.to_numpy(float)
        tm = np.log1p(np.minimum(p.b_first.to_numpy(float) / features.FPS, TIME_CAP_S))
        cen = p.b_census_in.to_numpy(float)
    else:
        edge = p.a_end_edge.to_numpy(float)
        tm = np.log1p(np.minimum((p.n_frames.to_numpy(float) - 1 - p.a_last.to_numpy(float)).clip(min=0) / features.FPS, TIME_CAP_S))
        cen = p.a_census_out.to_numpy(float)
    cols = {"one": np.ones(len(p)), "edge": edge, "time": tm, "census": cen}
    return np.column_stack([cols[n] for n in names])


def _groups(keys: np.ndarray):
    uniq, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    return len(uniq), inv, first


def _lse(score: np.ndarray, grp: np.ndarray, const: np.ndarray) -> np.ndarray:
    """log(exp(const_g) + sum_{rows of group g} exp(score)) per group."""
    m = const.copy()
    np.maximum.at(m, grp, score)
    tot = np.exp(const - m)
    np.add.at(tot, grp, np.exp(score - m[grp]))
    return m + np.log(tot)


def _scores(z, gb, nb, cb, ga, na, ca, mode: str, iters: int = 200):
    """cb / ca: "nobody" log-weights per later / earlier piece. Returns p_in, p_out, score."""
    p_in = np.exp(z - _lse(z, gb, cb)[gb])
    p_out = np.exp(z - _lse(z, ga, ca)[ga])
    if mode == "min":
        return p_in, p_out, np.minimum(p_in, p_out)
    # Sinkhorn with slack: M_ab = u_a K_ab v_b, u_a (e^ca + sum_b K v) = 1, v_b (e^cb + sum_a K u) = 1
    lu, lv = np.zeros(na), np.zeros(nb)
    for _ in range(iters):
        lu = -_lse(z + lv[gb], ga, ca)
        lv_new = -_lse(z + lu[ga], gb, cb)
        if np.max(np.abs(lv_new - lv)) < 1e-6:
            lv = lv_new
            break
        lv = lv_new
    return p_in, p_out, np.exp(z + lu[ga] + lv[gb])


def posteriors(p: pd.DataFrame, params: dict, mode: str = "min") -> pd.DataFrame:
    """Admissible pairs of ONE clip with z, p_in = P(a | b), p_out = P(b | a), score."""
    p = admissible(p)
    if not len(p):
        return p.assign(z=[], p_in=[], p_out=[], score=[])
    z = design(p, params["s"], params["resid"], params["pair"]) @ np.asarray(params["w"])
    nb, gb, fb = _groups(p.b.to_numpy())
    na, ga, fa = _groups(p.a.to_numpy())
    cb = (context(p, "in", params["ctx"]) @ np.asarray(params["g_in"]))[fb]
    ca = (context(p, "out", params["ctx"]) @ np.asarray(params["g_out"]))[fa]
    p["z"] = z
    p["c_in"] = cb[gb]
    p["c_out"] = ca[ga]
    p["p_in"], p["p_out"], p["score"] = _scores(z, gb, nb, cb, ga, na, ca, mode)
    return p


def kind(q: pd.DataFrame) -> np.ndarray:
    """2 = a ends at an image border and b starts at one, 1 = one of the two, 0 = neither."""
    return q.a_end_edge.to_numpy(int) + q.b_start_edge.to_numpy(int)


def links(p: pd.DataFrame, params: dict, theta: float, mode: str = "min", peel: bool = False, emit: int = 0) -> list[list[int]]:
    """Links of one clip. ``emit``: least kind() of a link that is given out (2 = only pairs
    that left at a border and came back at one, 1 = at least one border, 0 = all). Pairs of
    a lower kind still compete, and with peel they still take their slots, but are not linked."""
    q = posteriors(p, params, mode)
    out: list[list[int]] = []
    if not len(q):
        return out
    q["kind"] = kind(q)
    if not peel:
        used_a, used_b = set(), set()
        for r in q.sort_values("score", ascending=False).itertuples():
            if r.score < theta:
                break
            if r.a in used_a or r.b in used_b:
                continue
            used_a.add(r.a)
            used_b.add(r.b)
            if r.kind >= emit:
                out.append([int(r.a), int(r.b)])
        return out
    while len(q):
        nb, gb, fb = _groups(q.b.to_numpy())
        na, ga, fa = _groups(q.a.to_numpy())
        _, _, sc = _scores(q.z.to_numpy(), gb, nb, q.c_in.to_numpy()[fb], ga, na, q.c_out.to_numpy()[fa], mode)
        i = int(np.argmax(sc))
        if sc[i] < theta:
            break
        a, b = int(q.a.iloc[i]), int(q.b.iloc[i])
        if int(q.kind.iloc[i]) >= emit:
            out.append([a, b])
        q = q[(q.a != a) & (q.b != b)]
    return out


def peel_sequence(p: pd.DataFrame, params: dict, mode: str = "sinkhorn", floor: float = 0.1) -> list[dict]:
    """The whole peeling order of one clip down to score ``floor``: [{a, b, score, kind}, ...].
    links(..., peel=True) at threshold theta >= floor is the prefix of this list up to the
    first entry with score < theta (see cut_sequence)."""
    q = posteriors(p, params, mode)
    out: list[dict] = []
    if not len(q):
        return out
    q["kind"] = kind(q)
    while len(q):
        nb, gb, fb = _groups(q.b.to_numpy())
        na, ga, fa = _groups(q.a.to_numpy())
        _, _, sc = _scores(q.z.to_numpy(), gb, nb, q.c_in.to_numpy()[fb], ga, na, q.c_out.to_numpy()[fa], mode)
        i = int(np.argmax(sc))
        if sc[i] < floor:
            break
        a, b = int(q.a.iloc[i]), int(q.b.iloc[i])
        out.append({"a": a, "b": b, "score": float(sc[i]), "kind": int(q.kind.iloc[i]), "z": float(q.z.iloc[i]),
                    "p_in": float(q.p_in.iloc[i]), "p_out": float(q.p_out.iloc[i])})
        q = q[(q.a != a) & (q.b != b)]
    return out


def cut_sequence(sequence: list[dict], theta: float, emit: int = 2) -> list[list[int]]:
    """Links at threshold ``theta`` from a peel_sequence: its entries before the first one
    scoring below theta, those of kind >= emit."""
    out = []
    for s in sequence:
        if s["score"] < theta:
            break
        if s["kind"] >= emit:
            out.append([s["a"], s["b"]])
    return out


def piece_context(seq: str, params: dict) -> pd.DataFrame:
    """Per piece: first, last, start_edge, end_edge and its "nobody" log-weights c_in, c_out."""
    import geom

    e = geom.piece_ends(seq)
    t = pd.DataFrame({"piece": e.piece.astype(int), "first": e["first"].astype(int), "last": e["last"].astype(int),
                      "start_edge": e.start_side != "", "end_edge": e.end_side != ""})
    names = params["ctx"]
    tin = np.log1p(np.minimum(t["first"].to_numpy(float) / features.FPS, TIME_CAP_S))
    tout = np.log1p(np.minimum((geom.cameras(seq)["n"] - 1 - t["last"].to_numpy(float)).clip(min=0) / features.FPS, TIME_CAP_S))
    cs = features.census(seq).set_index("piece")
    cin = {"one": np.ones(len(t)), "edge": t.start_edge.to_numpy(float), "time": tin, "census": t.piece.map(cs.census_in).to_numpy(float)}
    cout = {"one": np.ones(len(t)), "edge": t.end_edge.to_numpy(float), "time": tout, "census": t.piece.map(cs.census_out).to_numpy(float)}
    t["c_in"] = np.column_stack([cin[n] for n in names]) @ np.asarray(params["g_in"])
    t["c_out"] = np.column_stack([cout[n] for n in names]) @ np.asarray(params["g_out"])
    return t


def links_grouped(seq: str, params: dict, theta: float, emit: int = 2, mode: str = "sinkhorn", log: list | None = None) -> list[list[int]]:
    """Links of one clip with competition between GROUPS of pieces.

    Every piece starts as its own group. The evidence that group H continues group G (all of
    G before all of H) is the best pair evidence z_ab over a in G, b in H; the "nobody" terms
    are those of the first piece of H and of the last piece of G. Groups compete as in
    posteriors() (doubly normalised with ``mode`` "sinkhorn"). The best-scoring pair of groups
    is merged while its score is >= theta, then everything is re-scored: pieces already joined
    no longer split the evidence among themselves (a player whose track ends in a main piece
    and a two-row piece at the border is one candidate, described by his best piece).
    A merge is given out as the link (a*, b*) of its best pair when the earlier group ends at
    an image border and the later one starts at one (``emit`` = 2; 1 = at least one of the two;
    0 = always); other merges stay internal."""
    q = admissible(features.pair_features(seq))
    out: list[list[int]] = []
    if not len(q):
        return out
    z = design(q, params["s"], params["resid"], params["pair"]) @ np.asarray(params["w"])
    pc = piece_context(seq, params).set_index("piece")
    ids = pc.index.to_numpy()
    index = {int(p): i for i, p in enumerate(ids)}
    first, last = pc["first"].to_numpy(), pc["last"].to_numpy()
    start_edge, end_edge = pc.start_edge.to_numpy(), pc.end_edge.to_numpy()
    c_in, c_out = pc.c_in.to_numpy(), pc.c_out.to_numpy()
    ia = np.array([index[int(a)] for a in q.a])
    ib = np.array([index[int(b)] for b in q.b])
    n = len(ids)
    grp = np.arange(n)                     # group of each piece = index of its representative
    g_first, g_last = first.copy(), last.copy()
    g_head, g_tail = np.arange(n), np.arange(n)   # first / last piece of each group
    while True:
        ga, gb = grp[ia], grp[ib]
        ok = (ga != gb) & (g_last[ga] < g_first[gb])
        if not ok.any():
            break
        rows = np.flatnonzero(ok)
        key = ga[rows].astype(np.int64) * n + gb[rows]
        order = np.lexsort((-z[rows], key))
        rows, key = rows[order], key[order]
        keep = np.ones(len(rows), bool)
        keep[1:] = key[1:] != key[:-1]     # best pair of each (earlier group, later group)
        rows = rows[keep]
        za, A, B = z[rows], ga[rows], gb[rows]
        nb_, gb_, fb_ = _groups(B)
        na_, ga_, fa_ = _groups(A)
        cb = c_in[g_head[B[fb_]]]
        ca = c_out[g_tail[A[fa_]]]
        _, _, sc = _scores(za, gb_, nb_, cb, ga_, na_, ca, mode)
        i = int(np.argmax(sc))
        if sc[i] < theta:
            break
        G, H, r = int(A[i]), int(B[i]), int(rows[i])
        k = int(end_edge[g_tail[G]]) + int(start_edge[g_head[H]])
        if k >= emit:
            out.append([int(q.a.iloc[r]), int(q.b.iloc[r])])
        if log is not None:
            log.append({"a": int(q.a.iloc[r]), "b": int(q.b.iloc[r]), "score": float(sc[i]), "kind": k, "emitted": k >= emit})
        grp[grp == H] = G                  # merge H into G (G is wholly before H)
        g_last[G] = g_last[H]
        g_tail[G] = g_tail[H]
    return out


# --------------------------------------------------------------------------- fitting (labels)
def fit_sigma(T: pd.DataFrame, resid: str = "r"):
    """s0, s1, g of sigma(dt) = s0 + s1 dt^g: least absolute deviation of the log residual of
    the true partners T (labelled true immediate pairs of the fit clips)."""
    dt = T.dt.to_numpy(float)
    r = T[resid].to_numpy(float)
    ok = np.isfinite(r)
    dt, r = dt[ok], r[ok]

    def loss(q):
        s0, s1, g = np.exp(q[0]), np.exp(q[1]), 1.0 / (1.0 + np.exp(-q[2]))
        return float(np.abs(np.log(r + 0.3) - np.log(s0 + s1 * dt**g + 0.3)).mean())

    best = None
    for g0 in (-0.5, 0.5):
        res = optimize.minimize(loss, x0=[np.log(0.5), np.log(2.0), g0], method="Nelder-Mead",
                                options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    q = best.x
    return float(np.exp(q[0])), float(np.exp(q[1])), float(1.0 / (1.0 + np.exp(-q[2])))


def fit(clips, resid: str = "r", pair=("lu", "ls", "iv", "tu"), ctx=("one",), edge_weight: float = 0.0) -> dict:
    """Fit sigma and the conditional-logit weights on the labelled pairs of ``clips``.
    edge_weight: extra weight of the groups whose own piece starts (ends) at an image border."""
    import fitdata  # labels: fit clips only

    P = admissible(fitdata.pooled(clips))
    T = P[P.imm]
    s = fit_sigma(T, resid)
    X = design(P, s, resid, pair)
    Yin, Yout = context(P, "in", ctx), context(P, "out", ctx)
    nb, gb, fb = _groups((P.seq + "|" + P.b.astype(str)).to_numpy())
    na, ga, fa = _groups((P.seq + "|" + P.a.astype(str)).to_numpy())
    same = P.same.to_numpy(bool)
    # a group counts when its own piece has an identity (>= 5 matched rows); its target is
    # "one of the same-person candidates" or, without any, "nobody"
    use_b = np.zeros(nb, bool)
    use_b[gb[P.b_ref.notna().to_numpy()]] = True
    use_a = np.zeros(na, bool)
    use_a[ga[P.a_ref.notna().to_numpy()]] = True
    has_b = np.zeros(nb, bool)
    has_b[gb[same]] = True
    has_a = np.zeros(na, bool)
    has_a[ga[same]] = True
    k, m = X.shape[1], Yin.shape[1]
    neg = np.full(len(P), -1e9)
    wb = 1.0 + edge_weight * P.b_start_edge.to_numpy(float)[fb]
    wa = 1.0 + edge_weight * P.a_end_edge.to_numpy(float)[fa]

    def nll(q):
        w, g_in, g_out = q[:k], q[k:k + m], q[k + m:]
        z = X @ w
        total = 0.0
        for n, g, c, use, has, wt in ((nb, gb, (Yin @ g_in)[fb], use_b, has_b, wb), (na, ga, (Yout @ g_out)[fa], use_a, has_a, wa)):
            den = _lse(z, g, c)
            num = _lse(np.where(same, z, neg), g, np.full(n, -1e9))
            num = np.where(has, num, c)
            total += float(-((num - den) * wt)[use].sum())
        return total / (wb[use_b].sum() + wa[use_a].sum()) + L2 * float(w @ w)

    res = optimize.minimize(nll, np.zeros(k + 2 * m), method="L-BFGS-B")
    return {"s": [float(v) for v in s], "resid": resid, "pair": list(pair), "ctx": list(ctx),
            "w": [float(v) for v in res.x[:k]], "g_in": [float(v) for v in res.x[k:k + m]],
            "g_out": [float(v) for v in res.x[k + m:]], "edge_weight": float(edge_weight), "nll": float(res.fun),
            "n_true": int(len(T)), "n_pairs": int(len(P))}


def heldout_nll(params: dict, seq: str, edge_only: bool = False) -> tuple[float, int]:
    """LABELS (evaluation of a held-out clip): summed negative log-likelihood of the right
    answer for its groups and their number (edge_only: only the groups whose own piece starts,
    respectively ends, at an image border)."""
    import fitdata

    P = admissible(fitdata.labelled_pairs(seq))
    z = design(P, params["s"], params["resid"], params["pair"]) @ np.asarray(params["w"])
    same = P.same.to_numpy(bool)
    total, n_used = 0.0, 0
    for key, side, g_ in ((P.b.to_numpy(), "in", params["g_in"]), (P.a.to_numpy(), "out", params["g_out"])):
        n, g, f = _groups(key)
        c = (context(P, side, params["ctx"]) @ np.asarray(g_))[f]
        ref = (P.b_ref if side == "in" else P.a_ref).notna().to_numpy()
        if edge_only:
            ref = ref & (P.b_start_edge if side == "in" else P.a_end_edge).to_numpy(bool)
        use = np.zeros(n, bool)
        use[g[ref]] = True
        has = np.zeros(n, bool)
        has[g[same]] = True
        den = _lse(z, g, c)
        num = _lse(np.where(same, z, -1e9), g, np.full(n, -1e9))
        num = np.where(has, num, c)
        total += float(-(num - den)[use].sum())
        n_used += int(use.sum())
    return total, n_used


def loco_params(clips=data.DEV_CLIPS, **kw) -> dict:
    """{held-out clip: params fitted on the other clips}."""
    return {held: fit(fit_clips, **kw) for fit_clips, held in data.loco_folds(clips)}


def method_links(params_by_clip: dict, theta: float, mode: str = "min", peel: bool = False, emit: int = 0) -> dict:
    """peel = "group": links_grouped (competition between groups); True / False: links()."""
    if peel == "group":
        return {seq: {"links": links_grouped(seq, pr, theta, emit, mode)} for seq, pr in params_by_clip.items()}
    return {seq: {"links": links(features.pair_features(seq), pr, theta, mode, peel, emit)} for seq, pr in params_by_clip.items()}
