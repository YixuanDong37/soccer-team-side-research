"""Label-free appearance features for pairs of gbf pieces (m_appear), for the fusion step.

    import sys; sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_appear")
    from features import pair_features
    df = pair_features("SNGS-162")                 # one row per candidate pair (a earlier, b later)
    df = pair_features("SNGS-162", cuts=[[53, 400]])   # the same after cuts (reid_lab.score's id convention)

Nothing here reads an annotation or reid_lab.score. Inputs: reid_lab.data only.

Prototype of a piece: mean of the unit PRTReid embeddings of its clean rows (box >= 40 px tall, not
touching the image edge, overlapping no other box); with fewer than MIN_CLEAN clean rows, of all its
rows that have an embedding (low_a / low_b flag). Three cosines are computed from it:
- cos_raw: between the unit prototypes (what analysis/appearance_match.py does);
- cos_tc:  after subtracting the team mean (mean of the prototype rows of the pieces labelled player of
           that team; a piece without a team label takes the team whose mean is nearer). Same-kit players
           share most of the raw vector; the remainder is what tells them apart;
- cos_w:   team-centred, then whitened with the within-piece covariance of the clip (WCCN, shrinkage
           WCCN_LAMBDA): directions in which one piece varies from frame to frame count less.

Candidate pairs (a, b): neither piece is a referee, a ends before b starts (no time overlap), both have
a prototype, same appearance team (label, or nearer team mean when the label is missing).
No limit on the gap, no verified-team requirement, no minimum number of clean rows.

Columns of pair_features(seq):
  a, b                       piece ids (a earlier)
  gap                        frames between a's last and b's first row
  cos_raw, cos_tc, cos_w     see above
  -- among the time-ordered candidates (a before b):
  rank_fwd_*, rank_bwd_*     rank of b among a's later candidates / of a among b's earlier candidates (1 = best)
  marg_fwd_*, marg_bwd_*     cosine minus the best other candidate on that side (negative when not the best;
                             NaN when there is no other candidate)
  rival_fwd_*, rival_bwd_*   like marg, but only against rivals that share a frame with the partner (so they are
                             certainly not the partner's person); NaN without such a rival
  n_fwd, n_bwd               number of candidates on each side
  -- among all allowed partners in either time order (same appearance team, no shared frame), which is
     what agglo.agglomerate tests on single pieces:
  umutual_*                  b is a's most similar allowed piece and a is b's
  urival_a_*, urival_b_*     cosine minus a's (b's) best cosine to an allowed piece that shares a frame with
                             b (with a); NaN without such a rival. The agglomeration links when both are >= margin.
  -- other-person references:
  imp_a_*, imp_b_*           highest cosine of a (of b) to a same-team piece seen together with it for >= COVISIBLE
                             frames (a certain other person); NaN when none
  zn_*                       cosine standardised against those certain-other-person cosines of a and of b
                             (mean of the two z-scores; NaN when either side has < 3 of them)
  -- what m_appear's own rule concludes:
  agglo_margin_w, agglo_margin_tc   the largest rival margin (0.40 ... 0.00, t_abs 0.5) at which the agglomeration
                             puts a and b into one group; NaN when it never does. A confidence tier: >= 0.25 is the
                             "balanced" operating point, >= 0.30 "cautious". (only with tiers=True)
  -- piece facts:
  rows_a, rows_b, clean_a, clean_b, low_a, low_b, medh_a, medh_b
  same_plugin                the plugin tracked both as one id
  team_a, team_b             team label (None when missing); ateam = appearance team used for the candidate rule
  vteam_a, vteam_b           gbf's verified team (None when not verified)
  role_a, role_b
  a_ends_at_edge, b_starts_at_edge
  dist_m, reach_ok           pitch distance between a's end and b's start; within 12 m/s * gap + 4 m (True when unknown)
"""
import os
import sys
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data  # noqa: E402

from core import FPS, ctx, norm_cuts  # noqa: E402

MIN_CLEAN = 5        # clean rows needed to use only clean rows (appearance_match.py's MIN_CLEAN)
COVISIBLE = 5        # frames seen together to count as certainly two people (appearance_match.py's COVISIBLE)
WCCN_LAMBDA = 0.3    # shrinkage of the within-piece covariance (share of its mean eigenvalue)
WCCN_MIN_ROWS = 10
SPEED, POS_ERR = 12.0, 2.0
KINDS = ("raw", "tc", "w")
TIER_MARGINS = (0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0)
TIER_T_ABS = 0.5


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


class Protos:
    """Prototypes of every piece of a clip (label-free)."""

    def __init__(self, seq, min_clean=MIN_CLEAN, row_sel=None, lam=WCCN_LAMBDA, cuts=()):
        c = ctx(seq, cuts)
        self.c = c
        P = c.P
        E = np.where(c.has[:, None], c.E, 0.0)
        sel = c.clean if row_sel is None else row_sel
        mean, low, used = {}, {}, {}
        for pid, rows in c.rows_of.items():
            r = rows[sel[rows] & c.has[rows]]
            low[pid] = len(r) < min_clean
            if low[pid]:
                r = rows[c.has[rows]]
            if len(r):
                mean[pid] = E[r].mean(axis=0)
                used[pid] = r
        self.mean, self.low, self.used = mean, low, used
        tm = {}                                           # team means from the pieces labelled player
        for t_ in ("left", "right"):
            rows = [used[int(p)] for p, r, tt in zip(P.piece, P.role, P.team)
                    if r == "player" and tt == t_ and int(p) in used and not low[int(p)]]
            rows = np.concatenate(rows) if rows else np.zeros(0, int)
            tm[t_] = E[rows].mean(axis=0) if len(rows) >= 50 else None
        known = [v for v in tm.values() if v is not None]
        glob = np.mean(known, axis=0) if known else E[c.has].mean(axis=0)
        self.team_mean, self.glob = tm, glob
        ateam, centre = {}, {}
        for pid, m in mean.items():
            lab = P.team[c.index[pid]]
            cands = [(float(unit(m) @ unit(v)), k) for k, v in tm.items() if v is not None]
            near = max(cands)[1] if cands else None
            team = lab if lab in ("left", "right") else near
            ateam[pid] = team
            centre[pid] = tm[team] if team is not None and tm.get(team) is not None else glob
        self.ateam = ateam
        self.raw = {p: unit(m) for p, m in mean.items()}
        self.tc = {p: unit(m - centre[p]) for p, m in mean.items()}
        D = [E[r] - E[r].mean(axis=0) for p, r in used.items() if not low[p] and len(r) >= WCCN_MIN_ROWS]
        if D:                                             # within-piece covariance: rows around their piece's mean
            D = np.concatenate(D)
            Sw = D.T @ D / len(D)
            w, V = np.linalg.eigh(Sw)
            T = (V / np.sqrt(np.clip(w, 0, None) + lam * w.mean())) @ V.T
        else:
            T = np.eye(E.shape[1])
        self.T = T
        self.w = {p: unit(T @ (m - centre[p])) for p, m in mean.items()}

    def get(self, kind):
        return {"raw": self.raw, "tc": self.tc, "w": self.w}[kind]


@lru_cache(maxsize=None)
def _protos(seq, cuts):
    return Protos(seq, cuts=cuts)


def protos(seq, cuts=()):
    return _protos(seq, norm_cuts(cuts))


def candidate_ids(seq, pr=None, cuts=()):
    """Pieces that may take part in a link: not a referee, with a prototype."""
    c = ctx(seq, cuts)
    pr = pr or protos(seq, cuts)
    return [int(p) for p, r in zip(c.P.piece, c.P.role) if r != "referee" and int(p) in pr.mean]


def _side_stats(S, ok, covis_cols):
    """For every allowed (i, j): rank of j in row i, margin to the best other allowed j', margin to the
    best other allowed j' that is co-visible with j, and the count of allowed entries in the row."""
    n, m = S.shape
    Sm = np.where(ok, S, -np.inf)
    order = np.argsort(-Sm, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(m)[None, :] + 1
    cnt = ok.sum(axis=1)
    best = Sm[np.arange(n), order[:, 0]]
    second = Sm[np.arange(n), order[:, 1]] if m > 1 else np.full(n, -np.inf)
    other = np.where(rank == 1, second[:, None], best[:, None])
    with np.errstate(invalid="ignore"):
        marg = np.where(np.isfinite(other), S - other, np.nan)
    rival = np.full(S.shape, np.nan)
    for j in range(m):
        cols = covis_cols[j]
        if len(cols):
            v = Sm[:, cols].max(axis=1)
            with np.errstate(invalid="ignore"):
                rival[:, j] = np.where(np.isfinite(v), S[:, j] - v, np.nan)
    return rank, marg, rival, cnt


def agglo_tiers(seq, kind="w", cuts=()):
    """{(a, b): largest margin at which agglo.agglomerate (t_abs TIER_T_ABS) puts a and b into one group}."""
    from agglo import agglomerate
    out = {}
    for m in TIER_MARGINS:                                # descending: the first hit is the largest margin
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in agglomerate(seq, kind=kind, t_abs=TIER_T_ABS, margin=m, cuts=cuts):
            parent[find(a)] = find(b)
        groups = {}
        for x in list(parent):
            groups.setdefault(find(x), []).append(x)
        for g in groups.values():
            for a in g:
                for b in g:
                    if a != b:
                        out.setdefault((a, b), m)
    return out


@lru_cache(maxsize=None)
def _pair_features(seq, cuts, tiers):
    c = ctx(seq, cuts)
    pr = protos(seq, cuts)
    P = c.P
    ids = candidate_ids(seq, pr, cuts)
    n = len(ids)
    ix = [c.index[p] for p in ids]
    first = P["first"].to_numpy()[ix]
    last = P["last"].to_numpy()[ix]
    ateam = np.array([pr.ateam[p] or "" for p in ids])
    covis = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if first[i] <= last[j] and first[j] <= last[i]:
                covis[i, j] = covis[j, i] = c.overlap_frames(ids[i], ids[j])
    same_team = ateam[:, None] == ateam[None, :]
    ok = (last[:, None] < first[None, :]) & same_team
    certain = (covis >= COVISIBLE) & same_team          # certainly two people of one team
    covis_any = covis > 0
    allowed = same_team & ~covis_any                    # either time order, as the agglomeration sees it
    np.fill_diagonal(allowed, False)
    ii, jj = np.nonzero(ok)
    out = {"a": np.array(ids, dtype=np.int64)[ii], "b": np.array(ids, dtype=np.int64)[jj], "gap": first[jj] - last[ii]}
    cols = [np.nonzero(covis_any[j] & same_team[j])[0] for j in range(n)]
    for kind in KINDS:
        M = np.stack([pr.get(kind)[p] for p in ids]) if n else np.zeros((0, 256))
        S = M @ M.T
        out[f"cos_{kind}"] = S[ii, jj]
        rank, marg, rival, cnt = _side_stats(S, ok, cols)
        out[f"rank_fwd_{kind}"] = rank[ii, jj]
        out[f"marg_fwd_{kind}"] = marg[ii, jj]
        out[f"rival_fwd_{kind}"] = rival[ii, jj]
        rank, marg, rival, cntb = _side_stats(S.T, ok.T, cols)
        out[f"rank_bwd_{kind}"] = rank[jj, ii]
        out[f"marg_bwd_{kind}"] = marg[jj, ii]
        out[f"rival_bwd_{kind}"] = rival[jj, ii]
        if kind == "raw":
            out["n_fwd"] = cnt[ii]
            out["n_bwd"] = cntb[jj]
        # unordered view (the agglomeration's first step)
        Sa = np.where(allowed, S, -np.inf)
        best = Sa.argmax(axis=1) if n else np.zeros(0, int)
        isbest = np.zeros((n, n), bool)
        isbest[np.arange(n), best] = np.isfinite(Sa[np.arange(n), best]) if n else False
        out[f"umutual_{kind}"] = (isbest & isbest.T)[ii, jj]
        Om = np.where(covis_any, 0.0, -np.inf)
        rv = (Sa[:, None, :] + Om[None, :, :]).max(axis=2) if n else np.zeros((0, 0))     # [g, h]: g's best among h's co-visible
        with np.errstate(invalid="ignore"):
            ur = np.where(np.isfinite(rv), S - rv, np.nan)
        out[f"urival_a_{kind}"] = ur[ii, jj]
        out[f"urival_b_{kind}"] = ur[jj, ii]
        Si = np.where(certain, S, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imax = np.nanmax(Si, axis=1) if n else np.zeros(0)
            imean = np.nanmean(Si, axis=1) if n else np.zeros(0)
            istd = np.nanstd(Si, axis=1) if n else np.zeros(0)
        icnt = certain.sum(axis=1)
        out[f"imp_a_{kind}"] = imax[ii]
        out[f"imp_b_{kind}"] = imax[jj]
        with np.errstate(invalid="ignore"):
            za = np.where(icnt[ii] >= 3, (S[ii, jj] - imean[ii]) / np.maximum(istd[ii], 1e-6), np.nan)
            zb = np.where(icnt[jj] >= 3, (S[ii, jj] - imean[jj]) / np.maximum(istd[jj], 1e-6), np.nan)
        out[f"zn_{kind}"] = (za + zb) / 2
    df = pd.DataFrame(out)
    if tiers:
        for kind in ("w", "tc"):
            tier = agglo_tiers(seq, kind, cuts)
            df[f"agglo_margin_{kind}"] = [tier.get((int(a), int(b)), np.nan) for a, b in zip(df.a, df.b)]
    meta = P.set_index("piece")
    for side in ("a", "b"):
        m = meta.loc[df[side]]
        df[f"rows_{side}"] = m.rows.to_numpy()
        df[f"clean_{side}"] = m.n_clean.to_numpy()
        df[f"low_{side}"] = [pr.low[int(p)] for p in df[side]]
        df[f"medh_{side}"] = m.med_h.to_numpy()
        df[f"team_{side}"] = m.team.to_numpy()
        df[f"vteam_{side}"] = m.vteam.to_numpy()
        df[f"role_{side}"] = m.role.to_numpy()
        df[f"plugin_{side}"] = m.plugin.to_numpy()
    df["ateam"] = [pr.ateam[int(p)] for p in df.a]
    df["same_plugin"] = df.plugin_a.to_numpy() == df.plugin_b.to_numpy()
    df["a_ends_at_edge"] = meta.loc[df.a].ends_at_edge.to_numpy()
    df["b_starts_at_edge"] = meta.loc[df.b].starts_at_edge.to_numpy()
    exy, sxy = meta.end_xy.to_dict(), meta.start_xy.to_dict()
    dist = [np.hypot(sxy[b][0] - exy[a][0], sxy[b][1] - exy[a][1]) if exy[a] is not None and sxy[b] is not None else np.nan
            for a, b in zip(df.a, df.b)]
    df["dist_m"] = dist
    df["reach_ok"] = ~(df.dist_m > SPEED * df.gap / FPS + 2 * POS_ERR)
    return df.drop(columns=["plugin_a", "plugin_b"])


def pair_features(seq: str, cuts=(), tiers: bool = False) -> pd.DataFrame:
    """One row per candidate pair of pieces of a dev clip (see the module docstring). Label-free.
    cuts: [[piece, frame], ...] applied first (reid_lab.score's sub-piece ids). tiers=True adds the
    agglo_margin_* columns (about 3 s per clip). The returned frame is cached: copy before changing it."""
    return _pair_features(seq, norm_cuts(cuts), bool(tiers))


if __name__ == "__main__":
    import time
    t0 = time.time()
    for s in data.DEV_CLIPS:
        d = pair_features(s, tiers=True)
        print(s, len(d), "pairs; in one group at margin >= 0.25 (w):", int((d.agglo_margin_w >= 0.25).sum()), flush=True)
    print(round(time.time() - t0, 1), "s")
    print(d.head(2).T)
