"""Label-free position / motion / camera-view features for every ordered pair of pieces.

    import sys; sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
    import features
    df = features.pair_features("SNGS-107")     # one row per (a, b) with a.last < b.first

Reads only reid_lab.data (tracks.parquet, calib.parquet). No labels, no appearance, and no
plugin identity is used in any "position" column (same_plugin is given as a separate column
for the fusion step; the position linker does not read it).

Columns (a = earlier piece, b = later piece; both gbf piece ids)
  a, b                 piece ids, a.last < b.first
  gap                  frames between a's last and b's first row (b.first - a.last - 1)
  dt                   seconds from a's last to b's first frame
  d, dx, dy            metres between a's end point and b's start point (b - a), from robust
                       line fits over each piece's last / first second (geom.piece_ends)
  sx, sy               collective shift of the visible outfield players between the two frames:
                       the per-frame median velocity of all tracked players, summed over the gap
  r, rx, ry            flow-compensated residual: (dx - sx, dy - sy) and its length; the place
                       where b appeared against where a would be had he moved with everyone else
  r_fwd, r_back        residuals under LOCAL advection: the end point of a carried forward to
                       the first frame of b by the velocity field of the visible players around
                       it (Gaussian kernel ADV_KERNEL_M, shrunk to the global flow), against the
                       start of b; and the start point of b carried back, against the end of a
  r_adv, r_adv_cv      their mean; and the mean residual vector with the own-velocity term of r_cv
  fwd_adv_inview_s, back_adv_inview_s   fwd_inview_s / back_inview_s along the advected paths
  r_cv                 residual when a additionally keeps his own end velocity relative to the
                       flow for up to CV_HORIZON_S (and b his start velocity backwards)
  speed_req            m/s needed to cover d less 2 m in dt
  reach                False when no path within MAX_SPEED (+ START_RADIUS at both ends)
                       connects the two points in the gap
  inview_s             the least time (s) a person going from a's end to b's start within
                       MAX_SPEED must have spent wholly inside the picture (feet and head at
                       least VIEW_MARGIN_PX inside) during the gap: 0 = he can have stayed out
                       of sight all the time (inf when reach is False). Min-plus dynamic
                       programme over a 1 m grid in steps of STEP frames.
  line_inview          share of the gap frames on which the straight constant-speed path
                       between the two points is inside the picture
  fwd_inview_s         seconds of the gap during which a's end point, carried along with the
                       collective flow, lies inside the picture: had he just moved with everyone
                       else he would have been seen that long before b appeared
  back_inview_s        the same for b's start point carried backwards through the gap
  a_end_margin, b_start_margin   px: signed distance of a's end point (at a.last) and b's
                       start point (at b.first) from the image border (> 0 inside)
  a_end_edge, b_start_edge       the last box of a / first box of b touches an image border
  a_end_side, b_start_side       which borders (letters of L R T B)
  same_team            1 same, 0 different, -1 unknown (a piece without a team label)
  same_role            1 / 0
  same_plugin          1 when both pieces come from one plugin track (NOT position evidence)
  a_rows, b_rows       rows of each piece
  a_last, b_first, a_first, b_last   frames; n_frames = frames of the clip
  b_census_in          per later piece, in [0, 1]: upper bound on the chance that a player of
                       its team who comes into the picture at b.first was never seen before
                       (team census, see census()); 0.5 without a team label
  a_census_out         the same for the earlier piece looking forward from a.last
"""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import geom  # noqa: E402

FPS = geom.FPS
GRID_M = 1.0              # grid cell of the reachability programme
GRID_X = (-62.0, 62.0)    # the pitch (105 x 68) and 9.5 / 10 m around it
GRID_Y = (-44.0, 44.0)
STEP = 10                 # frames per step of the programme (0.4 s)
MAX_SPEED = 8.0           # m/s: speed bound of the hidden path
START_RADIUS = 3.0        # m: slack around both end points (position error)
VIEW_MARGIN_PX = 20.0     # a person is "in the picture" when feet and head are this far inside
FLOW_HALF = 6             # frames: half chord of a row velocity
FLOW_MIN_PLAYERS = 3      # a frame needs this many player velocities for a flow
CV_HORIZON_S = 1.0        # own (relative) velocity is extrapolated this long at most
ADV_KERNEL_M = 15.0       # Gaussian kernel width of the local velocity field
ADV_SHRINK = 0.5          # pseudo-weight of the global median flow in the local field


# --------------------------------------------------------------------------- collective flow
@lru_cache(maxsize=None)
def flow(seq: str) -> np.ndarray:
    """S (n_frames, 2): cumulative collective displacement in metres. S[t2] - S[t1] is how far
    the visible outfield players moved together between frames t1 and t2 (per-frame median of
    the row velocities of all tracked players, 0 on frames with fewer than FLOW_MIN_PLAYERS)."""
    t = data.tracks(seq)
    n = geom.cameras(seq)["n"]
    t = t[(t.track_id >= 0) & t.pitch_usable & t.pitch_x.notna() & (t.role == "player")]
    t = t.sort_values(["track_id", "frame"])
    vel = [[] for _ in range(n)]
    for _, g in t.groupby("track_id", sort=False):
        f = g.frame.to_numpy(np.int64)
        if len(f) < 2 * FLOW_HALF + 1:
            continue
        xy = g[["pitch_x", "pitch_y"]].to_numpy(float)
        pos = {int(ff): i for i, ff in enumerate(f)}
        for ff in f:
            i, j = pos.get(int(ff) - FLOW_HALF), pos.get(int(ff) + FLOW_HALF)
            if i is None or j is None:
                continue
            v = (xy[j] - xy[i]) / (2 * FLOW_HALF / FPS)
            if np.hypot(*v) < 12.0:
                vel[int(ff)].append(v)
    F = np.zeros((n, 2))
    for f in range(n):
        if len(vel[f]) >= FLOW_MIN_PLAYERS:
            F[f] = np.median(np.array(vel[f]), axis=0)
    return np.cumsum(F, axis=0) / FPS


# --------------------------------------------------------------------------- local advection
@lru_cache(maxsize=None)
def _player_velocities(seq: str):
    """Per frame: (positions (k, 2), velocities (k, 2)) of the tracked players with a chord."""
    t = data.tracks(seq)
    n = geom.cameras(seq)["n"]
    t = t[(t.track_id >= 0) & t.pitch_usable & t.pitch_x.notna() & (t.role == "player")]
    t = t.sort_values(["track_id", "frame"])
    pos = [[] for _ in range(n)]
    vel = [[] for _ in range(n)]
    for _, g in t.groupby("track_id", sort=False):
        f = g.frame.to_numpy(np.int64)
        if len(f) < 2 * FLOW_HALF + 1:
            continue
        xy = g[["pitch_x", "pitch_y"]].to_numpy(float)
        at = {int(ff): i for i, ff in enumerate(f)}
        for k, ff in enumerate(f):
            i, j = at.get(int(ff) - FLOW_HALF), at.get(int(ff) + FLOW_HALF)
            if i is None or j is None:
                continue
            v = (xy[j] - xy[i]) / (2 * FLOW_HALF / FPS)
            if np.hypot(*v) < 12.0:
                pos[int(ff)].append(xy[k])
                vel[int(ff)].append(v)
    return [np.array(p).reshape(-1, 2) for p in pos], [np.array(v).reshape(-1, 2) for v in vel]


def _local_velocity(x: np.ndarray, pos: np.ndarray, vel: np.ndarray, glob: np.ndarray) -> np.ndarray:
    """Velocity field at points x (m, 2): kernel mean of the player velocities around each
    point (Gaussian, ADV_KERNEL_M), shrunk to the global median flow ``glob`` of the frame."""
    if len(pos) == 0:
        return np.tile(glob, (len(x), 1))
    d2 = ((x[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2)
    k = np.exp(-d2 / (2.0 * ADV_KERNEL_M**2))
    return (k @ vel + ADV_SHRINK * glob) / (k.sum(axis=1, keepdims=True) + ADV_SHRINK)


@lru_cache(maxsize=None)
def advected(seq: str):
    """(fwd, back): fwd[piece] = (n, 2) path of the end point of the piece carried forward by
    the local velocity field of the visible players (NaN before its last frame); back[piece]
    the start point carried backwards (NaN after its first frame)."""
    e = geom.piece_ends(seq)
    n = geom.cameras(seq)["n"]
    pos, vel = _player_velocities(seq)
    S = flow(seq)
    G = np.vstack([S[:1], np.diff(S, axis=0)]) * FPS      # global median velocity per frame
    ids = e.piece.to_numpy()
    ends = e[["end_x", "end_y"]].to_numpy(float)
    starts = e[["start_x", "start_y"]].to_numpy(float)
    last = e["last"].to_numpy()
    first = e["first"].to_numpy()
    F = np.full((len(e), n, 2), np.nan)
    B = np.full((len(e), n, 2), np.nan)
    for i in range(len(e)):
        F[i, last[i]] = ends[i]
        B[i, first[i]] = starts[i]
    ok_e = np.isfinite(ends).all(axis=1)
    ok_s = np.isfinite(starts).all(axis=1)
    for t in range(1, n):          # forward: x(t) = x(t-1) + v(x(t-1), t-1) / fps
        act = np.flatnonzero((last < t) & ok_e)
        if len(act):
            prev = F[act, t - 1]
            F[act, t] = prev + _local_velocity(prev, pos[t - 1], vel[t - 1], G[t - 1]) / FPS
    for t in range(n - 2, -1, -1):  # backward: x(t) = x(t+1) - v(x(t+1), t+1) / fps
        act = np.flatnonzero((first > t) & ok_s)
        if len(act):
            nxt = B[act, t + 1]
            B[act, t] = nxt - _local_velocity(nxt, pos[t + 1], vel[t + 1], G[t + 1]) / FPS
    return {int(p): F[i] for i, p in enumerate(ids)}, {int(p): B[i] for i, p in enumerate(ids)}


@lru_cache(maxsize=None)
def advected_inview(seq: str):
    """Cumulative in-picture frame counts of the advected paths (see carried_inview)."""
    cam = geom.cameras(seq)
    F, B = advected(seq)
    out_f, out_b = {}, {}
    for src_, dst in ((F, out_f), (B, out_b)):
        for p, path in src_.items():
            ok = np.isfinite(path).all(axis=1)
            m = np.zeros(len(path), bool)
            if ok.any():
                m[ok] = geom.view_margin(cam["P"][ok], path[ok]) >= VIEW_MARGIN_PX
            dst[p] = np.cumsum(m)
    return out_f, out_b


# --------------------------------------------------------------------------- visibility grid
def _grid():
    xs = np.arange(GRID_X[0], GRID_X[1] + 1e-9, GRID_M)
    ys = np.arange(GRID_Y[0], GRID_Y[1] + 1e-9, GRID_M)
    return xs, ys


@lru_cache(maxsize=None)
def view_steps(seq: str) -> np.ndarray:
    """V (n_steps, ny, nx): share of each step's frames on which a person standing on the grid
    cell is wholly in the picture (frames without a camera count as not in the picture: the
    programme then cannot say he must have been seen)."""
    cam = geom.cameras(seq)
    xs, ys = _grid()
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx, gy], axis=-1)  # (ny, nx, 2)
    n = cam["n"]
    n_steps = int(np.ceil(n / STEP))
    V = np.zeros((n_steps, len(ys), len(xs)), np.float32)
    cnt = np.zeros(n_steps)
    for f in range(n):
        s = f // STEP
        cnt[s] += 1
        if not cam["ok"][f]:
            continue
        m = geom.view_margin(cam["P"][f][None, None], pts)
        V[s] += (m >= VIEW_MARGIN_PX)
    V /= np.maximum(cnt, 1)[:, None, None]
    return V


def _disc(radius_cells: float) -> np.ndarray:
    k = int(np.floor(radius_cells))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells + 1e-9


def _cell(x: float, y: float):
    xs, ys = _grid()
    i = int(np.clip(np.rint((y - ys[0]) / GRID_M), 0, len(ys) - 1))
    j = int(np.clip(np.rint((x - xs[0]) / GRID_M), 0, len(xs) - 1))
    return i, j


@lru_cache(maxsize=None)
def hidden_path_costs(seq: str) -> dict:
    """{(a, b): least in-picture seconds of a speed-bounded path from a's end to b's start}
    for every pair with a.last < b.first and both points known (inf = out of reach)."""
    e = geom.piece_ends(seq)
    V = view_steps(seq)
    n_steps = V.shape[0]
    xs, ys = _grid()
    step_s = STEP / FPS
    move = _disc(MAX_SPEED * step_s / GRID_M)
    slack = _disc(START_RADIUS / GRID_M)
    big = np.float32(1e6)
    starts = e.sort_values("first")
    out: dict = {}
    for a in e.itertuples():
        if not (np.isfinite(a.end_x) and np.isfinite(a.end_y)):
            continue
        later = starts[starts["first"] > a.last]
        if not len(later):
            continue
        cost = np.full((len(ys), len(xs)), big, np.float32)
        i, j = _cell(a.end_x, a.end_y)
        cost[i, j] = 0.0
        cost = ndimage.minimum_filter(cost, footprint=slack, mode="constant", cval=big)
        s = a.last // STEP          # the step holding a's last frame: he is seen there, no cost
        by_step: dict = {}
        for b in later.itertuples():
            if np.isfinite(b.start_x) and np.isfinite(b.start_y):
                by_step.setdefault(b.first // STEP, []).append(b)
        if not by_step:
            continue
        last_step = max(by_step)
        first_round = True
        while s <= last_step:
            # cost = least in-picture time to stand on each cell during step s; one move further:
            reach = ndimage.minimum_filter(cost, footprint=move, mode="constant", cval=big)
            targets = by_step.get(s + 1, []) + (by_step.get(s, []) if first_round else [])
            for b in targets:  # b is seen in its own step: that step's visibility is not counted
                bi, bj = _cell(b.start_x, b.start_y)
                v = _local_min(reach, bi, bj, slack)
                out[(int(a.piece), int(b.piece))] = float(v) if v < big / 2 else float("inf")
            first_round = False
            s += 1
            if s >= n_steps:
                break
            cost = np.minimum(reach + V[s] * np.float32(step_s), big)
    return out


def _local_min(arr: np.ndarray, i: int, j: int, footprint: np.ndarray) -> float:
    k = footprint.shape[0] // 2
    i0, i1 = max(0, i - k), min(arr.shape[0], i + k + 1)
    j0, j1 = max(0, j - k), min(arr.shape[1], j + k + 1)
    sub = arr[i0:i1, j0:j1]
    fp = footprint[(i0 - i + k):(i1 - i + k), (j0 - j + k):(j1 - j + k)]
    return float(sub[fp].min())


# --------------------------------------------------------------------------- flow-carried visibility
@lru_cache(maxsize=None)
def carried_inview(seq: str):
    """(fwd, back): dicts piece -> cumulative count over frames t of "the piece's end (start)
    point carried with the collective flow to frame t is inside the picture". fwd[a][t] counts
    frames a.last < t' <= t; back[b][t] counts frames t' <= t (all before b.first matter)."""
    e = geom.piece_ends(seq)
    cam = geom.cameras(seq)
    S = flow(seq)
    n = cam["n"]
    fwd, back = {}, {}
    for r in e.itertuples():
        for name, t0, x, y, store in (("end", r.last, r.end_x, r.end_y, fwd), ("start", r.first, r.start_x, r.start_y, back)):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            xy = np.array([x, y]) + S - S[t0]
            m = geom.view_margin(cam["P"], xy) >= VIEW_MARGIN_PX
            if name == "end":
                m[: t0 + 1] = False
            else:
                m[t0:] = False
            store[int(r.piece)] = np.cumsum(m)
    return fwd, back


# --------------------------------------------------------------------------- team census
CENSUS_TEAM_SIZE = 10     # outfield players of a team
CENSUS_TOP = 5            # the robust maximum is the 5th largest per-frame count


@lru_cache(maxsize=None)
def census(seq: str) -> pd.DataFrame:
    """Per piece: census_in and census_out in [0, 1] (0.5 when the piece has no team label or
    is not a player).

    census_in = (10 - most players of its team seen at once before its first frame)
                / (10 - players of its team in the picture at its first frame, itself excluded):
    an upper bound on the chance that a player of this team who now comes into the picture
    has not been seen before (0 = everybody of the team has been seen: he must be a return).
    census_out is the same looking forward from its last frame (0 = everybody of the team is
    seen at once later on: he comes back within the clip unless he is among them already).
    Counts are distinct track ids with role player and that team label per frame; the
    "most seen at once" is the CENSUS_TOP-th largest count (robust to a few duplicate boxes)."""
    e = geom.piece_ends(seq)
    t = data.tracks(seq)
    t = t[(t.track_id >= 0) & (t.role == "player") & t.team.notna()]
    n = geom.cameras(seq)["n"]
    cnt = {}
    for team, g in t.groupby("team"):
        c = np.zeros(n, int)
        k = g.groupby("frame").track_id.nunique()
        c[k.index.to_numpy()] = k.to_numpy()
        cnt[team] = c

    def top(c: np.ndarray) -> int:
        if len(c) == 0:
            return 0
        return int(np.sort(c)[-CENSUS_TOP:].min()) if len(c) >= CENSUS_TOP else int(c.max())

    rows = []
    for r in e.itertuples():
        cin = cout = 0.5
        if r.role == "player" and r.team in cnt:
            c = cnt[r.team]
            seen = top(c[: r.first])
            vis = max(int(c[r.first]) - 1, 0)
            cin = float(np.clip(max(CENSUS_TEAM_SIZE - seen, 0) / max(CENSUS_TEAM_SIZE - vis, 1), 0.0, 1.0))
            seen = top(c[r.last + 1:])
            vis = max(int(c[r.last]) - 1, 0)
            cout = float(np.clip(max(CENSUS_TEAM_SIZE - seen, 0) / max(CENSUS_TEAM_SIZE - vis, 1), 0.0, 1.0))
        rows.append({"piece": int(r.piece), "census_in": cin, "census_out": cout})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- pair table
def _line_inview(seq: str, p: pd.DataFrame, samples: int = 12) -> np.ndarray:
    """Share of sampled gap frames on which the straight constant-speed path is in the picture."""
    cam = geom.cameras(seq)
    out = np.zeros(len(p))
    ta, tb = p.a_last.to_numpy(), p.b_first.to_numpy()
    pa = p[["a_end_x", "a_end_y"]].to_numpy(float)
    pb = p[["b_start_x", "b_start_y"]].to_numpy(float)
    acc = np.zeros(len(p))
    cnt = np.zeros(len(p))
    for q in (np.arange(samples) + 0.5) / samples:
        f = np.rint(ta + q * (tb - ta)).astype(int)
        inside = (f > ta) & (f < tb)
        xy = pa + q * (pb - pa)
        m = geom.view_margin(cam["P"][np.clip(f, 0, cam["n"] - 1)], xy)
        acc += inside & (m >= VIEW_MARGIN_PX)
        cnt += inside
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return out


def pair_features(seq: str) -> pd.DataFrame:
    """One row per ordered pair of pieces (a, b) of the clip with a.last < b.first; columns in
    the module docstring. Label-free. Returns a fresh copy (the table is cached)."""
    return _pair_features(seq).copy()


@lru_cache(maxsize=None)
def _pair_features(seq: str) -> pd.DataFrame:
    e = geom.piece_ends(seq)
    cam = geom.cameras(seq)
    S = flow(seq)
    a = e.add_prefix("a_")
    b = e.add_prefix("b_")
    p = a.merge(b, how="cross")
    p = p[p.a_last < p.b_first].reset_index(drop=True)
    p["gap"] = p.b_first - p.a_last - 1
    p["dt"] = (p.b_first - p.a_last) / FPS
    p["dx"] = p.b_start_x - p.a_end_x
    p["dy"] = p.b_start_y - p.a_end_y
    p["d"] = np.hypot(p.dx, p.dy)
    sh = S[p.b_first.to_numpy()] - S[p.a_last.to_numpy()]
    p["sx"], p["sy"] = sh[:, 0], sh[:, 1]
    p["rx"], p["ry"] = p.dx - p.sx, p.dy - p.sy
    p["r"] = np.hypot(p.rx, p.ry)
    # own velocity relative to the flow, kept for a short horizon, from both sides
    Fv = np.gradient(S, axis=0) * FPS
    h = np.minimum(p.dt.to_numpy(), CV_HORIZON_S) / 2.0
    va = p[["a_end_vx", "a_end_vy"]].to_numpy(float) - Fv[p.a_last.to_numpy()]
    vb = p[["b_start_vx", "b_start_vy"]].to_numpy(float) - Fv[p.b_first.to_numpy()]
    cv = np.column_stack([p.rx, p.ry]) - (va + vb) * h[:, None]
    p["r_cv"] = np.hypot(cv[:, 0], cv[:, 1])
    # local advection: end point of a carried forward to b.first, start point of b carried back
    AF, AB = advected(seq)
    ta_, tb_ = p.a_last.to_numpy(), p.b_first.to_numpy()
    ef = np.array([AF[a][t2] for a, t2 in zip(p.a_piece, tb_)]) - p[["b_start_x", "b_start_y"]].to_numpy(float)
    eb = np.array([AB[b][t1] for b, t1 in zip(p.b_piece, ta_)]) - p[["a_end_x", "a_end_y"]].to_numpy(float)
    p["r_fwd"] = np.hypot(ef[:, 0], ef[:, 1])
    p["r_back"] = np.hypot(eb[:, 0], eb[:, 1])
    p["r_adv"] = 0.5 * (p.r_fwd + p.r_back)
    mid = 0.5 * (-ef + eb)            # mean of the two residual vectors, as b - a
    cva = mid - (va + vb) * h[:, None]
    p["r_adv_cv"] = np.hypot(cva[:, 0], cva[:, 1])
    p["speed_req"] = np.maximum(p.d - 2.0, 0.0) / p.dt
    costs = hidden_path_costs(seq)
    iv = np.array([costs.get((int(x), int(y)), np.nan) for x, y in zip(p.a_piece, p.b_piece)])
    p["inview_s"] = iv
    p["reach"] = np.isfinite(iv)
    p["line_inview"] = _line_inview(seq, p)
    fwd, back = carried_inview(seq)
    ta, tb = p.a_last.to_numpy(), p.b_first.to_numpy()
    p["fwd_inview_s"] = [(fwd[a][t2 - 1] - fwd[a][t1]) / FPS if a in fwd else np.nan for a, t1, t2 in zip(p.a_piece, ta, tb)]
    p["back_inview_s"] = [(back[b][t2 - 1] - back[b][t1]) / FPS if b in back else np.nan for b, t1, t2 in zip(p.b_piece, ta, tb)]
    afw, abk = advected_inview(seq)
    p["fwd_adv_inview_s"] = [(afw[a][t2 - 1] - afw[a][t1]) / FPS for a, t1, t2 in zip(p.a_piece, ta, tb)]
    p["back_adv_inview_s"] = [(abk[b][t2 - 1] - abk[b][t1]) / FPS for b, t1, t2 in zip(p.b_piece, ta, tb)]
    pa = p[["a_end_x", "a_end_y"]].to_numpy(float)
    pb = p[["b_start_x", "b_start_y"]].to_numpy(float)
    p["a_end_margin"] = geom.view_margin(cam["P"][p.a_last.to_numpy()], pa)
    p["b_start_margin"] = geom.view_margin(cam["P"][p.b_first.to_numpy()], pb)
    p["a_end_edge"] = p.a_end_side != ""
    p["b_start_edge"] = p.b_start_side != ""
    known = p.a_team.notna() & p.b_team.notna()
    p["same_team"] = np.where(known, (p.a_team == p.b_team).astype(int), -1)
    p["same_role"] = (p.a_role == p.b_role).astype(int)
    p["same_plugin"] = (p.a_plugin == p.b_plugin).astype(int)
    p["n_frames"] = cam["n"]
    cs = census(seq).set_index("piece")
    p["a_census_out"] = p.a_piece.map(cs.census_out).to_numpy()
    p["b_census_in"] = p.b_piece.map(cs.census_in).to_numpy()
    out = p.rename(columns={"a_piece": "a", "b_piece": "b"})
    cols = ["a", "b", "gap", "dt", "d", "dx", "dy", "sx", "sy", "r", "rx", "ry", "r_cv", "r_fwd", "r_back", "r_adv", "r_adv_cv", "speed_req", "reach",
            "inview_s", "line_inview", "fwd_inview_s", "back_inview_s", "fwd_adv_inview_s", "back_adv_inview_s", "a_end_margin", "b_start_margin", "a_end_edge", "b_start_edge",
            "a_end_side", "b_start_side", "same_team", "same_role", "same_plugin", "a_rows", "b_rows",
            "a_last", "b_first", "a_first", "b_last", "n_frames", "a_census_out", "b_census_in", "a_role", "b_role", "a_team", "b_team"]
    return out[cols].copy()


if __name__ == "__main__":
    import time
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    for seq in data.DEV_CLIPS:
        t0 = time.time()
        df = pair_features(seq)
        print(seq, len(df), "pairs", round(time.time() - t0, 1), "s; reach", int(df.reach.sum()),
              "inview_s==0", int((df.inview_s == 0).sum()), "nan", int(df.inview_s.isna().sum()))
    print(df.head(8).to_string())
