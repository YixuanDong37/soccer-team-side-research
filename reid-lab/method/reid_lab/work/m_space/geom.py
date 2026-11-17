"""Label-free geometry helpers for the position / motion / camera-view evidence (m_space).

Everything here reads only reid_lab.data (tracking outputs + calibration). No labels.

World frame = SoccerNet (gbf.geometry): metres, origin centre spot, x along the touchlines
(+-52.5), y across (+-34, camera side y = +34), z DOWN (a head is at z = -1.8).
Camera: P = K R [I | -C]; pixels 1280 x 720.
"""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data  # noqa: E402

FPS = float(data.FPS)
W, H = float(data.WIDTH), float(data.HEIGHT)
HALF_L, HALF_W = 52.5, 34.0
EDGE_PX = 10.0            # a box this close to the image border "touches" it (data.EDGE_PX)
END_WINDOW = 25           # frames at a piece's end / start used for its position and velocity
MAX_SPEED_FIT = 10.0      # m/s cap on a fitted end / start velocity


@lru_cache(maxsize=None)
def cameras(seq: str) -> dict:
    """Per frame: P (n, 3, 4) projection matrices (NaN where no camera), ok (has a camera),
    usable (gbf pitch_usable), C (n, 3)."""
    c = data.calib(seq).sort_values("frame_index")
    n = int(max(c.frame_index.max(), data.tracks(seq).frame.max())) + 1
    P = np.full((n, 3, 4), np.nan)
    C = np.full((n, 3), np.nan)
    ok = np.zeros(n, bool)
    usable = np.zeros(n, bool)
    for r in c.itertuples():
        if r.K is None or r.R is None or r.C is None:
            continue
        K = np.asarray(r.K, float).reshape(3, 3)
        R = np.asarray(r.R, float).reshape(3, 3)
        Cc = np.asarray(r.C, float)
        if not (np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(Cc).all()):
            continue
        f = int(r.frame_index)
        P[f] = K @ R @ np.hstack([np.eye(3), -Cc[:, None]])
        C[f] = Cc
        ok[f] = True
        usable[f] = bool(r.pitch_usable)
    return {"P": P, "ok": ok, "usable": usable, "C": C, "n": n}


def project(P: np.ndarray, xyz: np.ndarray):
    """P (..., 3, 4), xyz (..., 3) -> uv (..., 2), depth (...,)."""
    h = np.concatenate([xyz, np.ones(xyz.shape[:-1] + (1,))], axis=-1)
    q = np.einsum("...ij,...j->...i", P, h)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[..., :2] / q[..., 2:3]
    return uv, q[..., 2]


def view_margin(P: np.ndarray, xy: np.ndarray, body: float = 1.8) -> np.ndarray:
    """Signed distance in px of a person standing on ground point(s) xy (..., 2) from the
    image border of camera(s) P (..., 3, 4): > 0 inside (the smaller of the feet's and the
    head's distance to the nearest border), < 0 outside; -inf behind the camera or without
    a camera."""
    z0 = np.zeros(xy.shape[:-1] + (1,))
    feet, d1 = project(P, np.concatenate([xy, z0], axis=-1))
    head, d2 = project(P, np.concatenate([xy, z0 - body], axis=-1))
    out = np.full(np.broadcast_shapes(xy.shape[:-1], P.shape[:-2]), np.inf)
    for uv in (feet, head):
        m = np.minimum(np.minimum(uv[..., 0], W - 1 - uv[..., 0]), np.minimum(uv[..., 1], H - 1 - uv[..., 1]))
        out = np.minimum(out, m)
    good = (d1 > 0) & (d2 > 0) & np.isfinite(out)
    return np.where(good, out, -np.inf)


def _side(x1, y1, x2, y2) -> str:
    """Which image border(s) a box touches: letters of L, R, T, B (empty = none)."""
    s = ""
    if x1 <= EDGE_PX:
        s += "L"
    if x2 >= W - 1 - EDGE_PX:
        s += "R"
    if y1 <= EDGE_PX:
        s += "T"
    if y2 >= H - 1 - EDGE_PX:
        s += "B"
    return s


def _fit(frames: np.ndarray, xy: np.ndarray, at: float):
    """Robust position at frame ``at`` and velocity (m/s) from points (frames, xy)."""
    n = len(frames)
    if n == 0:
        return (np.nan, np.nan), (0.0, 0.0), 0
    if n < 6 or frames.max() - frames.min() < 6:
        k = min(n, 5)
        order = np.argsort(np.abs(frames - at))[:k]
        med = np.median(xy[order], axis=0)
        return (float(med[0]), float(med[1])), (0.0, 0.0), n
    t = (frames - at) / FPS
    A = np.column_stack([np.ones(n), t])
    keep = np.ones(n, bool)
    coef = None
    for _ in range(2):  # least squares, then once more without the outliers
        coef, *_ = np.linalg.lstsq(A[keep], xy[keep], rcond=None)
        res = np.hypot(*(xy - A @ coef).T)
        thr = max(1.0, 2.5 * np.median(res[keep]) * 1.4826)
        new = res <= thr
        if new.sum() < 6:
            break
        keep = new
    pos, vel = coef[0], coef[1]
    sp = float(np.hypot(*vel))
    if sp > MAX_SPEED_FIT:
        vel = vel * MAX_SPEED_FIT / sp
    return (float(pos[0]), float(pos[1])), (float(vel[0]), float(vel[1])), int(keep.sum())


@lru_cache(maxsize=None)
def piece_ends(seq: str) -> pd.DataFrame:
    """One row per piece (track_id >= 0) with its start and end state on the pitch.

    first / last: frames. end_x, end_y (start_x, start_y): ground position at the last
    (first) frame from a robust line fit over the last (first) END_WINDOW frames of placed
    rows (pitch_usable, finite; rows whose box is cut by the bottom image border are left
    out when others exist: their feet are not seen). end_vx, end_vy: fitted velocity in m/s
    (0 when too few rows). end_side / start_side: image borders the last / first box
    touches. end_age / start_age: frames between the endpoint and the nearest row used.
    """
    t = data.tracks(seq)
    t = t[t.track_id >= 0].sort_values(["track_id", "frame"])
    rows = []
    for pid, g in t.groupby("track_id", sort=True):
        f = g.frame.to_numpy(np.int64)
        xy = g[["pitch_x", "pitch_y"]].to_numpy(float)
        box = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        placed = g.pitch_usable.to_numpy(bool) & np.isfinite(xy).all(axis=1)
        bottom = box[:, 3] >= H - 1 - EDGE_PX
        first, last = int(f[0]), int(f[-1])
        has_team = bool(g.team.notna().any())
        has_role = bool(g.role.notna().any())
        rec = {
            "piece": int(pid), "plugin": int(g.source_track_id.iloc[0]), "k": int(pid) // 100000,
            "first": first, "last": last, "rows": len(g),
            "team": g.team.mode().iloc[0] if has_team else None,
            "team_share": float((g.team == g.team.mode().iloc[0]).mean()) if has_team else 0.0,
            "role": g.role.mode().iloc[0] if has_role else None,
            "role_share": float((g.role == g.role.mode().iloc[0]).mean()) if has_role else 0.0,
            "end_side": _side(*box[-1]), "start_side": _side(*box[0]),
            "med_h": float(np.median(box[:, 3] - box[:, 1])),
            "mean_conf": float(g.conf.mean()),
        }
        for name, at, sel in (("end", last, f >= last - END_WINDOW + 1), ("start", first, f <= first + END_WINDOW - 1)):
            use = sel & placed & ~bottom
            if use.sum() < 2:
                use = sel & placed
            if use.sum() == 0 and placed.any():  # nothing placed in the window: nearest placed rows
                idx = np.flatnonzero(placed)
                near = idx[np.argsort(np.abs(f[idx] - at))[:5]]
                use = np.zeros(len(f), bool)
                use[near] = True
            pos, vel, n = _fit(f[use], xy[use], float(at))
            age = int(np.min(np.abs(f[use] - at))) if use.any() else -1
            rec.update({f"{name}_x": pos[0], f"{name}_y": pos[1], f"{name}_vx": vel[0], f"{name}_vy": vel[1],
                        f"{name}_n": n, f"{name}_age": age})
        rec.update({"end_u": float((box[-1, 0] + box[-1, 2]) / 2), "end_v": float(box[-1, 3]),
                    "start_u": float((box[0, 0] + box[0, 2]) / 2), "start_v": float(box[0, 3])})
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    for s in data.DEV_CLIPS[:2]:
        e = piece_ends(s)
        print(s, len(e))
        print(e.head(12).to_string())
        print(e[["end_n", "start_n", "end_age", "start_age"]].describe().to_string())
