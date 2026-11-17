"""Label-free camera-view helpers for the diag_cues diagnosis (reads calib.parquet only).

P_t = K R [I | -C] per frame (gbf.geometry conventions: SoccerNet world, metres, z down; image
u right, v down). Frames without a calibration have NaN matrices. check() compares the
projection with gbf.geometry.project.
"""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data  # noqa: E402

W, H = data.WIDTH, data.HEIGHT
PERSON_M = 1.8


@lru_cache(maxsize=None)
def matrices(seq: str):
    """(P, C, f): P (n_frames, 3, 4), C (n_frames, 3), f focal px (n_frames,); NaN where no camera."""
    cal = data.calib(seq).sort_values("frame_index")
    n = int(cal.frame_index.max()) + 1
    P = np.full((n, 3, 4), np.nan)
    Cc = np.full((n, 3), np.nan)
    foc = np.full(n, np.nan)
    for r in cal.itertuples():
        if r.K is None or r.R is None or r.C is None:
            continue
        K = np.asarray(r.K, float).reshape(3, 3)
        R = np.asarray(r.R, float).reshape(3, 3)
        c = np.asarray(r.C, float)
        if not (np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(c).all()):
            continue
        P[r.frame_index] = K @ R @ np.hstack([np.eye(3), -c[:, None]])
        Cc[r.frame_index] = c
        foc[r.frame_index] = K[0, 0]
    return P, Cc, foc


def project(seq: str, frames, xyz):
    """Project world points xyz (..., 3) at frames (broadcastable leading shape) -> u, v, ok (in front)."""
    P, _, _ = matrices(seq)
    frames = np.asarray(frames, dtype=np.int64)
    xyz = np.asarray(xyz, float)
    shape = np.broadcast_shapes(frames.shape, xyz.shape[:-1])
    fr = np.broadcast_to(frames, shape)
    pt = np.broadcast_to(xyz, (*shape, 3))
    inside = (fr >= 0) & (fr < len(P))
    Pm = P[np.clip(fr, 0, len(P) - 1)]
    h = np.einsum("...ij,...j->...i", Pm[..., :3], pt) + Pm[..., 3]
    ok = inside & np.isfinite(h).all(axis=-1) & (h[..., 2] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(ok, h[..., 0] / h[..., 2], np.nan)
        v = np.where(ok, h[..., 1] / h[..., 2], np.nan)
    return u, v, ok


def person_visible(seq: str, frames, x, y, margin_px: float = 0.0, height_m: float = PERSON_M):
    """Whether a person standing at ground point (x, y) is inside the picture at each frame:
    feet and head (height_m up, z = -height_m) both inside the image less margin_px on the left
    and right, feet above the bottom edge and head below the top edge. NaN where no camera."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = np.zeros_like(x)
    uf, vf, okf = project(seq, frames, np.stack([x, y, z], axis=-1))
    uh, vh, okh = project(seq, frames, np.stack([x, y, z - height_m], axis=-1))
    vis = okf & okh & (uf >= margin_px) & (uf <= W - 1 - margin_px) & (uh >= margin_px) & (uh <= W - 1 - margin_px) & (vf <= H - 1) & (vh >= 0)
    out = vis.astype(float)
    out[~(okf | okh) & ~np.isfinite(uf)] = np.nan  # no camera (or behind it)
    P, _, _ = matrices(seq)
    fr = np.broadcast_to(np.asarray(frames, dtype=np.int64), out.shape)
    nocam = ~np.isfinite(P[np.clip(fr, 0, len(P) - 1), 0, 0])
    out[nocam] = np.nan
    return out, uf


_GX, _GY = np.meshgrid(np.arange(-52.0, 52.5, 1.0), np.arange(-33.5, 34.0, 1.0))  # 105 x 68 cell centres
_GRID = np.stack([_GX.ravel(), _GY.ravel(), np.zeros(_GX.size)], axis=1)


@lru_cache(maxsize=None)
def view_table(seq: str):
    """Per frame: visible pitch area (m^2: 1 m cells whose centre projects into the image), the
    visible x-range on the near touchline (y=34), the halfway axis (y=0) and the far touchline
    (y=-34), the ground point of the image centre, focal length. NaN where no camera."""
    import pandas as pd

    P, Cc, foc = matrices(seq)
    n = len(P)
    rows = []
    xs = np.arange(-70.0, 70.01, 0.25)
    for f in range(n):
        if not np.isfinite(P[f, 0, 0]):
            rows.append({"frame": f})
            continue
        u, v, ok = project(seq, np.full(len(_GRID), f), _GRID)
        vis = ok & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        row = {"frame": f, "area_m2": float(vis.sum()), "focal": float(foc[f])}
        for name, yy in (("near", 34.0), ("mid", 0.0), ("far", -34.0)):
            pts = np.stack([xs, np.full_like(xs, yy), np.zeros_like(xs)], axis=1)
            uu, vv, oo = project(seq, np.full(len(xs), f), pts)
            inside = oo & (uu >= 0) & (uu <= W - 1) & (vv >= 0) & (vv <= H - 1)
            row[f"xl_{name}"] = float(xs[inside].min()) if inside.any() else np.nan
            row[f"xr_{name}"] = float(xs[inside].max()) if inside.any() else np.nan
        # ground point of the image centre (ray of the principal pixel)
        M = P[f, :, :3]
        d = np.linalg.solve(M, np.array([(W - 1) / 2, (H - 1) / 2, 1.0]))
        c = Cc[f]
        if abs(d[2]) > 1e-12:
            t = -c[2] / d[2]  # M (c + t d - c) = t (u, v, 1): t is the depth, > 0 in front
            if t > 0:
                g = c + t * d
                row["cx"], row["cy"] = float(g[0]), float(g[1])
        rows.append(row)
    return pd.DataFrame(rows)


def lateral_bounds(seq: str, frames, y):
    """Visible x-range [xl, xr] at ground line y for each frame (0.25 m steps over +-70 m)."""
    xs = np.arange(-70.0, 70.01, 0.25)
    frames = np.atleast_1d(np.asarray(frames, dtype=np.int64))
    y = np.broadcast_to(np.asarray(y, float), frames.shape)
    xl = np.full(len(frames), np.nan)
    xr = np.full(len(frames), np.nan)
    for i, (f, yy) in enumerate(zip(frames, y)):
        pts = np.stack([xs, np.full_like(xs, yy), np.zeros_like(xs)], axis=1)
        uu, vv, oo = project(seq, np.full(len(xs), f), pts)
        inside = oo & (uu >= 0) & (uu <= W - 1)
        if inside.any():
            xl[i], xr[i] = xs[inside].min(), xs[inside].max()
    return xl, xr


def check(seq: str = "SNGS-107") -> float:
    """Largest difference (px) between project() here and gbf.geometry.project on a few frames."""
    from gbf.geometry import project as gproject
    from gbf.sngs_import import cameras_by_frame

    cams = cameras_by_frame(data.calib(seq), usable_only=False)
    worst = 0.0
    pts = np.array([[0.0, 0.0, 0.0], [20.0, -10.0, 0.0], [-30.0, 25.0, -1.8], [52.5, 34.0, 0.0]])
    for f in (0, 100, 400, 749):
        if f not in cams:
            continue
        uv, depth, valid = gproject(cams[f], pts)
        u, v, ok = project(seq, np.full(len(pts), f), pts)
        both = valid & ok
        worst = max(worst, float(np.nanmax(np.abs(np.stack([u, v], axis=1)[both] - uv[both]))) if both.any() else 0.0)
        assert (valid == ok).all()
    return worst
