"""Label-free sanity checks of the geometry the features rest on.

1. Projection: gbf's pitch_x / pitch_y of every placed row, projected with the frame's camera
   (geom.cameras), must land on the row's foot point (foot_u, foot_v).
2. geom.project agrees with gbf.geometry.project.
3. The hidden-path programme on a synthetic view: a path that can go around the picture costs
   0 s, one that must cross it costs about the crossing time, and a target beyond reach is inf.
"""
import sys

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import features  # noqa: E402
import geom  # noqa: E402
from gbf.geometry import Camera, project  # noqa: E402

worst = 0.0
for seq in data.DEV_CLIPS:
    t = data.tracks(seq)
    cam = geom.cameras(seq)
    t = t[t.pitch_usable & t.pitch_x.notna() & t.foot_u.notna()]
    P = cam["P"][t.frame.to_numpy()]
    xyz = np.column_stack([t.pitch_x, t.pitch_y, np.zeros(len(t))])
    uv, depth = geom.project(P, xyz)
    err = np.hypot(uv[:, 0] - t.foot_u.to_numpy(), uv[:, 1] - t.foot_v.to_numpy())
    worst = max(worst, float(np.nanpercentile(err, 99)))
    print(f"{seq}: rows {len(t)}  reprojection error px median {np.nanmedian(err):.3f}  p99 {np.nanpercentile(err, 99):.3f}  max {np.nanmax(err):.2f}  depth>0 {float((depth > 0).mean()):.3f}")
print("1. worst p99 reprojection error px:", round(worst, 3))

seq = data.DEV_CLIPS[0]
c = data.calib(seq).iloc[100]
camera = Camera(K=np.asarray(c.K, float).reshape(3, 3), R=np.asarray(c.R, float).reshape(3, 3), C=np.asarray(c.C, float), width=int(c.width), height=int(c.height))
pts = np.array([[0.0, 0.0, 0.0], [10.0, -5.0, 0.0], [-20.0, 12.0, -1.8]])
uv_gbf, _, _ = project(camera, pts)
uv_me, _ = geom.project(geom.cameras(seq)["P"][int(c.frame_index)], pts)
print("2. max |gbf.project - geom.project| px:", float(np.abs(uv_gbf - uv_me).max()))

# 3. synthetic: picture = the strip |x| < 10 m for all y (so nobody can go around) for the
#    first test, and the box |x| < 10, |y| < 10 for the second (a way around exists).
xs = np.arange(features.GRID_X[0], features.GRID_X[1] + 1e-9, features.GRID_M)
ys = np.arange(features.GRID_Y[0], features.GRID_Y[1] + 1e-9, features.GRID_M)
gx, gy = np.meshgrid(xs, ys)
from scipy import ndimage  # noqa: E402


def cost(V, start, target, steps):
    big = np.float32(1e6)
    move = features._disc(features.MAX_SPEED * (features.STEP / features.FPS) / features.GRID_M)
    slack = features._disc(features.START_RADIUS / features.GRID_M)
    c = np.full(V.shape, big, np.float32)
    c[features._cell(*start)] = 0
    c = ndimage.minimum_filter(c, footprint=slack, mode="constant", cval=big)
    for _ in range(steps - 1):
        c = np.minimum(ndimage.minimum_filter(c, footprint=move, mode="constant", cval=big) + V * np.float32(features.STEP / features.FPS), big)
    reach = ndimage.minimum_filter(c, footprint=move, mode="constant", cval=big)
    v = features._local_min(reach, *features._cell(*target), slack)
    return float(v) if v < big / 2 else float("inf")


strip = (np.abs(gx) < 10).astype(np.float32)
box = ((np.abs(gx) < 10) & (np.abs(gy) < 10)).astype(np.float32)
print("3. strip, 40 m across in 25 steps (10 s): least in-picture s =", cost(strip, (-20, 0), (20, 0), 25), "(about 20 m / 8 m/s = 2.5 s, less the 3 m slack at both ends)")
print("   box, same points, 25 steps: least in-picture s =", cost(box, (-20, 0), (20, 0), 25), "(0: he can run around the box)")
print("   box, same points, 6 steps (2.4 s, reach 6 * 3.2 + 6 m slack = 25 m < 40 m):", cost(box, (-20, 0), (20, 0), 6))
print("   box, 8 steps (reach 31.6 m < 40 m):", cost(box, (-20, 0), (20, 0), 8), " 12 steps straight through (38.4 + 6 m):", cost(box, (-20, 0), (20, 0), 12))
