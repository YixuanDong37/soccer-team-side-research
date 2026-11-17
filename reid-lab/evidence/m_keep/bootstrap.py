"""Clip bootstrap of the held-out results (13 clips resampled with replacement, 5000 times).

    python -I bootstrap.py           reads out/*_loco_lam*.score.json (written by loco.py)

Reports, per method file: pooled right / wrong of the scored returns and 90% percentile
intervals of the right share and wrong share; and the paired difference to the baselines
(same resampled clips).  With 13 clips the intervals are wide; they are honest about that.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
rng = np.random.default_rng(0)
idx = rng.integers(0, len(data.DEV_CLIPS), size=(5000, len(data.DEV_CLIPS)))


def table(per_clip):
    return np.array([[per_clip[s]["returns"][k] for k in ("right", "wrong", "lost")] for s in data.DEV_CLIPS], float)


base = {b: table(score.evaluate(score.baseline_links(b))["per_clip"]) for b in ("split", "plugin", "mutual")}


def shares(t):
    tot = t[idx].sum(axis=1)                      # (B, 3)
    n = tot.sum(axis=1)
    return tot[:, 0] / n, tot[:, 1] / n


def ci(x):
    lo, hi = np.percentile(x, [5, 95])
    return f"[{lo:.3f}, {hi:.3f}]"


rows = {**base}
for f in sorted(OUT.glob("*_loco_lam*.score.json")):
    rows[f.name.replace(".score.json", "")] = table(json.loads(f.read_text(encoding="utf-8"))["per_clip"])
ref_r, ref_w = shares(base["mutual"])
spl_r, spl_w = shares(base["split"])
print(f"{'method':34s} right wrong lost | right share 90% CI      | wrong share 90% CI      | right - mutual 90% CI     | wrong - split 90% CI")
for name, t in rows.items():
    r, w = shares(t)
    tot = t.sum(axis=0).astype(int)
    n = tot.sum()
    print(f"{name:34s} {tot[0]:5d} {tot[1]:5d} {tot[2]:4d} | {tot[0] / n:.3f} {ci(r):16s} | {tot[1] / n:.3f} {ci(w):16s} | {ci(r - ref_r):24s} | {ci(w - spl_w)}")
