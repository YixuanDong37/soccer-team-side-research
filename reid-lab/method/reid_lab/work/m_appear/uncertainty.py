"""Cluster bootstrap over the 13 clips for a METHOD.json: how uncertain are the pooled shares?
python -I uncertainty.py out/agglo-lam3.method.json ..."""
import sys, os, json; sys.path.insert(0, "C:/Research/plugins/tools")
import numpy as np
from reid_lab import data, score
rng = np.random.default_rng(0)
split = {s: score.score_clip(s, [])["returns"]["wrong"] for s in data.DEV_CLIPS}
for path in sys.argv[1:]:
    m = json.load(open(path, encoding="utf-8"))["clips"]
    per = {s: score.score_clip(s, (m.get(s) or {}).get("links", [])) for s in data.DEV_CLIPS}
    R = np.array([[per[s]["returns"]["right"], per[s]["returns"]["wrong"] - split[s], per[s]["returns"]["right"] + per[s]["returns"]["wrong"] + per[s]["returns"]["lost"],
                   per[s]["links"]["right"], per[s]["links"]["wrong"]] for s in data.DEV_CLIPS], float)
    idx = rng.integers(0, len(R), size=(5000, len(R)))
    B = R[idx].sum(axis=1)
    q = lambda v: np.percentile(v, [5, 95]).round(3).tolist()
    tot = R.sum(axis=0)
    print(os.path.basename(path), "per clip right:", R[:, 0].astype(int).tolist(), "extra wrong:", R[:, 1].astype(int).tolist(), "scored:", R[:, 2].astype(int).tolist())
    print(f"   right share {tot[0] / tot[2]:.3f} (90 % interval {q(B[:, 0] / B[:, 2])}); extra wrong share {tot[1] / tot[2]:.3f} ({q(B[:, 1] / B[:, 2])}); "
          f"link precision {tot[3] / (tot[3] + tot[4]):.3f} ({q(B[:, 3] / (B[:, 3] + B[:, 4]))})")
