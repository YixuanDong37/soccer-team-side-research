"""Per-clip view of the held-out result and clip-bootstrap intervals (reads results.json and the
methods/*_cuts.csv written by make_methods.py; scores the baselines per clip).

    python -I report.py [p0]
"""
import json
import sys

sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np
import pandas as pd
from reid_lab import data, score

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
p0 = sys.argv[1] if len(sys.argv) > 1 else "0.7"
res = json.load(open(HERE + "results.json"))["points"][p0]["heldout"]
cuts = pd.read_csv(HERE + f"methods/heldout_p{p0}_cuts.csv")
base = {k: score.baseline_links(k) for k in ("split", "mutual")}
rows = []
for seq in data.DEV_CLIPS:
    c = cuts[cuts.seq == seq]
    s0 = score.score_clip(seq, [])
    m0 = score.score_clip(seq, base["mutual"][seq]["links"])
    s1, m1, m2 = res["split_per_clip"][seq], res["mutual_per_clip"][seq], res["mutual_rerun_per_clip"][seq]
    n = s0["tracked_rows"]
    rows.append({"clip": seq, "cuts": len(c), **{k: int((c.label == k).sum()) for k in ("T", "F", "t", "f", "U")},
                 "rows": n, "imp_split": s0["impure_rows"], "imp_split+cuts": s1["impure_rows"],
                 "wrong_split": s0["returns"]["wrong"], "wrong_split+cuts": s1["wrong"],
                 "mutual R/W/L": "{right}/{wrong}/{lost}".format(**m0["returns"]), "imp_mutual": m0["impure_rows"],
                 "mutual+cuts R/W/L": "{right}/{wrong}/{lost}".format(**m1), "imp_mutual+cuts": m1["impure_rows"],
                 "rerun+cuts R/W/L": "{right}/{wrong}/{lost}".format(**m2), "imp_rerun+cuts": m2["impure_rows"]})  # fmt: skip
df = pd.DataFrame(rows)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60)
print(df.to_string(index=False))
tot = df[["cuts", "T", "F", "t", "f", "U", "rows", "imp_split", "imp_split+cuts", "wrong_split", "wrong_split+cuts", "imp_mutual", "imp_mutual+cuts", "imp_rerun+cuts"]].sum()
print("total", tot.to_dict())

# clip bootstrap: resample the 13 clips
rng = np.random.default_rng(0)
B = 10000
idx = rng.integers(0, len(df), size=(B, len(df)))
T_, F_, R, I0, I1 = (df[c].to_numpy(float) for c in ("T", "F", "rows", "imp_split", "imp_split+cuts"))
prec = T_[idx].sum(1) / np.maximum(1, T_[idx].sum(1) + F_[idx].sum(1))
i0 = I0[idx].sum(1) / R[idx].sum(1)
i1 = I1[idx].sum(1) / R[idx].sum(1)
q = lambda a: np.round(np.quantile(a, [0.025, 0.975]), 4).tolist()  # noqa: E731
print(f"cut precision T/(T+F) = {T_.sum() / (T_.sum() + F_.sum()):.3f}, clip-bootstrap 95% {q(prec)}")
print(f"impure rows split {I0.sum() / R.sum():.4f} {q(i0)} -> split+cuts {I1.sum() / R.sum():.4f} {q(i1)}; ratio {I1.sum() / I0.sum():.3f} {q(i1 / i0)}")
df.to_csv(HERE + f"per_clip_p{p0}.csv", index=False)
