"""(5d) Does one cosine bar transfer between clips? Leave-one-clip-out check on piece pairs
(diagnosis of the cue; pair level, not the project's return score).

Pool: time-disjoint piece pairs, same person against different persons of one annotated team.
For each held-out clip the bar is the smallest value that reaches the target precision on the
other 12 clips' pool; it is applied unchanged to the held-out clip. One parameter (the bar).
Needs out/t5c_piece_pairs.parquet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import common as C  # noqa: E402

pd.set_option("display.width", 250)
PP = pd.read_parquet(C.OUT / "t5c_piece_pairs.parquet")
PP = PP[PP.kind.isin(["same", "hard"])]


def bar_for(g, col, target):
    """Smallest bar whose pooled precision on g is >= target (None when never reached)."""
    s = g.sort_values(col, ascending=False)
    tp = (s.kind == "same").cumsum().to_numpy()
    n = np.arange(1, len(s) + 1)
    ok = np.where(tp / n >= target)[0]
    return float(s[col].iat[ok[-1]]) if len(ok) else None


tabs = []
for col in ("mean_all", "median_all", "mean_all_teamcentred", "median_all_teamcentred", "mean_clean_snorm", "mean_all_teamcentred_snorm"):
    for target in (0.90, 0.95, 0.98):
        tp = fp = n_same = 0
        bars, worst = [], 1.0
        for fit, held in C.data.loco_folds():
            b = bar_for(PP[PP.seq.isin(fit)].dropna(subset=[col]), col, target)
            h = PP[PP.seq == held]
            n_same += int((h.kind == "same").sum())
            if b is None:
                continue
            bars.append(b)
            t_, f_ = int(((h[col] >= b) & (h.kind == "same")).sum()), int(((h[col] >= b) & (h.kind == "hard")).sum())
            tp += t_
            fp += f_
            if t_ + f_:
                worst = min(worst, t_ / (t_ + f_))
        tabs.append({"score": col, "target precision (fit clips)": target, "bar min": min(bars), "bar max": max(bars), "held-out same kept": tp, "recall": tp / n_same,
                     "held-out hard kept": fp, "held-out precision": tp / max(tp + fp, 1), "worst held-out clip": worst})  # fmt: skip
T = pd.DataFrame(tabs)
print(T.round(3).to_string())
T.to_csv(C.OUT / "t5d_bar_transfer.csv", index=False)
