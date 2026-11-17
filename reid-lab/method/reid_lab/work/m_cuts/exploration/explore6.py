"""DIAGNOSIS: univariate look at the candidate features against truth (diag_cand.csv)."""
import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 280); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 800)
df = pd.read_csv("C:/Research/plugins/tools/reid_lab/work/m_cuts/diag_cand.csv")
df["speed"] = df.d_m / df.dt
df["m_d"] = df.cos - df.alt_d
df["m_r"] = df.cos - df.alt_r
df["take"] = df.d_cos - df.cos
print(df.sw.sum(), len(df), df.cos.isna().sum(), df[df.cos.isna()].sw.sum())
print("loc among sw:", df[df.sw]["loc"].value_counts(dropna=False).to_dict(), " loc among ~sw:", df[~df.sw]["loc"].value_counts(dropna=False).to_dict())


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.r_[pos, neg]
    r = pd.Series(allv).rank().to_numpy()
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


for name, sub in (("all", df), ("gap 2-4", df[(df.gap >= 2) & (df.gap < 5)]), ("gap 5-14", df[(df.gap >= 5) & (df.gap < 15)]), ("gap 15-49", df[(df.gap >= 15) & (df.gap < 50)]), ("gap>=50", df[df.gap >= 50])):
    print(f"--- {name}: n={len(sub)} sw={int(sub.sw.sum())}")
    for col in ("cos", "cos_whole", "gap", "d_m", "speed", "jump_h", "iou_self", "d_iou", "r_iou", "alt_d", "alt_r", "m_d", "m_r", "take", "box_h", "rows_before", "rows_after", "nb", "na"):
        pos, neg = sub[sub.sw][col].to_numpy(float), sub[~sub.sw][col].to_numpy(float)
        q = lambda a: np.round(np.nanquantile(a, [0.1, 0.5, 0.9]), 2) if np.isfinite(a).any() else None
        print(f"   {col:12s} AUC {auc(pos, neg):.3f}  pos q10/50/90 {q(pos)}  neg {q(neg)}  nan pos {np.isnan(pos).mean():.2f} neg {np.isnan(neg).mean():.2f}")
# the low-cos negatives: what are they
neg = df[(~df.sw) & (df.cos < 0.75)].sort_values("cos")
print(len(neg))
print(neg[["seq", "piece", "f0", "f1", "gap", "kind", "cos", "cos_whole", "nb", "na", "clean_b", "clean_a", "d_m", "dt", "d_iou", "alt_d", "alt_r", "loc", "role", "team", "box_h", "rows_before", "rows_after"]].head(70).to_string())
