"""DIAGNOSIS: 2-D look, cos vs alternative margins (diag_cand.csv)."""
import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 280); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 800)
df = pd.read_csv("C:/Research/plugins/tools/reid_lab/work/m_cuts/diag_cand.csv")
df["m_d"] = df.cos - df.alt_d
df["m_r"] = df.cos - df.alt_r
df["mm"] = np.maximum(df.m_d.fillna(9), df.m_r.fillna(9))
df["mn"] = np.minimum(df.m_d.fillna(9), df.m_r.fillna(9))
df["truth"] = np.where(df.sw | (df["loc"] == True), "T", np.where(df["loc"] == False, "F", "U"))
cb = [0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
mb = [-1, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 10]
df["cbin"] = pd.cut(df.cos, cb, right=False)
df["mbin"] = pd.cut(df.mm, mb, right=False)
df["nbin"] = pd.cut(df.mn, mb, right=False)
for col in ("mbin", "nbin"):
    print(col, "(rows cos, cols margin): T / F counts")
    t = df[df.truth == "T"].groupby(["cbin", col], observed=False).size().unstack(fill_value=0)
    f = df[df.truth == "F"].groupby(["cbin", col], observed=False).size().unstack(fill_value=0)
    print((t.astype(str) + "/" + f.astype(str)).to_string())
print("alt missing:", df.alt_d.isna().sum(), df.alt_r.isna().sum())
# by min side rows
df["side"] = np.minimum(df.nb, df.na)
df["sbin"] = pd.cut(df.side, [0, 5, 10, 20, 21], right=False)
t = df[df.truth == "T"].groupby(["cbin", "sbin"], observed=False).size().unstack(fill_value=0)
f = df[df.truth == "F"].groupby(["cbin", "sbin"], observed=False).size().unstack(fill_value=0)
print("rows cos, cols min(nb, na): T/F")
print((t.astype(str) + "/" + f.astype(str)).to_string())
df["gbin"] = pd.cut(df.gap, [0, 2, 5, 15, 50, 1000], right=False)
t = df[df.truth == "T"].groupby(["cbin", "gbin"], observed=False).size().unstack(fill_value=0)
f = df[df.truth == "F"].groupby(["cbin", "gbin"], observed=False).size().unstack(fill_value=0)
print("rows cos, cols gap: T/F")
print((t.astype(str) + "/" + f.astype(str)).to_string())
df["speed"] = df.d_m / df.dt
df["vbin"] = pd.cut(df.d_m.fillna(-1), [-2, 0, 1, 2, 4, 8, 1000], right=False)
t = df[df.truth == "T"].groupby(["cbin", "vbin"], observed=False).size().unstack(fill_value=0)
f = df[df.truth == "F"].groupby(["cbin", "vbin"], observed=False).size().unstack(fill_value=0)
print("rows cos, cols d_m: T/F")
print((t.astype(str) + "/" + f.astype(str)).to_string())
