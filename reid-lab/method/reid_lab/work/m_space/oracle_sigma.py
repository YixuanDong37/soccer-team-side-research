"""ORACLE / DIAGNOSIS ONLY. What does the residual of true partners depend on besides dt?"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import fitdata  # noqa: E402
import linker  # noqa: E402

pd.set_option("display.width", 250)
p = linker.admissible(fitdata.pooled(data.DEV_CLIPS))
T = p[p.imm].copy()
T["shift"] = np.hypot(T.sx, T.sy)
T["kind"] = T.a_end_edge.astype(int) + T.b_start_edge.astype(int)
for res in ("r", "r_cv"):
    s = linker.fit_sigma(T, res)
    T["u"] = T[res] / linker.sigma(T.dt.to_numpy(), s)
    print(res, "sigma", np.round(s, 3))
    print("  u quantiles by kind:")
    print(T.groupby("kind").u.describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())
    T["shiftbin"] = pd.cut(T["shift"], [-1, 1, 3, 6, 12, 100])
    print("  u by flow shift (m):")
    print(T.groupby("shiftbin", observed=True).u.describe(percentiles=[.5, .75, .9]).round(2).to_string())
    T["dtbin"] = pd.cut(T.dt, [0, 0.4, 1, 3, 6, 12, 40])
    print("  u by dt:")
    print(T.groupby("dtbin", observed=True).u.describe(percentiles=[.5, .75, .9]).round(2).to_string())
    T["rowbin"] = pd.cut(np.minimum(T.a_rows, T.b_rows), [0, 5, 15, 40, 1000])
    print("  u by smaller piece length (rows):")
    print(T.groupby("rowbin", observed=True).u.describe(percentiles=[.5, .75, .9]).round(2).to_string())
