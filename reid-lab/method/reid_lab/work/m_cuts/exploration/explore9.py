"""DIAGNOSIS: small-side gap candidates (a side with < 5 rows): labels by cos / gap."""
import sys
sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np, pandas as pd
import loco
pd.set_option("display.width", 280); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 800)
tab = loco.table()
tab.to_pickle("C:/Research/plugins/tools/reid_lab/work/m_cuts/tab_cache.pkl")
print(len(tab), tab.label.value_counts().to_dict())
small = tab[(tab.rows_before < 5) | (tab.rows_after < 5)]
print("small-side candidates", len(small), small.label.value_counts().to_dict())
small = small.assign(cbin=pd.cut(small.cos, [0, 0.6, 0.7, 0.8, 0.85, 0.9, 1.01], right=False), gbin=pd.cut(small.gap, [0, 5, 10, 25, 50, 100, 1000], right=False))
print(small.groupby(["cbin", "label"], observed=False).size().unstack(fill_value=0))
print(small.groupby(["gbin", "label"], observed=False).size().unstack(fill_value=0))
big = tab[(tab.rows_before >= 5) & (tab.rows_after >= 5)]
big = big.assign(gbin=pd.cut(big.gap, [0, 5, 10, 25, 50, 100, 1000], right=False))
print("big sides by gap"); print(big.groupby(["gbin", "label"], observed=False).size().unstack(fill_value=0))
