import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
seq = "SNGS-107"
t = data.tracks(seq)
print(t.dtypes)
print(t.head(5))
print(len(t), t.frame.min(), t.frame.max())
c = data.calib(seq)
print(c.dtypes); print(c.head(3))
print(c.calib_status.value_counts(), c.pitch_usable.mean())
r = data.refine(seq)
print(r.keys())
sa = r.get("stitch_audit") or r.get("stitch")
print({k: (v if not isinstance(v, list) else len(v)) for k, v in sa.items() if k not in ("params",)})
p = data.pieces(seq)
print(p.head(20)); print(len(p))
a = data.appearance(seq)
print(a.keys())
