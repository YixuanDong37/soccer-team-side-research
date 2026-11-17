import sys; sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
seq = "SNGS-107"
t = data.tracks(seq)
print(t.dtypes)
print(t.head(5).T)
print(len(t), t.track_id.nunique(), (t.track_id<0).sum())
e = data.embeddings(seq)
print(e.shape, np.isnan(e[:,0]).mean())
print(data.clean_rows(seq).mean())
c = data.calib(seq)
print(c.dtypes); print(c.head(3).T)
r = data.refine(seq)
print(list(r.keys()))
print({k: (type(v).__name__, (list(v.keys())[:20] if isinstance(v, dict) else (v[:3] if isinstance(v, list) else v))) for k, v in r.items()})
a = data.appearance(seq)
print({k: (type(v).__name__, (list(v.keys())[:20] if isinstance(v, dict) else (v[:3] if isinstance(v, list) else v))) for k, v in a.items()})
print(data.pieces(seq).head(10))
for s in data.DEV_CLIPS:
    tt = data.tracks(s); print(s, len(tt), tt.track_id.nunique(), tt.frame.min(), tt.frame.max(), data.shot_cuts(s), tt.role.value_counts().to_dict())
