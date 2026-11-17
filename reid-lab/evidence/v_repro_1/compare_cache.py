"""v_repro_1 / compare_cache.py -- are the label-free tables rebuilt here from nothing equal to fuse_r1's cache?
(fuse_r1 copied its app_/space_/ends_ tables from work/fuse instead of building them.)  Read-only on fuse_r1."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
A = Path("C:/Research/plugins/tools/reid_lab/work/v_repro_1/cache")
B = Path("C:/Research/plugins/tools/reid_lab/work/fuse_r1/cache")
bad = []
for name in ("cut_index.json",):
    a, b = json.loads((A / name).read_text()), json.loads((B / name).read_text())
    print(name, "equal" if a == b else "DIFFERENT")
    if a != b: bad.append(name)
a, b = json.loads((A / "cuts.json").read_text()), json.loads((B / "cuts.json").read_text())
common = set(a) & set(b)
diff = [k for k in common if a[k] != b[k]]
print("cuts.json: keys here", len(a), "there", len(b), "common", len(common), "different", len(diff), "missing there", len(set(a) - set(b)))
if diff or set(a) - set(b): bad.append("cuts.json")
n = same = 0
maxdiff = 0.0
for f in sorted(A.glob("*.pkl")):
    g = B / f.name
    n += 1
    if not g.exists():
        bad.append(("missing in fuse_r1", f.name)); continue
    x, y = pd.read_pickle(f), pd.read_pickle(g)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        bad.append(("shape", f.name, x.shape, y.shape)); continue
    ok = True
    for c in x.columns:
        u, v = x[c].to_numpy(), y[c].to_numpy()
        if u.dtype.kind in "fc" or v.dtype.kind in "fc":
            u = u.astype(float); v = v.astype(float)
            eq = (u == v) | (np.isnan(u) & np.isnan(v))
            if not eq.all():
                d = np.nanmax(np.abs(u - v)[~eq]) if (~eq).any() else 0.0
                maxdiff = max(maxdiff, float(d)); ok = False
                bad.append(("values", f.name, c, int((~eq).sum()), float(d)))
        else:
            try:
                e = (pd.Series(u).fillna("__nan__").to_numpy() == pd.Series(v).fillna("__nan__").to_numpy()).all()
            except Exception:
                e = all((p == q) or (p != p and q != q) for p, q in zip(u, v))
            if not e:
                ok = False; bad.append(("values", f.name, c))
    same += ok
print("pkl tables here:", n, "exactly equal to fuse_r1's:", same, "max abs diff of unequal floats:", maxdiff)
print("problems:", bad[:20] if bad else "none", "" if len(bad) <= 20 else f"... ({len(bad)})")
