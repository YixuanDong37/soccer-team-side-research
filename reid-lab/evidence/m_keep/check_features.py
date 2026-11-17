"""Label-free self-check: the two implementations of the pair features agree on the re-entry pairs."""
import sys, time
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
import numpy as np
from reid_lab import data
import features
bad = 0
for s in data.DEV_CLIPS:
    t0 = time.time()
    r = features.pair_features(s)
    a = features.pair_features(s, "all")
    m = r.merge(a[a.plugin_join == 1], on=["a", "b"], suffixes=("", "__v"))
    assert len(m) == len(r), (s, len(m), len(r))
    for c in ("cos", "margin_a", "margin_b", "rank_a", "rank_b", "dist_m", "excess_m", "imp_pct"):
        x, y = m[c].to_numpy(float), m[c + "__v"].to_numpy(float)
        ok = np.isclose(x, y, equal_nan=True, atol=1e-9)
        if not ok.all():
            bad += 1
            print(s, c, "differs", m.loc[~ok, ["a", "b", c, c + "__v"]].to_string())
    print(s, "re-entry pairs", len(r), "all forward pairs", len(a), "plugin joins among them", int(a.plugin_join.sum()), f"{time.time() - t0:.1f}s")
print("problems:", bad)
