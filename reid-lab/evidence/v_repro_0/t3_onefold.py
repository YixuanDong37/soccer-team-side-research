"""Probe: one LOCO fold; pooled counts on its 12 FIT clips for every variant and setting (this is
exactly the view the selection rule gets; the held-out clip is not shown)."""
import pickle
import sys
from collections import Counter

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
from reid_lab import data  # noqa: E402
import explore_engine as loco  # noqa: E402

fold = sys.argv[1] if len(sys.argv) > 1 else "SNGS-107"
r = loco.run_fold(fold)
pickle.dump(r, open(f"C:/Research/plugins/tools/reid_lab/work/v_repro_0/cache/onefold_{fold}.pkl", "wb"))
print(fold, r["seconds"], "s", r["models"])
fit = [s for s in data.DEV_CLIPS if s != fold]
for variant in loco.VARIANTS:
    print(variant)
    for gi, (t, m) in enumerate(loco.GRID):
        c, lk, imp, rows = Counter(), Counter(), 0, 0
        for s in fit:
            x = r["res"][(variant, gi, s)]
            c.update(x["per_return"])
            lk.update({k: v for k, v in x["lk"].items() if k != "n"})
            imp += x["impure"]
            rows += x["rows"]
        print(f"  t_abs {t:.1f} margin {m:.2f}: {c['right']:3d}/{c['wrong']:2d}/{c['lost']:3d}  links {lk['right']}/{lk['wrong']}/{lk['unknown']}  impure {imp / rows:.4f}")
