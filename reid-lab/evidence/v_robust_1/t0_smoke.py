"""v_robust_1 / t0_smoke.py -- smoke test of r3_sensitivity.run_clip on one clip (LABELS: scoring)."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import r3_sensitivity as r3
import vr_common as vc
seq = sys.argv[1] if len(sys.argv) > 1 else "SNGS-112"
s, out, sec = r3.run_clip(seq)
print(s, sec, "s", len(out), "cells")
for n in ("base", "margin -10%", "margin +10%", "MAX_SPEED 7.2", "WCCN_LAMBDA 0.27", "MIN_CLEAN 4", "solid from 20 boxes"):
    print(n, {p: vc.counts(out[(n, p)]["per_return"]) for p in r3.POINTS[:3]})
