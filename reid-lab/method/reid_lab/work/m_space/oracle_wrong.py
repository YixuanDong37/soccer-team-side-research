"""ORACLE / DIAGNOSIS ONLY. The wrong links of the LOCO position linker at a threshold: what
did the linked pair look like, and what did the right candidate look like?
python -I oracle_wrong.py [variant] [theta]"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import features  # noqa: E402
import labels  # noqa: E402
import linker  # noqa: E402
from explore_models import VARIANTS  # noqa: E402

pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 500)
name = sys.argv[1] if len(sys.argv) > 1 else "adv_cv_fa"
theta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
lp = linker.loco_params(**VARIANTS[name])
cols = ["a", "b", "gap", "d", "r_adv_cv", "fwd_adv_inview_s", "back_adv_inview_s", "inview_s", "a_rows", "b_rows", "same_team", "same_plugin", "z", "p_in", "p_out", "score"]
for seq in data.DEV_CLIPS:
    weak = labels.identity_map(seq, 1)
    strong = labels.identity_map(seq, 5)
    q = linker.posteriors(features.pair_features(seq), lp[seq], "sinkhorn")
    for a, b in linker.links(features.pair_features(seq), lp[seq], theta, "sinkhorn", True, 2):
        ia, ib = strong.get(a), strong.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        print(f"\n{seq} WRONG link {a} (person {ia}) -> {b} (person {ib})")
        print(q[(q.a == a) & (q.b == b)][cols].round(2).to_string(index=False))
        cand = q[(q.b == b) & (q.a.map(weak) == ib)]
        print("  right earlier pieces for b:" if len(cand) else "  no admissible right earlier piece for b")
        if len(cand):
            print(cand[cols].round(2).to_string(index=False))
        cand = q[(q.a == a) & (q.b.map(weak) == ia)]
        print("  right later pieces for a:" if len(cand) else "  no admissible right later piece for a")
        if len(cand):
            print(cand[cols].round(2).to_string(index=False))
