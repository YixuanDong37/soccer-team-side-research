"""DIAGNOSIS ONLY: which returns change outcome between methods, the false cuts, the links of
the re-run mutual rule that are wrong.

    python -I diag_final.py [p0]
"""
import json
import sys
from collections import Counter

sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np
import pandas as pd
from reid_lab import data, score
import features as F

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 800)
p0 = sys.argv[1] if len(sys.argv) > 1 else "0.7"
M = {k: json.load(open(HERE + f"methods/heldout_p{p0}_{k}.json"))["clips"] for k in ("split", "mutual", "mutual_rerun", "plugin")}
base = {k: score.baseline_links(k) for k in ("split", "mutual", "plugin")}
trans = {k: Counter() for k in ("split", "mutual", "mutual_rerun", "plugin")}
for seq in data.DEV_CLIPS:
    c = score.clip(seq)
    for k in trans:
        b = base["mutual" if k == "mutual_rerun" else k][seq]
        r0 = score.score_clip(seq, b["links"])["per_return"]
        r1 = score.score_clip(seq, M[k][seq]["links"], M[k][seq]["cuts"])["per_return"]
        for ret, x, y in zip(c.returns, r0, r1):
            trans[k][(x, y)] += 1
            if x != y and x != "undetected":
                print(f"{seq} {k}: return ref {ret['ref_id']} last {ret['last']} first {ret['first']} gap {ret['gap']}: {x} -> {y}; A piece {int(c.piece[ret['a']])} B piece {int(c.piece[ret['b']])}")
for k, v in trans.items():
    print(k, {f"{a}->{b}": n for (a, b), n in sorted(v.items()) if a != "undetected"})

cuts = pd.read_csv(HERE + f"methods/heldout_p{p0}_cuts.csv")
print("false cuts (F) and weak false (f):")
print(cuts[cuts.label.isin(["F", "f"])][["seq", "piece", "f0", "f1", "gap", "kind", "cos", "alt_d", "alt_r", "d_m", "dt", "nb", "na", "rows_before", "rows_after", "role", "prob", "label"]].round(3).to_string())
print("cuts by label and size of the smaller side:")
cuts["small"] = np.minimum(cuts.rows_before, cuts.rows_after)
print(cuts.groupby([pd.cut(cuts.small, [0, 1, 2, 5, 10, 20, 50, 10000], right=False), "label"], observed=False).size().unstack(fill_value=0))
print("cuts by gap and label:")
print(cuts.groupby([pd.cut(cuts.gap, [0, 2, 5, 10, 25, 50, 100, 1000], right=False), "label"], observed=False).size().unstack(fill_value=0))
print("cuts per clip:", cuts.groupby("seq").size().to_dict())
print("kinds:", cuts.kind.value_counts().to_dict(), " roles:", cuts.role.fillna("none").value_counts().to_dict())
