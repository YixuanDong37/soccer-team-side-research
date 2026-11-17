"""DIAGNOSIS ONLY: true switches that the held-out cuts do not separate, with the first-level
candidate nearest to each (features on the whole piece, probability under the held-out weights).

    python -I diag_missed.py [p0]        (after make_methods.py)
"""
import sys

sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np
import pandas as pd
from reid_lab import data
import features as F
import truth as T
import loco

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 800)
p0 = sys.argv[1] if len(sys.argv) > 1 else "0.7"
cuts = pd.read_csv(HERE + f"methods/heldout_p{p0}_cuts.csv")
tab = loco.table()
rows = []
for seq in data.DEV_CLIPS:
    sws = T.switches(seq)
    hit = set(cuts[(cuts.seq == seq) & (cuts.sw_id >= 0)].sw_id.tolist())
    par = F.heldout_params(seq, float(p0))
    for k, s in enumerate(sws):
        if k in hit:
            continue
        c = tab[(tab.seq == seq) & (tab.sw_id == k)]
        z = F.apply_model(par, c) if len(c) else np.array([])
        best = c.iloc[int(np.argmax(z))] if len(c) else None
        row = {"seq": seq[-3:], "piece": s["piece"], "a>b": f"{s['from_ref']}>{s['to_ref']}", "last_a": s["last_a"], "first_b": s["first_b"],
               "rows_a": s["rows_a"], "rows_b": s["rows_b"], "minor": min(s["rows_a"], s["rows_b"]), "n_cand": len(c)}
        if best is not None:
            row.update({"f0": best.f0, "f1": best.f1, "gap": best.gap, "kind": best.kind, "cos": round(best.cos, 3), "alt_d": round(best.alt_d, 3),
                        "alt_r": round(best.alt_r, 3), "d_m": round(best.d_m, 1), "prob": round(float(1 / (1 + np.exp(-z.max()))), 3)})
        rows.append(row)
df = pd.DataFrame(rows).sort_values("minor", ascending=False)
print(df.to_string(index=False))
print("missed", len(df), "of", sum(len(T.switches(s)) for s in data.DEV_CLIPS), "; rows on the smaller side of the missed switches:", int(df.minor.sum()))
print("without any candidate:", int((df.n_cand == 0).sum()), "; best candidate cos >= 0.9:", int((df.cos >= 0.9).sum()), "; 0.85-0.9:", int(((df.cos >= 0.85) & (df.cos < 0.9)).sum()), "; < 0.85:", int((df.cos < 0.85).sum()))
