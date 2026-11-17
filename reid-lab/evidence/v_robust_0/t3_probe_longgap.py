"""probe (LABELS, diagnosis): evidence behind the links over more than 10 s (recall point and margin 0.5)."""
import sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
for point in ("margin_0.50", "recall"):
    lk = pd.read_csv(f"C:/Research/plugins/tools/reid_lab/work/v_robust_0/out/links_{point}.csv")
    for name, s in (("gap > 250", lk[lk.gap > 250]), ("gap 10-250", lk[(lk.gap >= 10) & (lk.gap <= 250)])):
        q = lambda c: " / ".join(f"{x:.2f}" for x in np.quantile(s[c].dropna(), [0.1, 0.5, 0.9]))
        print(point, name, len(s), "links; m_app", q("m_app"), "; m_bonus", q("m_bonus"), "; pos_sink", q("pos_sink"), "; share with bonus > 0.3:", round(float((s.m_bonus > 0.3).mean()), 2),
              "; min rows", q("rows_a"), q("rows_b"), "; weak", s.weak.value_counts().to_dict(), "; kind", s.kind.value_counts().to_dict())
    d = lk[(lk.gap > 250) & (lk.weak == "disagree")]
    for r in d.itertuples():
        print("   long-gap disagreement:", r.seq, int(r.a), int(r.b), "gap", int(r.gap), "rows", r.rows_a, r.rows_b, "persons", r.ref_a, r.n_a, r.ref_b, r.n_b, "app", round(r.m_app, 2), "bonus", round(r.m_bonus, 2), r.judged)
