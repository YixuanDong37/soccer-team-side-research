"""probe (LABELS, diagnosis): links whose tracks labels differ in role or lack a team on one side."""
import sys
sys.dont_write_bytecode = True
import pandas as pd
for point in ("strict", "margin_0.50", "balanced", "recall"):
    lk = pd.read_csv(f"C:/Research/plugins/tools/reid_lab/work/v_robust_0/out/links_{point}.csv")
    rd = lk[lk.role_a != lk.role_b]
    print(point, "role labels differ:", len(rd))
    for r in rd.itertuples():
        print("   ", r.seq, int(r.a), int(r.b), r.role_a, r.role_b, "rows", r.rows_a, r.rows_b, "annot", r.ref_a, r.arole_a, r.n_a, ">", r.ref_b, r.arole_b, r.n_b, "judged", r.judged, "weak", r.weak, "cos", round(r.m_app, 2), "bonus", round(r.m_bonus, 2))
    nt = lk[lk.team_a.isna() != lk.team_b.isna()]
    bt = lk[lk.team_a.isna() & lk.team_b.isna()]
    print("  one side without team:", len(nt), nt.weak.value_counts().to_dict(), nt.judged.value_counts().to_dict(), "| both without:", len(bt), bt.weak.value_counts().to_dict())
    d = nt[nt.weak == "disagree"]
    print("  of those, annotated teams differ:", int((d.ateam_a.astype(str) != d.ateam_b.astype(str)).sum()), "of", len(d))
    # short-piece links
    short = lk[(lk.rows_a < 5) | (lk.rows_b < 5)]
    print("  links with a piece of < 5 rows:", len(short), short.weak.value_counts().to_dict(), "| others:", lk.drop(short.index).weak.value_counts().to_dict())
    neg = lk[lk.gap < 0]
    print("  links between interleaved pieces (b starts before a ends):", len(neg), neg.weak.value_counts().to_dict(), neg.judged.value_counts().to_dict())
