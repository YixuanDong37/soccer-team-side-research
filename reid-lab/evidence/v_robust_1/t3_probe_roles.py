"""v_robust_1 / t3_probe_roles.py -- (reads out/links_*.csv: LABELS in the a*/ref* columns) the links whose two pieces carry
different role labels in tracks.parquet, and the links whose majority persons differ in annotated team.  -> out/t3_probe_roles.txt"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import pandas as pd
import vr_common as vc
out = []
for point in ("strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall"):
    lk = pd.read_csv(vc.OUT / f"links_{point}.csv")
    d = lk[lk.role_a.astype(str) != lk.role_b.astype(str)]
    out.append(f"{point}: links with different role labels in tracks: {len(d)}")
    for r in d.itertuples():
        out.append(f"  {r.seq} {int(r.a)}>{int(r.b)} stage {int(r.stage)}: tracks {r.team_a}/{r.role_a} ({r.role_share_a:.2f} of rows) > {r.team_b}/{r.role_b} ({r.role_share_b:.2f}); rows {int(r.rows_a)}/{int(r.rows_b)}; "
                   f"annotated person {r.ref_a} ({r.arole_a}, {r.ateam_a}) x{int(r.n_a)} > {r.ref_b} ({r.arole_b}, {r.ateam_b}) x{int(r.n_b)}; judged {r.judged}, weak {r.weak}; cos_w {r.cos_w:.2f}, bonus {r.bonus:.2f}")
    t = lk[(lk.weak == "disagree") & (lk.ateam_a.astype(str) != lk.ateam_b.astype(str))]
    out.append(f"{point}: links whose two majority persons are of different ANNOTATED teams: {len(t)} (judged wrong: {int((t.judged == 'wrong').sum())})")
    for r in t.itertuples():
        out.append(f"  {r.seq} {int(r.a)}>{int(r.b)} stage {int(r.stage)}: tracks {r.team_a} ({r.team_share_a:.2f}) > {r.team_b} ({r.team_share_b:.2f}); rows {int(r.rows_a)}/{int(r.rows_b)}; "
                   f"person {int(r.ref_a)} ({r.arole_a}, {r.ateam_a}) x{int(r.n_a)} of {int(r.matched_a)} matched > {int(r.ref_b)} ({r.arole_b}, {r.ateam_b}) x{int(r.n_b)} of {int(r.matched_b)}; judged {r.judged}")
    m = lk[(lk.team_a.isna()) != (lk.team_b.isna())]
    out.append(f"{point}: links where exactly one piece has no team label: {len(m)}; judged right / wrong / unknown: {int((m.judged == 'right').sum())} / {int((m.judged == 'wrong').sum())} / {int((m.judged == 'unknown').sum())}; weak disagree {int((m.weak == 'disagree').sum())}")
(vc.OUT / "t3_probe_roles.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
