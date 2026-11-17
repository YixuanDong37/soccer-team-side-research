"""v_robust_1 / t2_probe_bonus.py -- (reads out/links_*.csv; LABELS only through their judged column) how much of a link's
score is position bonus, by gap length, and how flat the bonus is in the position score.   -> out/t2_probe_bonus.txt"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import numpy as np
import pandas as pd
import vr_common as vc

out = []
P = vc.params()
w = np.array(P["all13"]["pair_model"]["w"])
ratio = w[1] / w[0]
f = lambda s: ratio * (np.log(np.clip(s, 1e-3, 0.98) / (1 - np.clip(s, 1e-3, 0.98))) + np.log(999))
out.append("position bonus as a function of the position score pos_sink (all-13 pair model, cosine units): "
           + ", ".join(f"{s:g} -> {f(s):.2f}" for s in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9, 0.98)))
out.append("a link at margin m needs appearance + bonus >= 0.5 + m (nobody rival 0.5): strict 1.10, balanced 0.90, recall 0.70")
for point in ("strict", "balanced", "recall"):
    lk = pd.read_csv(vc.OUT / f"links_{point}.csv")
    co = lk[lk.stage == 1]
    out.append(f"\n{point}: core links {len(co)}")
    for name, lo, hi in (("gap < 50", -10**6, 49), ("50-124", 50, 124), ("125-250", 125, 250), ("> 250 (10 s)", 251, 10**6)):
        s = co[(co.gap >= lo) & (co.gap <= hi)]
        if not len(s):
            continue
        q = lambda c: " / ".join(f"{x:.2f}" for x in np.quantile(s[c].dropna(), [0.1, 0.5, 0.9]))
        out.append(f"  {name}: n {len(s)}; merge appearance 10/50/90%: {q('m_app')}; merge bonus: {q('m_bonus')}; pair pos_sink: {q('pos_sink')}; "
                   f"links whose appearance alone reaches the bar (m_app >= 0.5 + margin): {int((s.m_app >= 0.5 + s.margin_used).sum())}; "
                   f"pos_sink < 0.1: {int((s.pos_sink < 0.1).sum())}; judged right / wrong: {int((s.judged == 'right').sum())} / {int((s.judged == 'wrong').sum())}")
(vc.OUT / "t2_probe_bonus.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
