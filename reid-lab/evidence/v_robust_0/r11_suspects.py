"""v_robust_0 / r11_suspects.py -- (LABELS: diagnosis) for every link of the recall point that
score.py judges wrong: what did the annotation's own continuation look like to the linker?

    python -I -B r11_suspects.py     -> out/r11_suspects.md

For a wrong link a > b (annotated persons p > q, a earlier) the table lists
  chosen      cos_w and position score of (a, b), the pair the linker joined;
  true next   the best pair (a, x) where x is a later piece of the SAME person p (>= 5 matched rows):
              what the annotation says a should have continued as;
  true prev   the best pair (y, b) where y is an earlier piece of person q: what should precede b;
  mirror pair the highest cos_w between an earlier piece of q and a later piece of p (the other half
              of an exchange: high = after the gap p's annotation continues q's earlier look);
  swap        whether the clip holds the mirrored wrong link (a piece of q joined to a later piece of p),
              the numeric signature of two annotated identities exchanged after a gap.
Numbers only: nobody looked at frames, so "the annotation is wrong" stays a hypothesis.
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_0")
import numpy as np
import pandas as pd

import vr_common as vc
import vr_labels as vl
from vr_common import data, evidence, method, prep

lk = pd.read_csv(vc.OUT / "links_recall.csv")
wrong = lk[lk.judged == "wrong"]
L = ["# r11: the links judged wrong at the recall point against the annotation's own continuation (diagnosis)", "",
     "| clip | wrong link (persons) | chosen: cos_w, sink, fused | true next of a (person p): best cos_w, sink, fused, piece, gap | true prev of b (person q): best cos_w, sink, fused, piece | mirror pair (earlier piece of q > later piece of p): cos_w | mirrored link in the clip | reading |",
     "|---|---|---|---|---|---|---|---|"]
n_better = n_none = 0
for seq, g in wrong.groupby("seq", sort=False):
    par = vc.params()[seq]
    cuts = prep.cuts_for(seq, seq)
    tab = evidence.pair_table(seq, cuts, method.pos_params(seq))
    bonus = method.bonus_of(tab, par["pair_model"])
    tab = tab.assign(bonus=[bonus.get((int(a), int(b)), 0.0) for a, b in zip(tab.a, tab.b)])
    tab["fused"] = tab.cos_w + tab.bonus
    ident = vl.identities(seq, cuts)
    firm = {p: v[0] for p, v in ident.items() if v[1] >= 5}
    all_links = lk[lk.seq == seq]
    for r in g.itertuples():
        a, b, p, q = int(r.a), int(r.b), int(r.ref_a), int(r.ref_b)
        ch = tab[(tab.a == a) & (tab.b == b)]
        chs = f"{ch.cos_w.iloc[0]:.2f}, {ch.pos_sink.iloc[0]:.2f}, {ch.fused.iloc[0]:.2f}" if len(ch) else "-"
        nx = tab[(tab.a == a) & tab.b.map(lambda x: firm.get(int(x)) == p)].sort_values("fused", ascending=False)
        pv = tab[(tab.b == b) & tab.a.map(lambda x: firm.get(int(x)) == q)].sort_values("fused", ascending=False)
        nxs = f"{nx.cos_w.iloc[0]:.2f}, {nx.pos_sink.iloc[0]:.2f}, {nx.fused.iloc[0]:.2f}, piece {int(nx.b.iloc[0])}, gap {int(nx.gap.iloc[0])}" if len(nx) else "no later piece of p"
        pvs = f"{pv.cos_w.iloc[0]:.2f}, {pv.pos_sink.iloc[0]:.2f}, {pv.fused.iloc[0]:.2f}, piece {int(pv.a.iloc[0])}" if len(pv) else "no earlier piece of q"
        mirror = all_links[(all_links.ref_a == q) & (all_links.ref_b == p) & (all_links.weak == "disagree")]
        ms = "; ".join(f"{int(m.a)}>{int(m.b)} ({m.judged})" for m in mirror.itertuples()) or "no"
        mp = "-"
        if len(nx) and len(pv):
            best = None
            for y in pv.a.astype(int).tolist():
                for x in nx.b.astype(int).tolist():
                    row = tab[(tab.a == y) & (tab.b == x)]
                    if len(row) and (best is None or float(row.cos_w.iloc[0]) > best[0]):
                        best = (float(row.cos_w.iloc[0]), y, x)
            mp = f"{best[0]:.2f} ({best[1]} > {best[2]})" if best else "not a candidate pair"
        cf = float(ch.fused.iloc[0]) if len(ch) else np.nan
        better_next = len(nx) and float(nx.fused.iloc[0]) > cf
        better_prev = len(pv) and float(pv.fused.iloc[0]) > cf
        if not len(nx) and not len(pv):
            reading = "the annotation offers no alternative on either side: p never comes back and q was not there before"
            n_none += 1
        elif better_next or better_prev:
            reading = "the annotated continuation scored HIGHER than the chosen pair on one side: the mutual-best / rival rule decided"
            n_better += 1
        else:
            reading = "the annotated continuation scored lower than the chosen pair on both sides: evidence and annotation disagree"
        L.append(f"| {seq} | {a} > {b} ({p} > {q}) | {chs} | {nxs} | {pvs} | {mp} | {ms} | {reading} |")
L += ["", f"{len(wrong)} wrong links: {n_none} without any annotated alternative, {n_better} where the annotated continuation scored higher on one side.", ""]
(vc.OUT / "r11_suspects.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
