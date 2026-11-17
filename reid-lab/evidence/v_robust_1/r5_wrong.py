"""v_robust_1 / r5_wrong.py -- (LABELS: diagnosis; numbers are not method results) what the remaining
wrong returns and wrong links are, from out/links_<point>.csv and out/returns_<point>.csv.

    python -I -B r5_wrong.py     -> out/r5_wrong.md

For every wrong return: the annotated person, the gap, the pieces of rows A and B, the chain B's
piece was linked into (each piece with its majority annotated person and matched rows) and the
link(s) of that chain that join two different annotated people, with the linker's evidence.
For every link score.py judges wrong: the evidence, next to the distribution over the right links.
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import numpy as np
import pandas as pd

import vr_common as vc
import vr_labels as vl
from vr_common import data, prep

POINTS = ("strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall")
EV = ["gap", "rows_a", "rows_b", "clean_a", "clean_b", "cos_w", "bonus", "pos_sink", "plug", "kind", "m_s", "m_app", "m_bonus", "m_rg", "m_rh", "m_size"]
L = ["# r5: the remaining wrong returns and wrong links (diagnosis with labels)", ""]


def chain_of(links, piece):
    g = vc.groups_of(links)
    for members in g.values():
        if piece in members:
            return members
    return [piece]


for point in POINTS:
    lk = pd.read_csv(vc.OUT / f"links_{point}.csv")
    rt = pd.read_csv(vc.OUT / f"returns_{point}.csv")
    M = vc.load_method(point)
    L += [f"## {point}", ""]
    wr = rt[rt.outcome == "wrong"]
    L += [f"### Wrong returns: {len(wr)} ({wr.seq.value_counts().to_dict()})", ""]
    for r in wr.itertuples():
        cuts = prep.norm_cuts(M[r.seq]["cuts"])
        ident = vl.identities(r.seq, cuts)
        attrs = vl.ref_attrs(r.seq)
        members = chain_of(M[r.seq]["links"], int(r.piece_b))
        meta = {}
        t = data.tracks(r.seq)
        ids = vc.piece_ids_after_cuts(r.seq, cuts)
        for p in members:
            f = t.frame.to_numpy()[ids == p]
            meta[p] = (int(f.min()), int(f.max()), int(len(f)))
        members = sorted(members, key=lambda p: meta[p][0])
        at = attrs[int(r.ref_id)]
        L.append(f"- **{r.seq} person {int(r.ref_id)}** ({at['role']}, team {at['team']}, jersey {at['jersey']}): away frames {int(r.last) + 1}-{int(r.first) - 1} "
                 f"(gap {int(r.gap)} = {r.gap / 25:.1f} s); row A in piece {int(r.piece_a)}, row B in piece {int(r.piece_b)}; rows of B's group before B: {r.b_group_before}")
        L.append("  - chain of B's piece: " + " > ".join(
            f"{p} [{meta[p][0]}-{meta[p][1]}, {meta[p][2]} rows, person {ident.get(p, (None, 0, 0))[0]} x{ident.get(p, (None, 0, 0))[1]}]" for p in members))
        sub = lk[(lk.seq == r.seq) & lk.a.isin(members) & lk.b.isin(members)]
        bad = sub[(sub.weak == "disagree")]
        for b in bad.itertuples():
            aa, ab = attrs.get(int(b.ref_a), {}), attrs.get(int(b.ref_b), {})
            L.append(f"  - link joining two people: {int(b.a)} > {int(b.b)} (persons {int(b.ref_a)} x{int(b.n_a)} > {int(b.ref_b)} x{int(b.n_b)}; "
                     f"annotated {aa.get('role')}/{aa.get('team')}/#{aa.get('jersey')} > {ab.get('role')}/{ab.get('team')}/#{ab.get('jersey')}; judged {b.judged}): "
                     f"gap {int(b.gap)}, rows {int(b.rows_a)}/{int(b.rows_b)}, clean {b.clean_a:.0f}/{b.clean_b:.0f}, cos_w {b.cos_w:.2f}, bonus {b.bonus:.2f} "
                     f"(pos_sink {b.pos_sink:.2f}, plug {b.plug:.0f}), merge: S {b.m_s:.2f} = app {b.m_app:.2f} + bonus {b.m_bonus:.2f}, margins {b.m_rg:.2f} / {b.m_rh:.2f}, "
                     f"group size {int(b.m_size)}, stage {int(b.stage)}; labels in tracks: {b.team_a}/{b.role_a} > {b.team_b}/{b.role_b}, jersey reads {b.jersey_a} (x{int(b.jersey_n_a)}) > {b.jersey_b} (x{int(b.jersey_n_b)})")
        if not len(bad):
            L.append("  - no link of the chain joins two different majority persons (the mix is inside a piece)")
    L += ["", f"### Links judged wrong by score.py: {int((lk.judged == 'wrong').sum())}; weak-label disagreements among the unjudged: {int(((lk.judged == 'unknown') & (lk.weak == 'disagree')).sum())}", "",
          "| clip | link | stage | persons (rows) | gap | rows | clean | cos_w | bonus | sink | plug | S = app + bonus | margins | size |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in lk[lk.judged == "wrong"].itertuples():
        L.append(f"| {b.seq} | {int(b.a)} > {int(b.b)} | {int(b.stage)} | {int(b.ref_a)} ({int(b.n_a)}) > {int(b.ref_b)} ({int(b.n_b)}) | {int(b.gap)} | {int(b.rows_a)}/{int(b.rows_b)} | {b.clean_a:.0f}/{b.clean_b:.0f} | "
                 f"{b.cos_w:.2f} | {b.bonus:.2f} | {b.pos_sink:.2f} | {b.plug:.0f} | {b.m_s:.2f} = {b.m_app:.2f} + {b.m_bonus:.2f} | {b.m_rg:.2f} / {b.m_rh:.2f} | {int(b.m_size)} |")
    L += ["", "Evidence quantiles (10% / 50% / 90%) by label:", "", "| column | judged right | judged wrong | weak disagree (unjudged) | weak agree (unjudged) |", "|---|---|---|---|---|"]
    grp = {"judged right": lk[lk.judged == "right"], "judged wrong": lk[lk.judged == "wrong"],
           "weak disagree": lk[(lk.judged == "unknown") & (lk.weak == "disagree")], "weak agree": lk[(lk.judged == "unknown") & (lk.weak == "agree")]}
    for c in EV:
        cells = []
        for g in grp.values():
            v = g[c].dropna().to_numpy(float)
            cells.append(" / ".join(f"{x:.2f}" for x in np.quantile(v, [0.1, 0.5, 0.9])) if len(v) else "-")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")
(vc.OUT / "r5_wrong.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
