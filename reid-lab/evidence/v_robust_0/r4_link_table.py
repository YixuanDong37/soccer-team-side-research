"""v_robust_0 / r4_link_table.py -- (LABELS: diagnosis) one row per link of a fuse METHOD file with
the evidence the linker had (label-free columns) and what the annotation says (label columns).

    python -I -B r4_link_table.py            -> out/links_<point>.csv for strict, balanced, recall, margin_0.50
                                                out/returns_<point>.csv (one row per annotated return)

Label-free columns (recomputed through fuse's apply path with the clip's held-out numbers):
  a, b                     the two (sub-)pieces of the link, a earlier
  gap                      frames between a's last and b's first frame
  rows_a/b, clean_a/b      boxes / clean boxes of the pieces
  cos_w                    piece-level whitened cosine;  bonus: the pair bonus in cosine units
  pos_sink, plug, kind     position assignment score, plugin join flag, border ends (0 / 1 / 2)
  m_s, m_app, m_bonus      fused score, appearance cosine and bonus of the MERGE that joined them
  m_rg, m_rh               margins of the two sides at that merge (S - max(rival, t0))
  m_order, m_size          index of the merge in the agglomeration, pieces in the merged group
  team_a/b, role_a/b       majority team / role label of the pieces in tracks.parquet
  jersey_a/b               most frequent jersey read of the piece (and its count), if any
Label columns: ref_a/ref_b (majority annotated person), n_a/n_b (its matched rows), judged
  (score.py's rule: right / wrong / unknown), weak (majority of >= 1 matched row: agree / disagree / none),
  ateam_*/arole_* annotated team and role of those persons.
"""
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_0")
import numpy as np
import pandas as pd

import vr_common as vc
import vr_labels as vl
from vr_common import data, evidence, linker, method, prep

FILES = {"strict": ("heldout", "strict", None), "balanced": ("heldout", "balanced", None), "recall": ("heldout", "recall", None),
         "margin_0.50": ("heldout", None, 0.5)}


def piece_meta(seq, cuts):
    t = data.tracks(seq)
    ids = vc.piece_ids_after_cuts(seq, cuts)
    t = t.assign(pid=ids)
    t = t[t.pid >= 0]
    out = {}
    for pid, g in t.groupby("pid"):
        team = g.team.dropna()
        role = g.role.dropna()
        j = g.jersey.dropna() if "jersey" in g else pd.Series(dtype=object)
        jm = j.astype(str).mode()
        out[int(pid)] = {"team": team.mode().iloc[0] if len(team) else None, "role": role.mode().iloc[0] if len(role) else None,
                         "role_share": float((role == role.mode().iloc[0]).mean()) if len(role) else None,
                         "jersey": jm.iloc[0] if len(jm) else None, "jersey_n": int((j.astype(str) == jm.iloc[0]).sum()) if len(jm) else 0,
                         "first": int(g.frame.min()), "last": int(g.frame.max()), "rows": len(g)}
    return out


def link_rows(seq, name, mode, point, margin):
    par = vc.params()[seq]
    m = margin if margin is not None else par["margin"][point]
    cuts = prep.cuts_for(seq, seq)
    table = evidence.pair_table(seq, cuts, method.pos_params(seq))
    bonus = method.bonus_of(table, par["pair_model"])
    log = []
    links = linker.agglomerate(seq, cuts, bonus, t_abs=par["t_abs"], margin=m, floor=par["floor"], t0=par["t0"], log=log)
    M = vc.load_method(name)[seq]
    assert sorted(map(tuple, links)) == sorted(map(tuple, M["links"])), (seq, name)
    tab = table.set_index(["a", "b"])
    meta = piece_meta(seq, cuts)
    ident = vl.identities(seq, cuts)
    attrs = vl.ref_attrs(seq)
    # merge that joined the two pieces of a link
    merge_of = {}
    for k, e in enumerate(log):
        gs, hs = set(e["g"]), set(e["h"])
        for a, b in links:
            if (a in gs and b in hs) or (a in hs and b in gs):
                merge_of[(a, b)] = (k, e)
    rows = []
    for a, b in links:
        r = {"seq": seq, "a": a, "b": b}
        ev = tab.loc[(a, b)] if (a, b) in tab.index else None
        ma, mb = meta[a], meta[b]
        r["gap"] = mb["first"] - ma["last"] - 1
        r["rows_a"], r["rows_b"] = ma["rows"], mb["rows"]
        if ev is not None:
            for c in ("clean_a", "clean_b", "cos_w", "pos_sink", "plug", "kind", "low", "pos_adm", "pos_known", "pos_unreach"):
                r[c] = float(ev[c])
        r["bonus"] = bonus.get((a, b), 0.0)
        k, e = merge_of[(a, b)]
        r.update(m_order=k, m_s=e["s"], m_app=e["app"], m_bonus=e["bonus"], m_rg=e["rival_g"], m_rh=e["rival_h"], m_size=len(e["g"]) + len(e["h"]))
        for side, mm in (("a", ma), ("b", mb)):
            r[f"team_{side}"], r[f"role_{side}"], r[f"jersey_{side}"], r[f"jersey_n_{side}"] = mm["team"], mm["role"], mm["jersey"], mm["jersey_n"]
        ia, ib = ident.get(a), ident.get(b)
        for side, i in (("a", ia), ("b", ib)):
            r[f"ref_{side}"], r[f"n_{side}"] = (i[0], i[1]) if i else (None, 0)
            at = attrs.get(i[0]) if i else None
            r[f"ateam_{side}"], r[f"arole_{side}"], r[f"ajersey_{side}"] = (at["team"], at["role"], at["jersey"]) if at else (None, None, None)
        if ia and ib and ia[1] >= 5 and ib[1] >= 5:
            r["judged"] = "right" if ia[0] == ib[0] else "wrong"
        else:
            r["judged"] = "unknown"
        r["weak"] = "none" if not (ia and ib) else ("agree" if ia[0] == ib[0] else "disagree")
        rows.append(r)
    ret = vl.returns_table(seq, links, [list(x) for x in cuts])
    return rows, ret


if __name__ == "__main__":
    from reid_lab import score  # LABELS: check of the re-derived outcomes

    for name, (mode, point, margin) in FILES.items():
        rows, rets = [], []
        for seq in data.DEV_CLIPS:
            r, t = link_rows(seq, name, mode, point, margin)
            M = vc.load_method(name)[seq]
            chk = score.score_clip(seq, M["links"], M["cuts"])
            assert [x["outcome"] for x in t] == chk["per_return"], (seq, name)
            assert sum(x["judged"] == "wrong" for x in r) == chk["links"]["wrong"] and sum(x["judged"] == "right" for x in r) == chk["links"]["right"]
            rows += r
            rets += t
        df = pd.DataFrame(rows)
        df.to_csv(vc.OUT / f"links_{name}.csv", index=False)
        pd.DataFrame(rets).to_csv(vc.OUT / f"returns_{name}.csv", index=False)
        print(name, len(df), "links;", df.judged.value_counts().to_dict(), "weak", df.weak.value_counts().to_dict(), flush=True)
