"""v_robust_1 / r7_team_role.py -- does fuse ever link pieces of different teams or roles?

    python -I -B r7_team_role.py     -> out/r7_team_role.md, out/r7_team_role.json

Part 1 is LABEL-FREE (tracks.parquet only, computed here independently of m_appear's context):
per link and per linked group, the majority team / role labels of the (sub-)pieces, frames shared
by two pieces of one group, a second earlier / later partner, jersey reads that contradict.
Part 2 is DIAGNOSIS WITH LABELS: the annotated team / role of the majority person of each piece.
"""
import json
import sys
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import numpy as np
import pandas as pd

import vr_common as vc
from vr_common import data, prep

POINTS = ("strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall", "frozen_strict", "frozen_balanced", "frozen_recall")
J = {}
L = ["# r7: links across teams or roles", "", "## Part 1: label-free (labels of tracks.parquet)", "",
     "| file | links | groups | team labels differ | one side without team | role labels differ | of which player-goalkeeper | referee in a link | "
     "groups with 2 team labels | groups with 2 roles | groups sharing a frame | second partner | jersey reads contradict (>= 2 reads each) |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
examples = []
for point in POINTS:
    M = vc.load_method(point)
    c = Counter()
    for seq in data.DEV_CLIPS:
        cuts = prep.norm_cuts(M[seq]["cuts"])
        t = data.tracks(seq)
        ids = vc.piece_ids_after_cuts(seq, cuts)
        t = t.assign(pid=ids)
        t = t[t.pid >= 0]
        meta = {}
        for pid, g in t.groupby("pid"):
            team, role, j = g.team.dropna(), g.role.dropna(), g.jersey.dropna().astype(str)
            jm = j.mode()
            meta[int(pid)] = {"team": team.mode().iloc[0] if len(team) else None, "role": role.mode().iloc[0] if len(role) else None,
                              "teams": set(team.unique()), "frames": set(g.frame.tolist()), "first": int(g.frame.min()),
                              "jersey": jm.iloc[0] if len(jm) else None, "jn": int((j == jm.iloc[0]).sum()) if len(jm) else 0,
                              "team_share": float((team == team.mode().iloc[0]).mean()) if len(team) else None}
        nxt, prv = Counter(), Counter()
        for a, b in M[seq]["links"]:
            ma, mb = meta[a], meta[b]
            c["links"] += 1
            nxt[a] += 1
            prv[b] += 1
            if ma["team"] and mb["team"] and ma["team"] != mb["team"]:
                c["team_diff"] += 1
                examples.append(f"{point} {seq} {a}>{b}: team {ma['team']} ({ma['team_share']:.2f}) > {mb['team']} ({mb['team_share']:.2f})")
            if (ma["team"] is None) != (mb["team"] is None):
                c["team_one_missing"] += 1
            if ma["role"] != mb["role"]:
                c["role_diff"] += 1
                if {ma["role"], mb["role"]} == {"player", "goalkeeper"}:
                    c["role_pg"] += 1
                if point in ("recall", "strict"):
                    examples.append(f"{point} {seq} {a}>{b}: role {ma['role']} > {mb['role']}")
            if "referee" in (ma["role"], mb["role"]):
                c["referee"] += 1
            if ma["jersey"] and mb["jersey"] and ma["jn"] >= 2 and mb["jn"] >= 2 and ma["jersey"] != mb["jersey"]:
                c["jersey_conflict"] += 1
                if point in ("recall", "strict"):
                    examples.append(f"{point} {seq} {a}>{b}: jersey reads {ma['jersey']} x{ma['jn']} > {mb['jersey']} x{mb['jn']}")
        c["second_partner"] += sum(v > 1 for v in nxt.values()) + sum(v > 1 for v in prv.values())
        for members in vc.groups_of(M[seq]["links"]).values():
            c["groups"] += 1
            teams = {meta[p]["team"] for p in members if meta[p]["team"]}
            roles = {meta[p]["role"] for p in members}
            c["g_team2"] += len(teams) > 1
            c["g_role2"] += len(roles) > 1
            seen, shared = set(), False
            for p in members:
                if seen & meta[p]["frames"]:
                    shared = True
                seen |= meta[p]["frames"]
            c["g_shared"] += shared
    J[point] = dict(c)
    L.append(f"| {point} | {c['links']} | {c['groups']} | {c['team_diff']} | {c['team_one_missing']} | {c['role_diff']} | {c['role_pg']} | {c['referee']} | "
             f"{c['g_team2']} | {c['g_role2']} | {c['g_shared']} | {c['second_partner']} | {c['jersey_conflict']} |")
L += ["", "Examples (strict and recall):", ""] + [f"- {e}" for e in examples[:40]]

# ------------------------------------------------------------------ part 2: labels
L += ["", "## Part 2: diagnosis with labels (annotated team / role of each piece's majority person)", "",
      "Links with a matched person on both sides (>= 1 matched row): the two persons' annotated team and role.", "",
      "| point | links with both sides matched | same person | different person | of those: different annotated team | different annotated role | "
      "player-goalkeeper | with a referee / other | different person, both >= 5 rows (score.py 'wrong') | of those different team | different role |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for point in ("strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall"):
    lk = pd.read_csv(vc.OUT / f"links_{point}.csv")
    m = lk[lk.weak != "none"]
    d = m[m.weak == "disagree"]
    td = d[d.ateam_a.astype(str) != d.ateam_b.astype(str)]
    rd = d[d.arole_a.astype(str) != d.arole_b.astype(str)]
    pg = rd[rd.apply(lambda r: {r.arole_a, r.arole_b} == {"player", "goalkeeper"}, axis=1)] if len(rd) else rd
    ro = rd[rd.apply(lambda r: bool({r.arole_a, r.arole_b} & {"referee", "other"}), axis=1)] if len(rd) else rd
    w = lk[lk.judged == "wrong"]
    wt = w[w.ateam_a.astype(str) != w.ateam_b.astype(str)]
    wrl = w[w.arole_a.astype(str) != w.arole_b.astype(str)]
    J[f"labels_{point}"] = {"matched": len(m), "same": int((m.weak == "agree").sum()), "different": len(d), "different_team": len(td), "different_role": len(rd),
                            "player_goalkeeper": len(pg), "referee_other": len(ro), "judged_wrong": len(w), "judged_wrong_diff_team": len(wt), "judged_wrong_diff_role": len(wrl)}
    L.append(f"| {point} | {len(m)} | {int((m.weak == 'agree').sum())} | {len(d)} | {len(td)} | {len(rd)} | {len(pg)} | {len(ro)} | {len(w)} | {len(wt)} | {len(wrl)} |")
    if point == "recall":
        TAIL = ["", "Recall point, links whose two majority persons differ in annotated team or role:", ""]
        for r in pd.concat([td, rd]).drop_duplicates(["seq", "a", "b"]).itertuples():
            TAIL.append(f"- {r.seq} {int(r.a)}>{int(r.b)}: person {int(r.ref_a)} ({r.arole_a}, {r.ateam_a}) x{int(r.n_a)} > person {int(r.ref_b)} ({r.arole_b}, {r.ateam_b}) x{int(r.n_b)}; "
                        f"rows {int(r.rows_a)}/{int(r.rows_b)}; tracks labels {r.team_a}/{r.role_a} > {r.team_b}/{r.role_b}; judged {r.judged}")
L += TAIL
(vc.OUT / "r7_team_role.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(vc.OUT / "r7_team_role.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
print("\n".join(L))
