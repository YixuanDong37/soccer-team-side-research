"""(1) Team and (6) role: gbf's labels against the annotation (diagnosis; uses labels).

Row level: matched rows. Piece level: pieces with a known identity (majority annotated person,
>= 5 matched rows). Pair level: the 137 scored returns (piece of row A, piece of row B) and all
time-disjoint piece pairs of different persons.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import common as C  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
data = C.data
res: dict = {}

rows_all, pieces_all, ret_all = [], [], []
for seq in C.CLIPS:
    t = data.tracks(seq)
    m = C.matched(seq).copy()
    for c in ("team", "role", "track_id", "x1", "y1", "x2", "y2"):
        m[c] = t[c].to_numpy()[m.trow.to_numpy()]
    side = C.touch_sides(m.x1, m.y1, m.x2, m.y2)
    m["edge"] = side["left"] | side["right"] | side["top"] | side["bottom"]
    m["h"] = m.y2 - m.y1
    m["seq"] = seq
    rows_all.append(m)

    p = C.piece_ident(seq).copy()
    p["seq"] = seq
    verdict = {int(x["track_id"]): x for x in data.refine(seq)["teams"]["tracks"]}
    p["verdict"] = [verdict.get(int(i), {}).get("verdict", "not_listed") for i in p.piece]
    p["cleared"] = [verdict.get(int(i), {}).get("cleared") for i in p.piece]
    tt = t[t.track_id >= 0]
    g = tt.groupby("track_id")
    p["team_share"] = p.piece.map(g.team.apply(lambda s: s.notna().mean()))
    p["team_values"] = p.piece.map(g.team.apply(lambda s: s.dropna().nunique()))
    p["team_major_share"] = p.piece.map(g.team.apply(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan))
    p["role_major_share"] = p.piece.map(g.role.apply(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan))
    p["role_values"] = p.piece.map(g.role.apply(lambda s: s.dropna().nunique()))
    pieces_all.append(p)
    ret_all.append(C.returns(seq))

R = pd.concat(rows_all, ignore_index=True)
P = pd.concat(pieces_all, ignore_index=True)
# Per-clip left/right mapping: a clip whose labelled rows mostly disagree with the annotation is
# 'flipped' (the two teams are told apart, but named the other way round). team_as_given keeps
# gbf's label; team is the label after undoing a flip, so that 'accuracy' below measures whether
# the two teams are told apart, which is what a same-team gate needs.
_pl = R[R.role_ref.isin(["player", "goalkeeper"]) & R.team.notna()]
flipped = sorted(seq for seq, g in _pl.groupby("seq") if (g.team == g.team_ref).mean() < 0.5)
print("clips whose left/right naming is flipped against the annotation:", flipped)
swap = {"left": "right", "right": "left"}
for D in (R, P):
    D["team_as_given"] = D["team"]
    f = D.seq.isin(flipped) & D.team.notna()
    D.loc[f, "team"] = D.loc[f, "team"].map(swap)
res["team_flipped_clips"] = flipped
_g = R[R.role_ref.isin(["player", "goalkeeper"]) & R.team_as_given.notna()]
res["team_rows_accuracy_as_given"] = round(float((_g.team_as_given == _g.team_ref).mean()), 4)
print("row accuracy with the labels as given (no flip undone):", res["team_rows_accuracy_as_given"])
RET = pd.concat(ret_all, ignore_index=True)
P.to_csv(C.OUT / "t1_pieces.csv", index=False)

# ---------------------------------------------------------------- team, row level
print("=== TEAM, row level (matched rows) ===")
pl = R[R.role_ref.isin(["player", "goalkeeper"])]
cov = pl.team.notna().mean()
acc = (pl.team == pl.team_ref)[pl.team.notna()].mean()
res["team_rows"] = {"matched_rows_of_annotated_players_gk": len(pl), "coverage": round(cov, 4), "accuracy_where_labelled": round(acc, 4)}
print(res["team_rows"])
rf = R[R.role_ref == "referee"]
res["team_rows"]["annotated_referee_rows_with_a_team_label"] = round(rf.team.notna().mean(), 4)
print("annotated referee rows with a gbf team label:", round(rf.team.notna().mean(), 4), "of", len(rf))
per = []
for seq, g in pl.groupby("seq"):
    lab = g[g.team.notna()]
    per.append({"seq": seq, "rows": len(g), "coverage": g.team.notna().mean(), "accuracy": (lab.team == lab.team_ref).mean(),
                "L->L": int(((lab.team == "left") & (lab.team_ref == "left")).sum()), "L->R": int(((lab.team == "left") & (lab.team_ref == "right")).sum()),
                "R->L": int(((lab.team == "right") & (lab.team_ref == "left")).sum()), "R->R": int(((lab.team == "right") & (lab.team_ref == "right")).sum())})  # fmt: skip
per = pd.DataFrame(per)
print(per.to_string())
per.to_csv(C.OUT / "t1_team_rows_per_clip.csv", index=False)
print("by gbf role of the row:")
print(pl.groupby("role").apply(lambda g: pd.Series({"rows": len(g), "coverage": g.team.notna().mean(), "accuracy": (g.team == g.team_ref)[g.team.notna()].mean()}), include_groups=False).to_string())
print("by annotated role:")
print(pl.groupby("role_ref").apply(lambda g: pd.Series({"rows": len(g), "coverage": g.team.notna().mean(), "accuracy": (g.team == g.team_ref)[g.team.notna()].mean()}), include_groups=False).to_string())
print("by edge / box height (720p):")
pl = pl.assign(hbin=pd.cut(pl.h, [0, 30, 40, 60, 90, 2000]))
tab = pl.groupby(["edge", "hbin"], observed=True).apply(lambda g: pd.Series({"rows": len(g), "coverage": g.team.notna().mean(), "accuracy": (g.team == g.team_ref)[g.team.notna()].mean()}), include_groups=False)
print(tab.to_string())
tab.reset_index().to_csv(C.OUT / "t1_team_rows_by_edge_height.csv", index=False)

# ---------------------------------------------------------------- team, piece level
print("\n=== TEAM, piece level ===")
PI = P[P.ref_id.notna() & P.role_ref.isin(["player", "goalkeeper"])].copy()
PI["has_team"] = PI.team.notna()
PI["team_ok"] = PI.team == PI.team_ref
print("pieces:", len(P), "with identity:", int(P.ref_id.notna().sum()), "of annotated players/GK:", len(PI))


def team_line(name, g):
    lab = g[g.has_team]
    return {"set": name, "pieces": len(g), "coverage": g.has_team.mean() if len(g) else np.nan, "accuracy": lab.team_ok.mean() if len(lab) else np.nan,
            "wrong": int((~lab.team_ok).sum()), "mixed_labels": int((g.team_values > 1).sum())}  # fmt: skip


RS = RET[RET.scored]
a_keys = set(zip(RS.seq, RS.piece_a))
b_keys = set(zip(RS.seq, RS.piece_b))
key = list(zip(PI.seq, PI.piece))
PI["is_ret_a"] = [k in a_keys for k in key]
PI["is_ret_b"] = [k in b_keys for k in key]
lines = [team_line("all pieces with identity (players/GK)", PI),
         team_line("starts at image edge", PI[PI.starts_at_edge]),
         team_line("ends at image edge", PI[PI.ends_at_edge]),
         team_line("starts or ends at edge", PI[PI.starts_at_edge | PI.ends_at_edge]),
         team_line("neither", PI[~(PI.starts_at_edge | PI.ends_at_edge)]),
         team_line("re-entry pieces (k >= 1)", PI[PI.k >= 1]),
         team_line("piece of a return's row A", PI[PI.is_ret_a]),
         team_line("piece of a return's row B", PI[PI.is_ret_b]),
         team_line("rows < 10", PI[PI.rows < 10]),
         team_line("rows 10-49", PI[(PI.rows >= 10) & (PI.rows < 50)]),
         team_line("rows 50-199", PI[(PI.rows >= 50) & (PI.rows < 200)]),
         team_line("rows >= 200", PI[PI.rows >= 200]),
         team_line("annotated goalkeepers", PI[PI.role_ref == "goalkeeper"]),
         ]  # fmt: skip
for v in sorted(PI.verdict.unique()):
    lines.append(team_line(f"team check verdict = {v}", PI[PI.verdict == v]))
T = pd.DataFrame(lines)
print(T.to_string())
T.to_csv(C.OUT / "t1_team_pieces.csv", index=False)
res["team_pieces"] = T.to_dict("records")
print("cleared reasons:", P.cleared.value_counts(dropna=False).to_dict())
print("wrong-team pieces by clip:", PI[PI.has_team & ~PI.team_ok].groupby("seq").size().to_dict())
print("wrong-team pieces rows quantiles:", C.qtable(PI[PI.has_team & ~PI.team_ok].rows), " purity:", C.qtable(PI[PI.has_team & ~PI.team_ok].purity))
print("wrong-team pieces with purity >= 0.9:", int((PI.has_team & ~PI.team_ok & (PI.purity >= 0.9)).sum()))
# pieces of annotated referees that carry a team label
PR = P[P.ref_id.notna() & (P.role_ref == "referee")]
print("pieces of annotated referees:", len(PR), "with a gbf team:", int(PR.team.notna().sum()), "gbf role:", PR.role.value_counts().to_dict())
res["team_referee_pieces"] = {"pieces": len(PR), "with_team": int(PR.team.notna().sum())}

# ---------------------------------------------------------------- team, pair level
print("\n=== TEAM, pair level ===")
pk = {(r.seq, r.piece): r for r in P.itertuples()}
rows = []
for r in RS.itertuples():
    a, b = pk[(r.seq, r.piece_a)], pk[(r.seq, r.piece_b)]
    rows.append({"seq": r.seq, "gap": r.gap, "a_team": a.team, "b_team": b.team, "ref_team": C.person_attrs(r.seq).at[r.ref_id, "team"],
                 "a_ver": a.verdict, "b_ver": b.verdict, "a_role": a.role, "b_role": b.role, "ref_role": C.person_attrs(r.seq).at[r.ref_id, "role"],
                 "a_ident_ok": a.ref_id == r.ref_id, "b_ident_ok": b.ref_id == r.ref_id, "a_rows": a.rows, "b_rows": b.rows,
                 "a_role_share": a.role_major_share, "b_role_share": b.role_major_share})  # fmt: skip
RP = pd.DataFrame(rows)
RP.to_csv(C.OUT / "t1_return_pairs.csv", index=False)
both = RP.a_team.notna() & RP.b_team.notna()
st = {"scored_returns": len(RP), "both_pieces_have_team": int(both.sum()), "labels_equal": int((RP.a_team == RP.b_team)[both].sum()),
      "labels_differ": int((RP.a_team != RP.b_team)[both].sum()), "one_missing": int((RP.a_team.notna() ^ RP.b_team.notna()).sum()),
      "both_missing": int((RP.a_team.isna() & RP.b_team.isna()).sum()),
      "A_team_wrong": int(((RP.a_team != RP.ref_team) & RP.a_team.notna()).sum()), "B_team_wrong": int(((RP.b_team != RP.ref_team) & RP.b_team.notna()).sum()),
      "both_verified(consistent)": int(((RP.a_ver == "consistent") & (RP.b_ver == "consistent")).sum())}  # fmt: skip
print(st)
res["team_return_pairs"] = st
print("differing-label returns:")
print(RP[both & (RP.a_team != RP.b_team)].to_string())
# all time-disjoint pairs of pieces with identity: how much does 'same team label' prune?
cnt = {"same_person": [0, 0, 0], "diff_person_same_team": [0, 0, 0], "diff_person_diff_team": [0, 0, 0]}  # [pairs, both labelled, labels equal]
for seq, g in PI.groupby("seq"):
    g = g.sort_values("first").reset_index(drop=True)
    f, l, rid, tm, rt = g["first"].to_numpy(), g["last"].to_numpy(), g.ref_id.to_numpy(), g.team.to_numpy(object), g.team_ref.to_numpy(object)
    for i in range(len(g)):
        for j in range(len(g)):
            if l[i] < f[j]:
                k = "same_person" if rid[i] == rid[j] else ("diff_person_same_team" if rt[i] == rt[j] else "diff_person_diff_team")
                cnt[k][0] += 1
                if tm[i] is not None and tm[j] is not None and not (pd.isna(tm[i]) or pd.isna(tm[j])):
                    cnt[k][1] += 1
                    cnt[k][2] += int(tm[i] == tm[j])
pt = pd.DataFrame([{"pair type (annotation)": k, "pairs": v[0], "both labelled": v[1], "share labelled": v[1] / v[0], "labels equal": v[2], "equal share": v[2] / max(v[1], 1)} for k, v in cnt.items()])
print(pt.to_string())
pt.to_csv(C.OUT / "t1_team_pairs.csv", index=False)
res["team_pairs_all"] = pt.to_dict("records")

# ---------------------------------------------------------------- role
print("\n=== ROLE ===")
print("row level, confusion (rows: annotated role, cols: gbf role):")
cm = pd.crosstab(R.role_ref, R.role.fillna("none"))
print(cm.to_string())
cm.to_csv(C.OUT / "t6_role_rows_confusion.csv")
res["role_rows_accuracy"] = round(float((R.role == R.role_ref).mean()), 4)
print("row accuracy:", res["role_rows_accuracy"])
PA = P[P.ref_id.notna()].copy()
print("piece level, confusion (annotated role of the piece's person vs piece majority role):")
cmp_ = pd.crosstab(PA.role_ref, PA.role.fillna("none"))
print(cmp_.to_string())
cmp_.to_csv(C.OUT / "t6_role_pieces_confusion.csv")
res["role_pieces_accuracy"] = round(float((PA.role == PA.role_ref).mean()), 4)
key = list(zip(PA.seq, PA.piece))
PA["in_return"] = [(k in a_keys) or (k in b_keys) for k in key]
PR_ = PA[PA.in_return]
print("pieces involved in scored returns:", len(PR_))
cmr = pd.crosstab(PR_.role_ref, PR_.role.fillna("none"))
print(cmr.to_string())
cmr.to_csv(C.OUT / "t6_role_return_pieces_confusion.csv")
res["role_return_pieces"] = {"pieces": len(PR_), "accuracy": round(float((PR_.role == PR_.role_ref).mean()), 4),
                             "wrong": int((PR_.role != PR_.role_ref).sum()), "role_major_share_p10": round(float(PR_.role_major_share.quantile(0.1)), 3),
                             "mixed_role_pieces": int((PR_.role_values > 1).sum())}  # fmt: skip
print(res["role_return_pieces"])
print("wrong-role return pieces:")
print(PR_[PR_.role != PR_.role_ref][["seq", "piece", "rows", "role", "role_ref", "role_major_share", "purity", "team", "team_ref"]].to_string())
# return pairs: role of A and B
rr = {"pairs": len(RP), "A_role_ok": int((RP.a_role == RP.ref_role).sum()), "B_role_ok": int((RP.b_role == RP.ref_role).sum()),
      "A_and_B_roles_equal": int((RP.a_role == RP.b_role).sum()), "roles_differ": int((RP.a_role != RP.b_role).sum()),
      "ref_goalkeeper_returns": int((RP.ref_role == "goalkeeper").sum())}  # fmt: skip
print(rr)
res["role_return_pairs"] = rr
print(RP[RP.a_role != RP.b_role][["seq", "gap", "a_role", "b_role", "ref_role", "a_rows", "b_rows", "a_role_share", "b_role_share", "a_ident_ok", "b_ident_ok"]].to_string())
print("role by annotated role, pieces by size:")
PA["size"] = pd.cut(PA.rows, [0, 9, 49, 199, 100000], labels=["<10", "10-49", "50-199", ">=200"])
PA["role_ok"] = PA.role == PA.role_ref
print(PA.groupby(["role_ref", "size"], observed=True).role_ok.agg(["size", "mean"]).to_string())
# role-veto effect on all time-disjoint same-person pairs
same = diff = same_eq = diff_eq = 0
for seq, g in PA.groupby("seq"):
    g = g.sort_values("first").reset_index(drop=True)
    f, l, rid, ro = g["first"].to_numpy(), g["last"].to_numpy(), g.ref_id.to_numpy(), g.role.to_numpy(object)
    for i in range(len(g)):
        for j in range(len(g)):
            if l[i] < f[j]:
                if rid[i] == rid[j]:
                    same += 1
                    same_eq += int(ro[i] == ro[j])
                else:
                    diff += 1
                    diff_eq += int(ro[i] == ro[j])
res["role_pairs_all"] = {"same_person_pairs": same, "same_person_roles_equal": same_eq, "diff_person_pairs": diff, "diff_person_roles_equal": diff_eq}
print(res["role_pairs_all"])
(C.OUT / "t1_team_role.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

# ---------------------------------------------------------------- extras
print("\n=== EXTRAS ===")
# (a) returns whose two pieces both belong (by majority) to the returning person: the only ones a
#     piece-level link can get right without cuts
lk = RP[RP.a_ident_ok & RP.b_ident_ok]
both_l = lk.a_team.notna() & lk.b_team.notna()
ex = {"returns_both_pieces_are_the_person": len(lk), "both_labelled": int(both_l.sum()), "labels_equal": int((lk.a_team == lk.b_team)[both_l].sum()),
      "labels_differ": int((lk.a_team != lk.b_team)[both_l].sum()), "one_label_missing": int((lk.a_team.notna() ^ lk.b_team.notna()).sum()),
      "A_missing": int(lk.a_team.isna().sum()), "B_missing": int(lk.b_team.isna().sum()),
      "roles_equal": int((lk.a_role == lk.b_role).sum())}  # fmt: skip
print(ex)
res["team_linkable_return_pairs"] = ex
print("returns by (A piece is the person, B piece is the person):", RP.groupby(["a_ident_ok", "b_ident_ok"]).size().to_dict())
# (b) a piece without a team label: can it inherit the label of another piece of the same plugin id?
PIx = P[P.ref_id.notna() & P.role_ref.isin(["player", "goalkeeper"])]
sib = P[P.team.notna()].groupby(["seq", "plugin"]).team.agg(lambda s: s.mode().iloc[0])
miss = PIx[PIx.team.isna()]
got = [(sib.get((r.seq, r.plugin)), r.team_ref) for r in miss.itertuples()]
have = [(a, b) for a, b in got if a is not None]
ex2 = {"pieces_with_identity_without_team": len(miss), "a_sibling_piece_of_the_plugin_id_has_a_team": len(have),
       "sibling_team_right": sum(a == b for a, b in have), "sibling_team_wrong": sum(a != b for a, b in have)}  # fmt: skip
print(ex2)
res["team_inherit_from_plugin_sibling"] = ex2
(C.OUT / "t1_team_role.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
