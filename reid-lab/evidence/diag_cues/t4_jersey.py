"""(4) Jersey: tracks.jersey / jersey_conf against the annotated jersey numbers (diagnosis).

tracks.jersey in refine/tracks.parquet is what gbf kept after gbf.track_tools.confident_jerseys;
the plugin's raw per-row reads are not among the lab's files and are not looked at here.
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


def norm(v):
    """Jersey as a canonical string ('7', not '07' or 7.0); None when missing."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s or None


rows_all, pieces_all = [], []
for seq in C.CLIPS:
    t = data.tracks(seq)
    m = C.matched(seq).copy()
    for c in ("jersey", "jersey_conf", "team", "role", "y1", "y2"):
        m[c] = t[c].to_numpy()[m.trow.to_numpy()]
    m["seq"] = seq
    rows_all.append(m)
    p = C.piece_ident(seq).copy()
    p["seq"] = seq
    tt = t[t.track_id >= 0]
    g = tt.groupby("track_id")
    p["jersey_rows"] = p.piece.map(g.jersey.apply(lambda s: int(s.notna().sum())))
    p["jersey_values"] = p.piece.map(g.jersey.apply(lambda s: s.dropna().map(norm).nunique()))
    p["jersey"] = p.piece.map(g.jersey.apply(lambda s: norm(s.dropna().map(norm).mode().iloc[0]) if s.notna().any() else None))
    p["jersey_conf_mean"] = p.piece.map(g.jersey_conf.mean())
    p["jersey_conf_rows"] = p.piece.map(g.jersey_conf.apply(lambda s: int(s.notna().sum())))
    pieces_all.append(p)
R = pd.concat(rows_all, ignore_index=True)
P = pd.concat(pieces_all, ignore_index=True)
R["j"] = R.jersey.map(norm)
R["j_ref"] = R.jersey_ref.map(norm)
P["j_ref"] = P.jersey_ref.map(norm)

print("=== JERSEY, row level ===")
pl = R[R.role_ref.isin(["player", "goalkeeper"])]
print("matched rows of annotated players/GK:", len(pl), "; annotation has a number on", round(pl.j_ref.notna().mean(), 4))
print("rows with tracks.jersey:", int(pl.j.notna().sum()), f"({pl.j.notna().mean():.4f})", "; rows with jersey_conf:", int(pl.jersey_conf.notna().sum()))
print("jersey without conf:", int((pl.j.notna() & pl.jersey_conf.isna()).sum()), "; conf without jersey:", int((pl.j.isna() & pl.jersey_conf.notna()).sum()))
have = pl[pl.j.notna()]
known = have[have.j_ref.notna()]
res["rows"] = {"matched_player_rows": len(pl), "annotation_has_number_share": round(float(pl.j_ref.notna().mean()), 4),
               "rows_with_jersey": int(pl.j.notna().sum()), "coverage": round(float(pl.j.notna().mean()), 4),
               "rows_with_jersey_and_annotated_number": len(known), "accuracy": round(float((known.j == known.j_ref).mean()), 4) if len(known) else None,
               "rows_with_jersey_where_annotation_has_none": int(have.j_ref.isna().sum())}  # fmt: skip
print(res["rows"])
print("by clip:")
pc = pl.groupby("seq").apply(lambda g: pd.Series({"rows": len(g), "ann_has_number": g.j_ref.notna().mean(), "coverage": g.j.notna().mean(),
                                                   "judged": int((g.j.notna() & g.j_ref.notna()).sum()),
                                                   "accuracy": (g.j == g.j_ref)[g.j.notna() & g.j_ref.notna()].mean()}), include_groups=False)  # fmt: skip
print(pc.to_string())
pc.reset_index().to_csv(C.OUT / "t4_jersey_rows_per_clip.csv", index=False)
print("by jersey_conf (rows with a jersey and an annotated number):")
kb = known.assign(cbin=pd.cut(known.jersey_conf, [-0.001, 0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]))
cb = kb.groupby("cbin", observed=True).apply(lambda g: pd.Series({"rows": len(g), "accuracy": (g.j == g.j_ref).mean()}), include_groups=False)
print(cb.to_string())
cb.reset_index().astype({"cbin": str}).to_csv(C.OUT / "t4_jersey_rows_by_conf.csv", index=False)
res["rows_by_conf"] = {str(k): {"rows": int(v.rows), "accuracy": round(float(v.accuracy), 4)} for k, v in cb.iterrows()}
print("rows with jersey_conf but no jersey kept (conf distribution):", C.qtable(pl.jersey_conf[pl.j.isna() & pl.jersey_conf.notna()]))
print("gbf role of rows with a jersey:", have.role.value_counts().to_dict())

print("\n=== JERSEY, piece level ===")
PI = P[P.ref_id.notna() & P.role_ref.isin(["player", "goalkeeper"])].copy()
hj = PI[PI.jersey.notna()]
kj = hj[hj.j_ref.notna()]
res["pieces"] = {"pieces_with_identity": len(PI), "annotated_person_has_number": int(PI.j_ref.notna().sum()), "pieces_with_jersey": len(hj),
                 "coverage": round(len(hj) / len(PI), 4), "pieces_with_several_numbers": int((PI.jersey_values > 1).sum()),
                 "judged": len(kj), "right": int((kj.jersey == kj.j_ref).sum()), "wrong": int((kj.jersey != kj.j_ref).sum()),
                 "jersey_on_person_without_annotated_number": int(hj.j_ref.isna().sum())}  # fmt: skip
print(res["pieces"])
print("pieces with a jersey, by clip:", hj.groupby("seq").size().to_dict())
print("share of the piece's rows carrying the jersey:", C.qtable(hj.jersey_rows / hj.rows))
print("wrong pieces:")
print(kj[kj.jersey != kj.j_ref][["seq", "piece", "rows", "jersey_rows", "jersey", "j_ref", "jersey_conf_mean", "purity", "team", "team_ref"]].to_string())
kj2 = kj.assign(cbin=pd.cut(kj.jersey_conf_mean, [-0.001, 0.5, 0.7, 0.85, 0.95, 1.0]), ok=kj.jersey == kj.j_ref)
print("piece accuracy by mean jersey_conf:")
print(kj2.groupby("cbin", observed=True).ok.agg(["size", "mean"]).to_string())
kj3 = kj.assign(nbin=pd.cut(kj.jersey_rows, [0, 4, 9, 24, 99, 10000]), ok=kj.jersey == kj.j_ref)
print("piece accuracy by number of rows with the jersey:")
print(kj3.groupby("nbin", observed=True).ok.agg(["size", "mean"]).to_string())
# persons: how many annotated persons get a (right) number on some piece
persons = PI.groupby(["seq", "ref_id"]).apply(lambda g: pd.Series({"pieces": len(g), "pieces_with_jersey": int(g.jersey.notna().sum()),
                                                                    "any_right": bool(((g.jersey == g.j_ref) & g.jersey.notna()).any())}), include_groups=False)  # fmt: skip
res["persons"] = {"persons_with_pieces": len(persons), "persons_with_a_jersey_piece": int((persons.pieces_with_jersey > 0).sum()),
                  "persons_with_two_or_more_jersey_pieces": int((persons.pieces_with_jersey >= 2).sum())}  # fmt: skip
print(res["persons"])

print("\n=== JERSEY, pair level ===")
RET = pd.concat([C.returns(s) for s in C.CLIPS], ignore_index=True)
RS = RET[RET.scored]
pk = {(r.seq, r.piece): r for r in P.itertuples()}
n_both = n_eq = n_a = n_b = 0
for r in RS.itertuples():
    a, b = pk[(r.seq, r.piece_a)], pk[(r.seq, r.piece_b)]
    n_a += pd.notna(a.jersey)
    n_b += pd.notna(b.jersey)
    if pd.notna(a.jersey) and pd.notna(b.jersey):
        n_both += 1
        n_eq += a.jersey == b.jersey
res["return_pairs"] = {"scored_returns": len(RS), "A_piece_has_jersey": int(n_a), "B_piece_has_jersey": int(n_b), "both": int(n_both), "both_equal": int(n_eq)}
print(res["return_pairs"])
# all time-disjoint pairs of pieces that both carry a jersey (any identity status)
cnt = {"same_person": [0, 0], "diff_person": [0, 0], "unknown": [0, 0]}
sameteam = {"same_person": [0, 0], "diff_person": [0, 0], "unknown": [0, 0]}
PJ = P[P.jersey.notna()]
for seq, g in PJ.groupby("seq"):
    g = g.sort_values("first").reset_index(drop=True)
    for i in range(len(g)):
        for j in range(len(g)):
            a, b = g.iloc[i], g.iloc[j]
            if a["last"] < b["first"]:
                k = "unknown" if (pd.isna(a.ref_id) or pd.isna(b.ref_id)) else ("same_person" if a.ref_id == b.ref_id else "diff_person")
                cnt[k][0] += 1
                cnt[k][1] += int(a.jersey == b.jersey)
                if pd.notna(a.team) and pd.notna(b.team) and a.team == b.team:
                    sameteam[k][0] += 1
                    sameteam[k][1] += int(a.jersey == b.jersey)
res["pairs_both_with_jersey"] = {k: {"pairs": v[0], "equal_number": v[1]} for k, v in cnt.items()}
res["pairs_both_with_jersey_same_team_label"] = {k: {"pairs": v[0], "equal_number": v[1]} for k, v in sameteam.items()}
print("time-disjoint pairs, both with a jersey:", res["pairs_both_with_jersey"])
print("... and the same gbf team label:", res["pairs_both_with_jersey_same_team_label"])
print("pieces with a jersey in total (any identity):", len(PJ), "; by clip:", PJ.groupby("seq").size().to_dict())
(C.OUT / "t4_jersey.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
