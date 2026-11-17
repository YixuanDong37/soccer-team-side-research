"""Build the final one-row-per-return CSV (returns.csv in this folder) and the summary tables as
Markdown (-> out/tables.md, and the per-return appendix -> out/appendix.md). DIAGNOSIS ONLY (rule 5c).

Run last: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I make_tables.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ret = pd.read_csv(OUT / "returns.csv")
fn = pd.read_csv(OUT / "mutual_funnel.csv")
sw = pd.read_csv(OUT / "swap_suspects.csv")
und = pd.read_csv(OUT / "undetected.csv")
df = ret.merge(fn, on=["clip", "ret"], how="left")
flag = set(zip(sw["clip"], sw.ret_i)) | set(zip(sw["clip"], sw.ret_j))
df["label_suspect_exchange"] = [(c, r) in flag for c, r in zip(df["clip"], df["ret"])]
df["appearance_contradicts_annotation"] = df.true_minus_best_impostor <= -0.05
und["why"] = und.missing_side + ": " + und.reason
u = und.groupby(["clip", "ret"]).why.agg(" | ".join).rename("undetected_why").reset_index()
df = df.merge(u, on=["clip", "ret"], how="left")
df.to_csv(HERE / "returns.csv", index=False)

L = []


def P(*a):
    L.append(" ".join(str(x) for x in a))


def md(t, index=True):
    t = t.copy()
    if index:
        t = t.reset_index()
    cols = [str(c) if not isinstance(c, tuple) else " / ".join(str(x) for x in c) for c in t.columns]
    P("| " + " | ".join(cols) + " |")
    P("|" + "|".join("---" for _ in cols) + "|")
    for row in t.itertuples(index=False):
        P("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:g}" if isinstance(v, float) else str(v)) for v in row) + " |")
    P("")


sc = df[df.status == "scored"].copy()
orc = json.loads((OUT / "oracles.json").read_text())
probe = json.loads((OUT / "probe_hole_cuts.json").read_text())
probe_or = json.loads((OUT / "probe_hole_cuts_oracle.json").read_text())

P("## T1 pooled scores")
rows = []
names = {"split": "split = gbf now (no links)", "plugin": "plugin (keep every plugin join)", "mutual": "mutual (appearance, margin 0.05)",
         "O5_candidate_pairs_only": "ORACLE: perfect choice among appearance_match candidate pairs",
         "O1_links_known": "ORACLE: link all pieces of known identity (>= 5 matched rows)",
         "O2_links_any": "ORACLE: ... plus pieces with 1-4 matched rows",
         "O3_cuts_links_known": "ORACLE: cut switches (runs >= 5 rows) + link known pieces",
         "O4_cuts_links_any": "ORACLE: cut switches (runs >= 5 rows) + link every piece",
         "O6_cuts1_links_any": "ORACLE: cut at every change of person + link every piece"}  # fmt: skip
for k, n in names.items():
    v = orc[k]
    rows.append({"what": n, "right": v["right"], "wrong": v["wrong"], "lost": v["lost"], "links": v["links"],
                 "links right/wrong/unknown": f"{v['links_right']}/{v['links_wrong']}/{v['links_unknown']}", "impure rows %": round(100 * v["impure_row_share"], 2)})  # fmt: skip
for k, v in probe.items():
    rows.append({"what": f"PROBE (label-free): cut every piece at every hole >= {k[2:]} frames, no links ({v['cuts']} cuts)", "right": v["right"], "wrong": v["wrong"],
                 "lost": v["lost"], "links": 0, "links right/wrong/unknown": "0/0/0", "impure rows %": round(100 * v["impure_row_share"], 2)})  # fmt: skip
for k, v in probe_or.items():
    rows.append({"what": f"ORACLE on the H=10 hole cuts: {k}", "right": v["right"], "wrong": v["wrong"], "lost": v["lost"], "links": v["links"],
                 "links right/wrong/unknown": f"{v['links_right']}/{v['links_wrong']}/{v['links_unknown']}", "impure rows %": round(100 * v["impure_row_share"], 2)})  # fmt: skip
md(pd.DataFrame(rows), index=False)

P("## T2 per clip")
g = df.groupby("clip", sort=False)


def cnt(col, val):
    return g[col].agg(lambda s: int((s == val).sum()))


t = pd.DataFrame({"returns": g.size(), "scored": cnt("status", "scored"), "undetected": g.status.agg(lambda s: int((s != "scored").sum())),
                  "plugin R": cnt("out_plugin", "right"), "plugin W": cnt("out_plugin", "wrong"), "mutual R": cnt("out_mutual", "right"), "mutual W": cnt("out_mutual", "wrong"),
                  "split W": cnt("out_split", "wrong"), "oracle known R": cnt("out_O1_links_known", "right"), "oracle +fragments R": cnt("out_O2_links_any", "right"),
                  "B row in fragment": g.b_kind.agg(lambda s: int((s == "fragment").sum())), "B row in foreign piece": g.b_kind.agg(lambda s: int((s == "foreign").sum())),
                  "label suspects": g.label_suspect_exchange.sum().astype(int)})  # fmt: skip
t.loc["all"] = t.sum()
md(t)

P("## T3 ceiling classes (167 returns)")
md(df.ceiling_class.value_counts().rename("returns").to_frame())
P("## T3b ceiling class x outcome now")
md(pd.crosstab(sc.ceiling_class, sc.out_plugin).add_prefix("plugin ").join(pd.crosstab(sc.ceiling_class, sc.out_mutual).add_prefix("mutual ")))
P("## T4 piece under row A x piece under row B (137 scored)")
md(pd.crosstab(sc.a_kind.rename("row A in"), sc.b_kind.rename("row B in"), margins=True))
P("## T4b same plugin id on both sides")
md(pd.crosstab(sc.same_plugin.astype(bool).rename("A and B share the plugin id"), sc.out_plugin, margins=True))

P("## T5 by gap length (all 167, annotated gap)")
df["gap_bucket"] = pd.cut(df.gap_s, [0, 1, 2, 5, 10, 100], labels=["<= 1 s", "1-2 s", "2-5 s", "5-10 s", "> 10 s"])
rows = []
for name, d in list(df.groupby("gap_bucket", observed=True)) + [("all", df)]:
    rows.append({"gap": name, "returns": len(d), "undetected": int((d.status != "scored").sum()), "scored": int((d.status == "scored").sum()),
                 "plugin R": int((d.out_plugin == "right").sum()), "plugin W": int((d.out_plugin == "wrong").sum()),
                 "mutual R": int((d.out_mutual == "right").sum()), "mutual W": int((d.out_mutual == "wrong").sum()),
                 "oracle known R": int((d.out_O1_links_known == "right").sum()),
                 "row pair is a candidate": int((d.is_candidate == True).sum()),  # noqa: E712
                 "own pieces on both sides": int((d.anchor_a.notna() & d.anchor_b.notna()).sum()),
                 "true pair beats every other person": int((d.true_minus_best_impostor > 0).sum()),
                 "... by >= 0.05": int((d.true_minus_best_impostor >= 0.05).sum())})  # fmt: skip
md(pd.DataFrame(rows), index=False)

P("## T6 image side of exit and re-entry (annotation, 167)")
md(pd.crosstab(df.exit_side.rename("exit side"), df.reentry_side.rename("re-entry side"), margins=True))
P(f"same side: {(df.exit_side == df.reentry_side).sum()} of {len(df)}")
P("")
P("annotated implied speed (m/s) quantiles 10/25/50/75/90/95/max:", df.ann_speed_mps.quantile([.1, .25, .5, .75, .9, .95, 1]).round(2).tolist())
P("annotated distance exit->re-entry (m) quantiles 10/25/50/75/90/95/max:", df.ann_dist_m.quantile([.1, .25, .5, .75, .9, .95, 1]).round(1).tolist())
a = df[df.anchor_a.notna() & df.anchor_b.notna()]
P("system (own pieces) implied speed quantiles 10/25/50/75/90/95/max:", a.anchors_sys_speed_mps.quantile([.1, .25, .5, .75, .9, .95, 1]).round(2).tolist(), "n", int(a.anchors_sys_speed_mps.notna().sum()))
P("own-piece pairs within appearance_match's reach (12 m/s + 4 m):", a.anchors_reach_ok.value_counts(dropna=False).to_dict())
P("")

P("## T7 mutual rule funnel (137 scored)")
md(pd.crosstab(sc.funnel, sc.out_mutual, margins=True))
gg = sc[sc.funnel == "candidate generation"]
P("## T8 candidate generation losses: every gate failing for the nearest earlier own piece (55 returns)")
md(gg.fn_gate_nearest_own.value_counts().rename("returns").to_frame())
t = pd.Series({"gap > 10 s (alone or with others)": int(gg.fn_gate_nearest_own.str.contains("gap > 10").sum()),
               "team unverified on A or B": int(gg.fn_gate_nearest_own.str.contains("team unverified").sum()),
               "fewer than 5 clean rows on A or B": int(gg.fn_gate_nearest_own.str.contains("clean rows").sum()),
               "overlap in time": int(gg.fn_gate_nearest_own.str.contains("overlap").sum()),
               "out of reach": int(gg.fn_gate_nearest_own.str.contains("reach").sum()),
               "only one gate fails": int((gg.fn_gates_n == 1).sum())}).rename("returns").to_frame()  # fmt: skip
md(t)
P("## T8b the same gates on the 155 pairs of own pieces (scored and undetected returns)")
why = a.anchors_not_candidate_why.fillna("")
t = pd.Series({"pairs": len(a), "candidate": int((a.anchors_candidate == True).sum()),  # noqa: E712
               "gap > 10 s": int(why.str.contains("gap > 10").sum()), "team unverified": int(why.str.contains("team unverified").sum()),
               "clean rows < 5": int(why.str.contains("clean rows").sum()), "overlap in time": int(why.str.contains("overlap").sum()),
               "out of reach": int(why.str.contains("reach").sum())}).rename("pairs").to_frame()  # fmt: skip
md(t)
md(a.anchors_mutual_status.value_counts().rename("pairs").to_frame())
P("path G on the same 155 pairs:")
md(a.anchors_pathG.value_counts().rename("pairs").to_frame())
P("descriptor kinds (earlier/later):", a.anchors_desc_kinds.value_counts().to_dict())
P("cos of the true pair, quantiles 5/25/50/75/95:", a.anchors_cos_any.quantile([.05, .25, .5, .75, .95]).round(3).tolist())
P(f"true pair beats the best other-person piece on both sides: {(a.true_minus_best_impostor > 0).sum()} of {len(a)}")
for th in (0.02, 0.03, 0.05, 0.10):
    P(f"... by >= {th}: {(a.true_minus_best_impostor >= th).sum()} of {len(a)}")
b = a[a["clip"] != "SNGS-162"]
P(f"without SNGS-162: beats by > 0: {(b.true_minus_best_impostor > 0).sum()} of {len(b)}; by >= 0.05: {(b.true_minus_best_impostor >= 0.05).sum()}")
teams = a.anchors_teams.fillna("")
P("plugin team labels of the two own pieces differ:", int(teams.apply(lambda s: len(set(s.split("/"))) > 1 and "None" not in s and "nan" not in s).sum()), "; a label is missing:", int(teams.str.contains("None|nan").sum()))
P("sides (system boxes): own piece A ends at an edge", int((a.anchor_a_end_side != "none").sum()), "; own piece B starts at an edge", int((a.anchor_b_start_side != "none").sum()),
  "; both and same side", int(((a.anchor_a_end_side == a.anchor_b_start_side) & (a.anchor_a_end_side != "none")).sum()), "of", len(a))
P("lead of own piece A before the exit (frames) quantiles 50/75/90/max:", a.anchor_a_lead_frames.quantile([.5, .75, .9, 1]).tolist(),
  "; delay of own piece B after re-entry:", a.anchor_b_delay_frames.quantile([.5, .75, .9, 1]).tolist())
P("")

P("## T9 decision losses: who is the rival (25 returns with a true candidate pair, not linked)")
d = sc[sc.funnel.fillna("").str.startswith("decision") & (sc.funnel != "decision: linked")]
md(pd.crosstab(d.funnel, [d.fn_rival_b_is.fillna("no rival").rename("B side"), d.fn_rival_a_is.fillna("no rival").rename("A side")]))

P("## T10 row pair (piece A, piece B) of the 137 scored returns under the mutual rule")
md(pd.crosstab(sc.mutual_status, sc.out_mutual, margins=True))
c51 = sc[sc.is_candidate == True]  # noqa: E712
P("candidate row pairs:", len(c51), "; true partner rank 1 for A:", int((c51.rank_of_b_for_a == 1).sum()), "; rank 1 for B:", int((c51.rank_of_a_for_b == 1).sum()),
  "; rank 1 on both:", int(((c51.rank_of_b_for_a == 1) & (c51.rank_of_a_for_b == 1)).sum()))
P("cos of candidate row pairs quantiles 5/25/50/75/95:", c51.cos_strict.quantile([.05, .25, .5, .75, .95]).round(3).tolist())
P("")
P("path G on the 137 row pairs:")
md(sc.pathG.value_counts().rename("returns").to_frame())

ml = pd.read_csv(OUT / "mutual_links.csv")
P("## T11 mutual links that are not right")
md(ml[ml.label != "right"][["clip", "earlier", "later", "label", "cos", "min_margin", "gap_s", "a_clean", "b_clean", "a_person", "b_person", "a_purity", "b_purity", "a_true_partner", "b_true_partner"]], index=False)
P("right links: same plugin id on both sides:", ml[ml.label == "right"].same_plugin.value_counts().to_dict())
for m in (0.05, 0.06, 0.08, 0.10, 0.15):
    P(f"min margin >= {m}:", ml[ml.min_margin >= m].label.value_counts().to_dict())
for th in (0.90, 0.93, 0.95):
    P(f"cos >= {th}:", ml[ml.cos >= th].label.value_counts().to_dict())
P("")

pj = pd.read_csv(OUT / "plugin_joins.csv")
P("## T12 plugin joins (93)")
md(pd.crosstab(pj.label, pj.soft_label, margins=True))


def q(s):
    return "/".join(f"{v:g}" for v in s.quantile([0, .25, .5, .75, 1]).round(3))


t = pj.groupby("label").agg(joins=("label", "size"), gap_s=("gap_s", q), rows_earlier=("a_rows", q), rows_later=("b_rows", q), cos_any=("cos_any", q),
                            other_pieces_scoring_higher=("pieces_scoring_higher", q), speed_mps=("speed_mps", q), height_ratio=("h_ratio", q),
                            same_edge_side=("same_side", "sum"), team_label_equal=("team_equal", "sum"), within_reach=("reach_ok", "sum"))  # fmt: skip
t["shortest_piece_le_4_rows"] = pj.groupby("label").apply(lambda d: int((d[["a_rows", "b_rows"]].min(axis=1) <= 4).sum()), include_groups=False)
md(t.T)
P("wrong joins:")
md(pj[pj.label == "wrong"][["clip", "earlier", "later", "gap_s", "a_rows", "b_rows", "a_clean", "b_clean", "a_person", "b_person", "cos_any", "desc_kinds", "pieces_scoring_higher", "pool", "a_end_side", "speed_mps"]], index=False)
k = pj[pj.label != "unknown"]
for th in (0.90, 0.92, 0.95):
    P(f"cos_any >= {th}:", k[k.cos_any >= th].label.value_counts().to_dict(), "| unknown joins kept:", pj[(pj.label == "unknown") & (pj.cos_any >= th)].soft_label.value_counts().to_dict())
P("no other piece scores higher:", k[k.pieces_scoring_higher == 0].label.value_counts().to_dict(), "; <= 1 higher:", k[k.pieces_scoring_higher <= 1].label.value_counts().to_dict())
P("cos_any >= 0.90 and <= 1 other piece higher:", k[(k.cos_any >= 0.9) & (k.pieces_scoring_higher <= 1)].label.value_counts().to_dict(),
  "| unknown joins kept:", pj[(pj.label == "unknown") & (pj.cos_any >= 0.9) & (pj.pieces_scoring_higher <= 1)].soft_label.value_counts().to_dict())
P("")

P("## T13 undetected: one row per missing side (34 sides of 30 returns)")
md(pd.crosstab(und.reason, und.missing_side, margins=True))
P("box height at the gap < 40 px (720p):", int((und.ann_box_h_720p < 40).sum()), "of", len(und),
  "; nearest matched row of the person, frames from the gap, quantiles 25/50/75/max:", und.nearest_matched_row_frames_from_gap.quantile([.25, .5, .75, 1]).tolist())
uu = df[df.status != "scored"]
P("undetected returns with an own piece on both sides:", int((uu.anchor_a.notna() & uu.anchor_b.notna()).sum()), "of", len(uu),
  "; of those, true pair beats every other-person piece:", int((uu.true_minus_best_impostor > 0).sum()),
  "; is an appearance_match candidate:", int((uu.anchors_candidate == True).sum()))  # noqa: E712
P("")
md(und[["clip", "ret", "ref_id", "missing_side", "gap_s", "side", "stretch_frames", "best_iou_in_window", "frames_fitting_box_taken_by_other", "ann_box_h_720p", "ann_box_w_720p",
        "nearest_matched_row_frames_from_gap", "own_piece_frames_from_gap", "reason"]], index=False)  # fmt: skip

P("## T14 holes >= 10 frames inside gbf pieces (not split: no box at the edge next to the hole)")
ho = pd.read_csv(OUT / "holes.csv")
md(pd.crosstab(pd.cut(ho.hole_frames, [9, 14, 24, 49, 99, 1000], labels=["10-14", "15-24", "25-49", "50-99", ">= 100"]).rename("hole frames"), ho.label, margins=True))
kk = ho[ho.label != "unknown (no matched row on one side)"]
t = pd.DataFrame({"cos < 0.90": kk[kk.cos_before_after < 0.9].label.value_counts(), "cos >= 0.90": kk[kk.cos_before_after >= 0.9].label.value_counts(),
                  "no cosine (a side has < 5 clean rows)": kk[kk.cos_before_after.isna()].label.value_counts()}).fillna(0).astype(int)  # fmt: skip
md(t)
mx = pd.read_csv(OUT / "mixed_pieces.csv")
P(f"switch events with >= 5 matched rows on both sides: {len(mx)} in {mx.groupby(['clip', 'piece']).ngroups} pieces; a hole >= 10 frames lies between the two runs in "
  f"{(mx.largest_hole_frames >= 10).sum()}; segment cosine known in {mx.cos_segments.notna().sum()}, below 0.90 in {(mx.cos_segments < 0.9).sum()}")
P("")
swr = pd.read_csv(OUT / "switch_returns.csv")
P("returns that need a cut:")
md(swr[["clip", "ret", "ref_id", "piece_b", "b_kind", "piece_rows", "majority_person", "frame_b", "person_run_rows_from_b", "frames_missing_before_b", "centre_jump_px",
        "implied_speed_mps", "gbf_edge_touch_prev_or_b", "out_split", "out_plugin"]], index=False)  # fmt: skip

P("## T15 label suspects")
md(sw[["clip", "person_i", "jersey_i", "person_j", "jersey_j", "team", "before_i", "after_i", "before_j", "after_j", "cos_annotated_i", "cos_annotated_j", "cos_exchanged_i_to_j",
       "cos_exchanged_j_to_i", "clean_rows", "exit_i", "reentry_i", "exit_j", "reentry_j"]], index=False)  # fmt: skip
ids = pd.read_csv(OUT / "id_split_suspects.csv")
md(ids[(ids.cos >= 0.9) & (ids.margin >= 0.03)][["clip", "id_first", "id_second", "team", "jersey_first", "jersey_second", "first_ends", "second_starts", "gap_s", "end_side",
                                                 "start_side", "dist_m", "piece_first", "piece_second", "cos", "best_other_later", "best_other_earlier", "margin"]], index=False)  # fmt: skip
neg = df[df.appearance_contradicts_annotation]
P("returns whose own pieces look less alike than one of them and another person's piece, by >= 0.05:", len(neg), "by clip:", neg["clip"].value_counts().to_dict())
P("")

P("## T16 ranking ceiling")
rc = json.loads((OUT / "rank_ceiling.json").read_text())
rows = []
for name, v in rc.items():
    rows.append({"earlier candidates": name, "returning B": v.get("returning: pieces B"), "top-1 is the person": v.get("returning: top-1 is the person", 0),
                 "top-1 another person": v.get("returning: top-1 is another person", 0), "top-1 a fragment": v.get("returning: top-1 is a fragment (unknown)", 0),
                 "no candidate": v.get("returning: no candidate", 0), "true piece removed": v.get("returning: true piece removed by the restriction", 0),
                 "new B": v.get("new: pieces B"), "new B with a top-1": v.get("new: would link (top-1 exists)", 0),
                 "cos>=0.93 & margin>=0.05: right/wrong/new": "/".join(str(v["cos>=0.93, margin>=0.05"][x]) for x in ("right", "wrong", "new")),
                 "cos>=0.95 & margin>=0.05": "/".join(str(v["cos>=0.95, margin>=0.05"][x]) for x in ("right", "wrong", "new"))})  # fmt: skip
md(pd.DataFrame(rows), index=False)

fr = pd.read_csv(OUT / "fragments.csv")
P("## T17 fragments and foreign pieces under row A / row B")
md(pd.crosstab([fr.side, fr.kind], fr.ceiling_class))
f2 = fr[fr.kind == "fragment"]
P(f"fragments: {len(f2)}; <= 4 rows: {(f2.rows <= 4).sum()}; 0 clean rows: {(f2.clean == 0).sum()}; start or end at the image edge: {(f2.starts_at_edge | f2.ends_at_edge).sum()}; "
  f"a gbf re-entry piece (k > 0): {(f2.k > 0).sum()}; person has an own piece on that side: {f2.own_piece.notna().sum()}; within 5 frames of it: {(f2.frames_between.abs() <= 5).sum()}; "
  f"within 12: {(f2.frames_between.abs() <= 12).sum()}")
fb = f2[f2.side == "B"]
P(f"B-side fragments: {len(fb)}; own piece follows: {fb.own_piece.notna().sum()}; within 5 frames: {(fb.frames_between.abs() <= 5).sum()}; box IoU across >= 0.3: "
  f"{(fb.box_iou_across >= 0.3).sum()}; centre shift <= 60 px: {(fb.centre_shift_px <= 60).sum()}; cos to own piece quantiles 25/50/75: {fb.cos_any_to_own.quantile([.25, .5, .75]).round(3).tolist()}")
(OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

# ----------------------------------------------------------------------------- appendix tables
A = []


def md2(t):
    cols = [str(c) for c in t.columns]
    A.append("| " + " | ".join(cols) + " |")
    A.append("|" + "|".join("---" for _ in cols) + "|")
    for row in t.itertuples(index=False):
        A.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:g}" if isinstance(v, float) else str(v)) for v in row) + " |")
    A.append("")


def short(r):
    def piece(tag):
        p = r[f"piece_{tag}"]
        if pd.isna(p):
            return "-"
        return f"{int(p)} ({r[f'{tag}_kind']}, {int(r[f'{tag}_rows'])} rows, {int(r[f'{tag}_clean'])} clean, {r[f'{tag}_team']}/{r[f'{tag}_role']})"
    if r.status == "scored":
        cand = "yes" if r.is_candidate is True or r.is_candidate == "True" else (r.not_candidate_why or "")
        rank = "" if pd.isna(r.rank_of_b_for_a) else f"{int(r.rank_of_b_for_a)}/{int(r.n_later_of_a)}, {int(r.rank_of_a_for_b)}/{int(r.n_earlier_of_b)}"
    else:
        cand, rank = "", ""
    return {"clip": r["clip"][5:], "ret": r.ret, "person": f"{r.ref_id} ({r.ann_team}, #{'' if pd.isna(r.ann_jersey) else int(r.ann_jersey)})", "gap s": r.gap_s,
            "sides": f"{r.exit_side}>{r.reentry_side}", "piece A": piece("a"), "piece B": piece("b"),
            "plugin/split/mutual": "/".join(str(r[f"out_{k}"])[0].upper() for k in ("plugin", "split", "mutual")) if r.status == "scored" else "undetected",
            "candidate pair": cand, "cos": "" if pd.isna(r.cos_any) else r.cos_any, "rank of B for A, of A for B": rank,
            "ann. m, m/s": f"{r.ann_dist_m}, {r.ann_speed_mps}", "own pieces": "" if pd.isna(r.anchor_a) or pd.isna(r.anchor_b) else f"{int(r.anchor_a)}>{int(r.anchor_b)} cos {r.anchors_cos_any}, margin {r.true_minus_best_impostor}",
            "class": {"linkable: existing pieces, known identity": "link", "linkable only through a piece with < 5 matched rows": "fragment",
                      "needs a cut around a stray run of < 5 rows": "stray cut", "needs an in-track switch fix (cut of a run >= 5 rows)": "switch cut",
                      "undetected (detection)": "undetected"}[r.ceiling_class]}  # fmt: skip


A.append("## A1 every return (R right, W wrong, L lost; margin = own-piece cosine minus the best cosine to another person's piece)")
md2(pd.DataFrame([short(r) for _, r in df.iterrows()]))
A.append("## A2 returns that are wrong under the plugin")
pw = df[df.out_plugin == "wrong"]
md2(pw[["clip", "ret", "ref_id", "gap_s", "piece_a", "a_kind", "piece_b", "b_kind", "b_rows", "b_ident", "b_person_rows", "b_prior_other_rows", "out_split", "out_mutual", "ceiling_class",
        "label_suspect_exchange", "appearance_contradicts_annotation"]])  # fmt: skip
(OUT / "appendix.md").write_text("\n".join(A) + "\n", encoding="utf-8")
