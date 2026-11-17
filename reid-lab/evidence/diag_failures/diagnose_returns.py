"""One row per annotated return (167) of the 13 dev clips -> out/returns.csv, plus the oracle
ceilings -> out/oracles.json and sanity checks. DIAGNOSIS ONLY (labels used freely, rule 5c):
nothing here is a method and no number here is a method result.

Run (after pathg_audit.py):
  C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I diagnose_returns.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script's folder
from common import FPS, MIN_MATCHED, cos, ctx, data, mutual_status, pair_gate, reach_ok, score, sides_1080  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
GAP_MIN = score.GAP_MIN


# ----------------------------------------------------------------------------- oracles
def oracle_links(c, min_rows: int):
    """Chain, per annotated person, the pieces whose majority person he is (>= min_rows matched rows)."""
    by = {}
    for p in c.P.values():
        if p["ident_any"] is not None and p["ident_n"] >= min_rows:
            by.setdefault(p["ident_any"], []).append(p)
    links = []
    for ps in by.values():
        ps = sorted(ps, key=lambda p: p["first"])
        links += [[ps[i]["piece"], ps[i + 1]["piece"]] for i in range(len(ps) - 1)]
    return links


def oracle_cuts(c, min_run: int = MIN_MATCHED):
    """Cut a piece where its matched person changes for a run of >= min_run rows."""
    cuts = []
    fr, rf, pc = c.clip.frames, c.clip.refs, c.clip.piece
    for pid in np.unique(pc[pc >= 0]):
        m = pc == pid
        order = np.argsort(fr[m], kind="stable")
        f, r = fr[m][order], rf[m][order]
        runs = []  # [ref, start_frame, n]
        for i in range(len(r)):
            if runs and runs[-1][0] == r[i]:
                runs[-1][2] += 1
            else:
                runs.append([int(r[i]), int(f[i]), 1])
        owner = None
        for ref, start, n in runs:
            if n < min_run:
                continue
            if owner is None:
                owner = ref
            elif ref != owner:
                cuts.append([int(pid), int(start)])
                owner = ref
    return cuts


def oracle_cut_links(c, cuts, min_rows: int):
    ids = score._ids_after_cuts(c.clip, cuts)
    by = {}
    for tid in np.unique(ids[ids >= 0]):
        m = ids == tid
        top, n = Counter(c.clip.refs[m].tolist()).most_common(1)[0]
        if n >= min_rows:
            by.setdefault(int(top), []).append((int(c.clip.frames[m].min()), int(tid)))
    links = []
    for ps in by.values():
        ps = sorted(ps)
        links += [[ps[i][1], ps[i + 1][1]] for i in range(len(ps) - 1)]
    return links


def kind(p, rid):
    """own = the piece is mainly this person (>= 5 matched rows); foreign = mainly another person;
    fragment = fewer than 5 matched rows of anyone (identity unknown to the scorer)."""
    if p["ident"] == rid:
        return "own"
    return "foreign" if p["ident"] is not None else "fragment"


def impostor_best(c, anchor, rid, later: bool, exclude=()):
    """Best cosine of `anchor` to any piece of ANOTHER person (or of nobody annotated) that could be
    confused with the true partner: no time overlap on the right side, same plugin team label or
    none, not a referee, any descriptor (strict where >= 5 clean rows, else all embedded rows)."""
    best, who = None, None
    for q in c.P.values():
        if q["piece"] == anchor["piece"] or q["piece"] in exclude or q["desc_any"] is None or q["role"] == "referee":
            continue
        if q["ident_any"] == rid:
            continue
        if anchor["team"] is not None and q["team"] is not None and q["team"] != anchor["team"]:
            continue
        if (q["first"] > anchor["last"]) if later else (q["last"] < anchor["first"]):
            s = float(anchor["desc_any"] @ q["desc_any"])
            if best is None or s > best:
                best, who = s, q["piece"]
    return best, who


# ----------------------------------------------------------------------------- main
def main():
    pathg = json.loads((OUT / "pathg_pairs.json").read_text(encoding="utf-8"))
    base = {n: score.baseline_links(n) for n in ("split", "plugin", "mutual")}
    methods = {n: {} for n in ("O1_links_known", "O2_links_any", "O3_cuts_links_known", "O4_cuts_links_any",
                               "O5_candidate_pairs_only", "O6_cuts1_links_any")}  # fmt: skip
    rows = []
    checks = Counter()
    for seq in data.DEV_CLIPS:
        c = ctx(seq)
        P = c.P
        # sanity: my reconstruction of appearance_match.py's candidate gate equals the file
        mine = {(a["piece"], b["piece"]) for a in P.values() for b in P.values() if a is not b and not pair_gate(a, b)}
        checks["cand_file"] += len(c.cand)
        checks["cand_mine"] += len(mine)
        checks["cand_diff"] += len(mine ^ set(c.cand))
        checks["cos_diff>1e-4"] += sum(abs(cos(P[a], P[b]) - s) > 1e-4 for (a, b), s in c.cand.items() if (a, b) in mine)
        my_links = sorted([a, b] for (a, b) in c.cand if mutual_status(c.cand, a, b)["status"] == "linked")
        checks["mutual_links_diff"] += int(my_links != sorted(base["mutual"][seq]["links"]))

        cuts = oracle_cuts(c)
        cuts1 = oracle_cuts(c, 1)
        cand_or = [[a, b] for (a, b) in c.cand if P[a]["ident"] is not None and P[a]["ident"] == P[b]["ident"]]
        methods["O1_links_known"][seq] = {"links": oracle_links(c, MIN_MATCHED)}
        methods["O2_links_any"][seq] = {"links": oracle_links(c, 1)}
        methods["O3_cuts_links_known"][seq] = {"links": oracle_cut_links(c, cuts, MIN_MATCHED), "cuts": cuts}
        methods["O4_cuts_links_any"][seq] = {"links": oracle_cut_links(c, cuts, 1), "cuts": cuts}
        methods["O5_candidate_pairs_only"][seq] = {"links": cand_or}
        methods["O6_cuts1_links_any"][seq] = {"links": oracle_cut_links(c, cuts1, 1), "cuts": cuts1}
        per = {n: score.score_clip(seq, base[n][seq]["links"])["per_return"] for n in base}
        per.update({n: score.score_clip(seq, m[seq]["links"], m[seq].get("cuts", []))["per_return"] for n, m in methods.items()})

        fr, rf, pc = c.clip.frames, c.clip.refs, c.clip.piece
        for i, r in enumerate(c.clip.returns):
            rid = r["ref_id"]
            g = c.people_by[rid]
            f_all = g.frame.to_numpy()
            before = g[g.frame == r["last"]].iloc[0]
            after = g[g.frame == r["first"]].iloc[0]
            dist_ann = float(np.hypot(after.pitch_x - before.pitch_x, after.pitch_y - before.pitch_y))
            el = (r["first"] - r["last"]) / FPS
            # the annotated visible stretches next to the gap (bounded by the person's other gaps >= 10 frames)
            brk = np.where(np.diff(f_all) - 1 >= GAP_MIN)[0]
            starts = np.concatenate([[f_all[0]], f_all[brk + 1]])
            ends = np.concatenate([f_all[brk], [f_all[-1]]])
            s_before = int(starts[ends == r["last"]][0])
            e_after = int(ends[starts == r["first"]][0])
            row = {
                "clip": seq, "ret": i, "ref_id": rid, "ann_role": c.att[rid]["role"], "ann_team": c.att[rid]["team"],
                "ann_jersey": c.att[rid]["jersey"], "exit_frame": r["last"], "reentry_frame": r["first"],
                "gap_frames": r["gap"], "gap_s": round(r["gap"] / FPS, 2),
                "exit_side": sides_1080(before, c.ref), "reentry_side": sides_1080(after, c.ref),
                "ann_exit_x": round(float(before.pitch_x), 1), "ann_exit_y": round(float(before.pitch_y), 1),
                "ann_reentry_x": round(float(after.pitch_x), 1), "ann_reentry_y": round(float(after.pitch_y), 1),
                "ann_dist_m": round(dist_ann, 1), "ann_speed_mps": round(dist_ann / el, 2),
                "stretch_before_frames": r["last"] - s_before + 1, "stretch_after_frames": e_after - r["first"] + 1,
            }  # fmt: skip
            a, b = r["a"], r["b"]
            row["status"] = "scored" if a is not None and b is not None else ("undetected: " + ("A and B" if a is None and b is None else "A" if a is None else "B") + " missing")
            for tag, idx in (("a", a), ("b", b)):
                if idx is None:
                    row.update({f"frame_{tag}": None, f"piece_{tag}": None, f"plugin_{tag}": None})
                    continue
                p = P[int(pc[idx])]
                row.update({
                    f"frame_{tag}": int(fr[idx]), f"piece_{tag}": p["piece"], f"plugin_{tag}": p["plugin"],
                    f"{tag}_kind": kind(p, rid),
                    f"{tag}_first": p["first"], f"{tag}_last": p["last"], f"{tag}_rows": p["rows"], f"{tag}_clean": p["n_clean"],
                    f"{tag}_emb_rows": p["n_emb"], f"{tag}_team": p["team"], f"{tag}_vteam": p["vteam"],
                    f"{tag}_team_verdict": p["team_verdict"], f"{tag}_role": p["role"],
                    f"{tag}_matched": p["matched"], f"{tag}_ident": p["ident_any"], f"{tag}_ident_n": p["ident_n"],
                    f"{tag}_purity": p["purity"],
                    f"{tag}_person_rows": int(((pc == p["piece"]) & (rf == rid)).sum()),
                    f"{tag}_jersey": p["jersey"], f"{tag}_jersey_reads": p["jersey_reads"],
                    f"{tag}_team_agrees_ann": (p["team"] == c.att[rid]["team"]) if p["team"] is not None else None,
                })  # fmt: skip
            for n in per:
                row[f"out_{n}"] = per[n][i]

            # anchors: the person's OWN pieces (mainly him, >= 5 matched rows) nearest to the gap, inside
            # the annotated stretch before the exit / after the re-entry; found for undetected returns too
            own = [p["piece"] for p in P.values() if p["ident"] == rid]
            best_a = best_b = None
            for pid in own:
                m = (pc == pid) & (rf == rid)
                fa_ = fr[m & (fr <= r["last"]) & (fr >= s_before)]
                fb_ = fr[m & (fr >= r["first"]) & (fr <= e_after)]
                if len(fa_) and (best_a is None or fa_.max() > best_a[0]):
                    best_a = (int(fa_.max()), pid)
                if len(fb_) and (best_b is None or fb_.min() < best_b[0]):
                    best_b = (int(fb_.min()), pid)
            row["anchor_a"] = best_a[1] if best_a else None
            row["anchor_a_lead_frames"] = r["last"] - best_a[0] if best_a else None   # exit frame - last own row
            row["anchor_b"] = best_b[1] if best_b else None
            row["anchor_b_delay_frames"] = best_b[0] - r["first"] if best_b else None  # first own row - re-entry frame
            if best_a and best_b:
                qa, qb = P[best_a[1]], P[best_b[1]]
                row["anchors_same_piece"] = qa["piece"] == qb["piece"]
                row["anchors_same_plugin"] = qa["plugin"] == qb["plugin"]
                if qa["piece"] != qb["piece"]:
                    gate = pair_gate(qa, qb)
                    row["anchors_candidate"] = not gate
                    row["anchors_not_candidate_why"] = "; ".join(gate)
                    row["anchors_piece_gap_s"] = round((qb["first"] - qa["last"]) / FPS, 2)
                    row["anchors_mutual_status"] = mutual_status(c.cand, qa["piece"], qb["piece"])["status"]
                    row["anchors_pathG"] = pathg[seq].get(f"{qa['piece']},{qb['piece']}", "not an audit pair (gap > 10 s or overlap)")
                    row["anchors_cos_strict"] = None if cos(qa, qb) is None else round(cos(qa, qb), 4)
                    row["anchors_cos_any"] = None if cos(qa, qb, "desc_any") is None else round(cos(qa, qb, "desc_any"), 4)
                    row["anchors_desc_kinds"] = f"{qa['desc_kind']}/{qb['desc_kind']}"
                    row["anchors_clean"] = f"{qa['n_clean']}/{qb['n_clean']}"
                    row["anchors_teams"] = f"{qa['team']}/{qb['team']}"
                    row["anchors_vteams"] = f"{qa['vteam']}/{qb['vteam']}"
                    row["anchors_roles"] = f"{qa['role']}/{qb['role']}"
                    row["anchor_a_end_side"], row["anchor_b_start_side"] = qa["end_side"], qb["start_side"]
                    if qa["desc_any"] is not None and qb["desc_any"] is not None:
                        s = float(qa["desc_any"] @ qb["desc_any"])
                        ia, wa = impostor_best(c, qa, rid, later=True)
                        ib, wb = impostor_best(c, qb, rid, later=False)
                        row["imp_best_cos_for_a"], row["imp_best_piece_for_a"] = (None if ia is None else round(ia, 4)), wa
                        row["imp_best_cos_for_b"], row["imp_best_piece_for_b"] = (None if ib is None else round(ib, 4)), wb
                        row["true_minus_best_impostor"] = round(s - max(v for v in (ia, ib, -1.0) if v is not None), 4)
                    if qa["last_pos"] is not None and qb["first_pos"] is not None:
                        (fa, xa, ya), (fb, xb, yb) = qa["last_pos"], qb["first_pos"]
                        d = float(np.hypot(xb - xa, yb - ya))
                        row.update({"anchors_sys_dist_m": round(d, 1),
                                    "anchors_sys_speed_mps": round(d / ((fb - fa) / FPS), 2) if fb > fa else None,
                                    "anchors_reach_ok": reach_ok(qa, qb) if fb > fa else None,
                                    "anchor_a_end_x": round(xa, 1), "anchor_a_end_y": round(ya, 1),
                                    "anchor_b_start_x": round(xb, 1), "anchor_b_start_y": round(yb, 1)})  # fmt: skip

            if a is not None and b is not None:
                pa, pb = P[int(pc[a])], P[int(pc[b])]
                # rows of B's piece before row B / of A's piece after row A: who are they?
                mb = (pc == pb["piece"]) & (fr < fr[b])
                ma = (pc == pa["piece"]) & (fr > fr[a])
                cb, ca = Counter(rf[mb].tolist()), Counter(rf[ma].tolist())
                row["b_prior_matched"] = int(mb.sum())
                row["b_prior_other_rows"] = int(sum(v for k, v in cb.items() if k != rid))
                row["a_after_matched"] = int(ma.sum())
                row["a_after_other_rows"] = int(sum(v for k, v in ca.items() if k != rid))
                row["a_ends_after_exit_by"] = pa["last"] - int(fr[a])
                row["b_starts_before_reentry_by"] = int(fr[b]) - pb["first"]
                row["same_piece"] = pa["piece"] == pb["piece"]
                row["same_plugin"] = pa["plugin"] == pb["plugin"]
                gate = pair_gate(pa, pb)
                row["is_candidate"] = (pa["piece"], pb["piece"]) in c.cand
                row["not_candidate_why"] = "; ".join(gate)
                assert row["is_candidate"] == (not gate), (seq, pa["piece"], pb["piece"], gate)
                row["pathG"] = pathg[seq].get(f"{pa['piece']},{pb['piece']}", "not an audit pair (gap > 10 s or overlap)")
                row["piece_gap_s"] = round((pb["first"] - pa["last"]) / FPS, 2)
                row["cos_strict"] = None if cos(pa, pb) is None else round(cos(pa, pb), 4)
                row["cos_any"] = None if cos(pa, pb, "desc_any") is None else round(cos(pa, pb, "desc_any"), 4)
                row["desc_kind_a"], row["desc_kind_b"] = pa["desc_kind"], pb["desc_kind"]
                ms = mutual_status(c.cand, pa["piece"], pb["piece"])
                row["mutual_status"] = ms["status"]
                for k in ("n_later_of_a", "n_earlier_of_b", "rank_of_b_for_a", "rank_of_a_for_b", "a_best_later", "b_best_earlier"):
                    row[k] = ms[k]
                for k in ("margin_a_side", "margin_b_side", "a_best_later_cos", "b_best_earlier_cos"):
                    row[k] = None if ms[k] is None else round(ms[k], 4)
                row["a_best_later_is_person"] = None if ms["a_best_later"] is None else bool(P[ms["a_best_later"]]["ident"] == rid)
                row["b_best_earlier_is_person"] = None if ms["b_best_earlier"] is None else bool(P[ms["b_best_earlier"]]["ident"] == rid)
                # the system's own pitch positions around the gap
                if pa["last_pos"] is not None and pb["first_pos"] is not None and pb["first_pos"][0] > pa["last_pos"][0]:
                    (fa, xa, ya), (fb, xb, yb) = pa["last_pos"], pb["first_pos"]
                    d = float(np.hypot(xb - xa, yb - ya))
                    row.update({"sys_exit_x": round(xa, 1), "sys_exit_y": round(ya, 1), "sys_reentry_x": round(xb, 1),
                                "sys_reentry_y": round(yb, 1), "sys_dist_m": round(d, 1),
                                "sys_speed_mps": round(d / ((fb - fa) / FPS), 2), "sys_reach_ok": reach_ok(pa, pb)})  # fmt: skip
                row["a_end_side"], row["b_start_side"] = pa["end_side"], pb["start_side"]
            rows.append(row)

    df = pd.DataFrame(rows)

    def ceiling(r):
        if not r["status"].startswith("scored"):
            return "undetected (detection)"
        if r["out_O1_links_known"] == "right":
            return "linkable: existing pieces, known identity"
        if r["out_O2_links_any"] == "right":
            return "linkable only through a piece with < 5 matched rows"
        if r["out_O3_cuts_links_known"] == "right" or r["out_O4_cuts_links_any"] == "right":
            return "needs an in-track switch fix (cut of a run >= 5 rows)"
        if r["out_O6_cuts1_links_any"] == "right":
            return "needs a cut around a stray run of < 5 rows"
        return "not recoverable by cuts + links"

    df["ceiling_class"] = df.apply(ceiling, axis=1)
    df.to_csv(OUT / "returns.csv", index=False)

    res = {"checks": dict(checks)}
    for n, m in {**base, **methods}.items():
        res[n] = score.evaluate(m)["pooled"]
        (OUT / f"METHOD_oracle_{n}.json").write_text(json.dumps({"name": f"ORACLE or BASELINE {n} (diagnosis, not a method)", "clips": m}), encoding="utf-8")
    res["ceiling_class"] = df.ceiling_class.value_counts().to_dict()
    (OUT / "oracles.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    print(len(df), df.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
