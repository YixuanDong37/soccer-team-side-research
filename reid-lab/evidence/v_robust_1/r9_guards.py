"""v_robust_1 / r9_guards.py -- (LABELS: scoring only; the guards themselves are label-free rules)
does a simple guard remove the remaining wrong returns / links of fuse_r1?

    python -I -B r9_guards.py     -> out/r9_guards.md, out/r9_guards.json

DIAGNOSIS, NOT A RESULT: the guards were thought of after looking at the wrong returns of these 13
clips (out/r5_wrong.md), so their numbers are in-sample for the choice of guard, although every
clip still runs with its fold's held-out numbers and no guard has a fitted parameter.

A. Post-filters on fuse_r1's links (a link is dropped; nothing else changes):
   clean >= k       both pieces of a link have at least k clean boxes (reid_lab.data.clean_rows)
   rows >= k        both pieces have at least k boxes
   gap <= 250       no link over more than 10 s
   position         a link needs a position bonus > 0 at the merge / pos_sink of the pair >= x
   app >= x         appearance cosine of the merge at least x;  piece cos_w >= x
   pair merges only the merge joined two single pieces (no group was involved)
   no plugin-only   drop links whose bonus rests on the plugin's join flag (plug = 1)
B. Inside the linker (linker2 re-run):
   pieces with fewer than k clean boxes take no part (neither solid nor leaf);
   appearance floor 0.4 / 0.5 / 0.6 instead of 0.3;  "nobody" 0.6 / 0.7;  no position bonus at all;
   solid from 10 / 20 boxes.
"""
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import numpy as np
import pandas as pd

import vr_common as vc
from vr_common import data, evidence, linker2, method, prep

from reid_lab import score  # LABELS: scoring

POINTS = ("strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall")


def piece_counts(seq, cuts):
    ids = vc.piece_ids_after_cuts(seq, cuts)
    clean = data.clean_rows(seq)
    rows = pd.Series(ids[ids >= 0]).value_counts().to_dict()
    cl = pd.Series(ids[(ids >= 0) & clean]).value_counts().to_dict()
    return rows, cl


def post_filters(lk):
    """{guard name: boolean keep mask over the links of one clip}."""
    ra, rb = lk.rows_a.to_numpy(), lk.rows_b.to_numpy()
    ca, cb = lk.clean_a.to_numpy(), lk.clean_b.to_numpy()
    out = {"none (fuse_r1 as is)": np.ones(len(lk), bool)}
    for k in (1, 2, 3, 5, 10):
        out[f"clean >= {k}"] = np.minimum(ca, cb) >= k
    for k in (10, 20):
        out[f"rows >= {k}"] = np.minimum(ra, rb) >= k
    out["gap <= 250 frames (10 s)"] = lk.gap.to_numpy() <= 250
    out["gap <= 125 frames (5 s)"] = lk.gap.to_numpy() <= 125
    out["position bonus > 0 at the merge"] = lk.m_bonus.to_numpy() > 0
    for x in (0.2, 0.4):
        out[f"pos_sink >= {x}"] = lk.pos_sink.fillna(0).to_numpy() >= x
    for x in (0.6, 0.7, 0.8):
        out[f"merge appearance >= {x}"] = lk.m_app.to_numpy() >= x
    for x in (0.6, 0.7):
        out[f"piece cos_w >= {x}"] = lk.cos_w.fillna(0).to_numpy() >= x
    out["pair merges only (group size 2)"] = lk.m_size.to_numpy() == 2
    out["no link with the plugin flag"] = lk.plug.fillna(0).to_numpy() == 0
    out["clean >= 1 and merge appearance >= 0.6"] = (np.minimum(ca, cb) >= 1) & (lk.m_app.to_numpy() >= 0.6)
    out["both margins >= 0.8"] = np.minimum(lk.m_rg.to_numpy(), lk.m_rh.to_numpy()) >= 0.8
    return out


def pooled(per):
    return vc.pool(per[s] for s in data.DEV_CLIPS), vc.pool(per[s] for s in data.DEV_CLIPS if s != "SNGS-162"), vc.counts(per["SNGS-162"]["per_return"])


def cell(per):
    a, b, c = pooled(per)
    return f"{vc.fmt_pool(a)} | {b['right']} / {b['wrong']} / {b['lost']} | {c['right']} / {c['wrong']} / {c['lost']}"


def keep(r):
    return {"per_return": r["per_return"], "links": r["links"], "impure_rows": r["impure_rows"], "tracked_rows": r["tracked_rows"]}


if __name__ == "__main__":
    J = {"post": {}, "inside": {}}
    L = ["# r9: simple guards against the remaining wrong returns of fuse_r1 (diagnosis; guards chosen after seeing these clips)", "",
         "Cells: right / wrong / lost | links right / wrong / unknown | impure rows | without SNGS-162 (119 scored) | SNGS-162 alone (18 scored).", ""]
    # ---------------------------------------------------------------- A: post-filters
    for point in POINTS:
        lk_all = pd.read_csv(vc.OUT / f"links_{point}.csv")
        M = vc.load_method(point)
        res = {}
        for seq in data.DEV_CLIPS:
            lk = lk_all[lk_all.seq == seq].reset_index(drop=True)
            for name, kp in post_filters(lk).items():
                links = lk.loc[kp, ["a", "b"]].astype(int).values.tolist()
                r = score.score_clip(seq, links, M[seq]["cuts"])
                res.setdefault(name, {})[seq] = {**keep(r), "dropped": int((~kp).sum())}
        L += [f"## A. Post-filters, {point}", "", "| guard | links dropped | right / wrong / lost | links r / w / u | impure | without SNGS-162 | SNGS-162 |", "|---|---|---|---|---|---|---|"]
        J["post"][point] = {}
        for name, per in res.items():
            a, b, c = pooled(per)
            J["post"][point][name] = {"pooled": a, "without_162": b, "only_162": c, "dropped": sum(v["dropped"] for v in per.values())}
            L.append(f"| {name} | {J['post'][point][name]['dropped']} | {cell(per)} |")
        L.append("")

    # ---------------------------------------------------------------- B: inside the linker
    af = prep._import("m_appear", "features")
    orig_candidates = af.candidate_ids
    INSIDE = {"none (fuse_r1 as is)": {}, "pieces with < 1 clean box take no part": {"clean": 1}, "pieces with < 2 clean boxes take no part": {"clean": 2},
              "pieces with < 3 clean boxes take no part": {"clean": 3}, "pieces with < 5 clean boxes take no part": {"clean": 5},
              "appearance floor 0.4": {"floor": 0.4}, "appearance floor 0.5": {"floor": 0.5}, "appearance floor 0.6": {"floor": 0.6},
              "nobody (t0) 0.6": {"t0": 0.6}, "nobody (t0) 0.7": {"t0": 0.7},
              "no position / plugin bonus (appearance only)": {"scale": 0.0}, "bonus x 0.5": {"scale": 0.5},
              "solid from 10 boxes": {"n_solid": 10}, "solid from 20 boxes": {"n_solid": 20},
              "floor 0.5 and < 1 clean box takes no part": {"floor": 0.5, "clean": 1}}
    res = {}
    P = vc.params()
    for seq in data.DEV_CLIPS:
        par = P[seq]
        cuts = prep.cuts_for(seq, seq)
        pp = method.pos_params(seq)
        cl = [list(x) for x in cuts]
        rows, cln = piece_counts(seq, cuts)
        full_tab = evidence.pair_table(seq, cuts, pp, 0)
        for name, g in INSIDE.items():
            ns = g.get("n_solid", method.N_SOLID)
            core_tab = evidence.pair_table(seq, cuts, pp, ns)
            sc = g.get("scale", 1.0)
            bonus_core = method.bonus_of(core_tab, par["pair_model"], scale=sc) if sc > 0 else None
            bonus_leaf = method.bonus_of(full_tab, par["pair_model"], only=method.LEAF_COLS, scale=sc) if sc > 0 else None
            if "clean" in g:
                af.candidate_ids = lambda s, pr=None, cuts=(), _c=cln, _k=g["clean"]: [p for p in orig_candidates(s, pr, cuts) if _c.get(p, 0) >= _k]
            try:
                done = {}
                for point in POINTS:
                    margin, leaves, _, _ = vc.setting(point, seq)
                    if (margin, leaves) not in done:
                        core, leaf = linker2.link(seq, cuts, bonus_core, bonus_leaf if leaves else None, margin=margin, floor=g.get("floor", par["floor"]),
                                                  t0=g.get("t0", par["t0"]), t_abs=par["t_abs"], n_solid=ns, leaves=leaves)
                        done[(margin, leaves)] = (core + leaf, keep(score.score_clip(seq, core + leaf, cl)))
                    links, r = done[(margin, leaves)]
                    if not g:
                        Mf = vc.load_method(point)[seq]
                        assert sorted(map(tuple, links)) == sorted(map(tuple, Mf["links"])), (seq, point)
                    res.setdefault((name, point), {})[seq] = r
            finally:
                af.candidate_ids = orig_candidates
        print(seq, flush=True)
    L += ["## B. Inside the linker (linker2 re-run with one change)", ""]
    for point in POINTS:
        L += [f"### {point}", "", "| guard | right / wrong / lost | links r / w / u | impure | without SNGS-162 | SNGS-162 |", "|---|---|---|---|---|---|"]
        J["inside"][point] = {}
        for name in INSIDE:
            per = res[(name, point)]
            a, b, c = pooled(per)
            J["inside"][point][name] = {"pooled": a, "without_162": b, "only_162": c}
            L.append(f"| {name} | {cell(per)} |")
        L.append("")
    (vc.OUT / "r9_guards.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (vc.OUT / "r9_guards.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
    print("\n".join(L))
