"""v_robust_0 / r9_guards.py -- (LABELS: scoring only; the guards themselves are label-free rules)
does a simple guard remove the remaining wrong returns / links?

    python -I -B r9_guards.py     -> out/r9_guards.md, out/r9_guards.json

DIAGNOSIS, NOT A RESULT: the guards were thought of after looking at the wrong returns of these 13
clips (out/r5_wrong.md), so their numbers are in-sample for the choice of guard, although every
clip still runs with its fold's held-out numbers and no guard has a fitted parameter.

A. Post-filters on fuse's links (a link is dropped; nothing else changes):
   rows >= k        both pieces of a link have at least k boxes
   clean >= k       both pieces have at least k clean boxes (reid_lab.data.clean_rows)
   no bridge < k    a piece with fewer than k boxes (clean boxes) that has a link on BOTH sides
                    loses both (a tiny piece may hang on a chain's end but not connect two chains)
   gap <= 250       no link over more than 10 s
   position         a link needs a position bonus > 0 at the merge
   app >= x         appearance cosine of the merge at least x
B. Inside the linker: pieces with fewer than k boxes / clean boxes take no part in the
   agglomeration at all (fuse's linker.agglomerate with m_appear's candidate list filtered).
"""
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_0")
import numpy as np
import pandas as pd

import vr_common as vc
from vr_common import data, evidence, linker, method, prep

from reid_lab import score  # LABELS: scoring

POINTS = ("strict", "margin_0.50", "balanced", "recall")


def piece_counts(seq, cuts):
    ids = vc.piece_ids_after_cuts(seq, cuts)
    clean = data.clean_rows(seq)
    rows = pd.Series(ids[ids >= 0]).value_counts().to_dict()
    cl = pd.Series(ids[(ids >= 0) & clean]).value_counts().to_dict()
    return rows, cl


def post_filters(lk, rows, cl):
    """{guard name: boolean keep mask over the links of one clip}."""
    ra, rb = lk.a.map(rows).to_numpy(), lk.b.map(rows).to_numpy()
    ca, cb = lk.a.map(lambda p: cl.get(p, 0)).to_numpy(), lk.b.map(lambda p: cl.get(p, 0)).to_numpy()
    out = {"none (fuse as is)": np.ones(len(lk), bool)}
    for k in (2, 3, 5, 10):
        out[f"rows >= {k}"] = np.minimum(ra, rb) >= k
    for k in (1, 2, 3, 5):
        out[f"clean >= {k}"] = np.minimum(ca, cb) >= k
    has_next, has_prev = set(lk.a), set(lk.b)
    for k in (3, 5, 10):
        small = {p for p in has_next & has_prev if rows[p] < k}
        out[f"no bridge through a piece with < {k} rows"] = ~(lk.a.isin(small) | lk.b.isin(small)).to_numpy()
    for k in (2, 3, 5):
        small = {p for p in has_next & has_prev if cl.get(p, 0) < k}
        out[f"no bridge through a piece with < {k} clean rows"] = ~(lk.a.isin(small) | lk.b.isin(small)).to_numpy()
    out["gap <= 250 frames (10 s)"] = lk.gap.to_numpy() <= 250
    out["position bonus > 0 at the merge"] = lk.m_bonus.to_numpy() > 0
    for x in (0.5, 0.6, 0.7):
        out[f"merge appearance >= {x}"] = lk.m_app.to_numpy() >= x
    out["clean >= 2 and gap <= 250"] = (np.minimum(ca, cb) >= 2) & (lk.gap.to_numpy() <= 250)
    return out


def pooled(per):
    return vc.pool(per[s] for s in data.DEV_CLIPS), vc.pool(per[s] for s in data.DEV_CLIPS if s != "SNGS-162"), vc.counts(per["SNGS-162"]["per_return"])


def cell(per):
    a, b, c = pooled(per)
    return f"{vc.fmt_pool(a)} | {b['right']} / {b['wrong']} / {b['lost']} | {c['right']} / {c['wrong']} / {c['lost']}"


J = {"post": {}, "inside": {}}
L = ["# r9: simple guards against the remaining wrong returns (diagnosis; guards chosen after seeing these clips)", "",
     "Cells: right / wrong / lost | links right / wrong / unknown | impure rows | without SNGS-162 (119 scored) | SNGS-162 alone (18 scored).", ""]
# ---------------------------------------------------------------- A: post-filters
for point in POINTS:
    lk_all = pd.read_csv(vc.OUT / f"links_{point}.csv")
    M = vc.load_method(point)
    res = {}
    for seq in data.DEV_CLIPS:
        lk = lk_all[lk_all.seq == seq].reset_index(drop=True)
        cuts = prep.norm_cuts(M[seq]["cuts"])
        rows, cl = piece_counts(seq, cuts)
        for name, keep in post_filters(lk, rows, cl).items():
            links = lk.loc[keep, ["a", "b"]].astype(int).values.tolist()
            r = score.score_clip(seq, links, M[seq]["cuts"])
            res.setdefault(name, {})[seq] = {"per_return": r["per_return"], "links": r["links"], "impure_rows": r["impure_rows"], "tracked_rows": r["tracked_rows"],
                                             "dropped": int((~keep).sum())}
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
INSIDE = {"none (fuse as is)": None, "pieces with < 2 rows take no part": ("rows", 2), "pieces with < 3 rows take no part": ("rows", 3),
          "pieces with < 5 rows take no part": ("rows", 5), "pieces with < 1 clean row take no part": ("clean", 1),
          "pieces with < 2 clean rows take no part": ("clean", 2), "pieces with < 3 clean rows take no part": ("clean", 3),
          "pieces with < 5 clean rows take no part": ("clean", 5)}
PTS = ("strict", "fixed_0.50", "balanced", "recall", "fixed_0.25")
res = {}
par_all = vc.params()
for seq in data.DEV_CLIPS:
    par = par_all[seq]
    cuts = prep.cuts_for(seq, seq)
    table = evidence.pair_table(seq, cuts, method.pos_params(seq))
    bonus = method.bonus_of(table, par["pair_model"])
    rows, cl = piece_counts(seq, cuts)
    for name, rule in INSIDE.items():
        if rule is None:
            af.candidate_ids = orig_candidates
        else:
            cnt, k = (rows, rule[1]) if rule[0] == "rows" else (cl, rule[1])
            af.candidate_ids = lambda s, pr=None, cuts=(), _c=cnt, _k=k: [p for p in orig_candidates(s, pr, cuts) if _c.get(p, 0) >= _k]
        try:
            for point in PTS:
                m = float(point.split("_")[1]) if point.startswith("fixed_") else par["margin"][point]
                links = linker.agglomerate(seq, cuts, bonus, t_abs=par["t_abs"], margin=m, floor=par["floor"], t0=par["t0"])
                r = score.score_clip(seq, links, [list(x) for x in cuts])
                res.setdefault((name, point), {})[seq] = {"per_return": r["per_return"], "links": r["links"], "impure_rows": r["impure_rows"],
                                                          "tracked_rows": r["tracked_rows"]}
        finally:
            af.candidate_ids = orig_candidates
    print(seq, flush=True)
L += ["## B. Inside the linker: tiny pieces take no part in the agglomeration", ""]
for point in PTS:
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
