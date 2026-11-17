"""fuse_r1 / fit_loco.py -- the leave-one-clip-out (LOCO) run of fuse_r1 (LABELS: used to fit the pair
model and to choose each operating point's setting on the 12 fit clips of a fold, and to score).

For a fold (held-out clip c) every fitted number has been fitted WITHOUT c:
  cuts          m_cuts' detector weights loco_params.json[c], P(switch) >= 0.7     (5 numbers, m_cuts)
  position      m_space's weights params_cache/excl_<c>.json                        (16 numbers, m_space)
  pair model    fitted here on the 12 fit clips: logistic regression of "same person, directly
                following" on cos_w, sink_lo, plug for pairs of SOLID pieces (3 weights + bias; weak
                labels, >= 1 matched row)
  setting       (leaves on / off, margin) per operating point, chosen here on the 12 fit clips by
                rule B of PREREG.txt (rules.py)
and all of it is applied unchanged to c. A fold runs in four phases so that the use of labels is easy
to audit: A evidence tables (label-free), B fit (labels of the FIT clips only), C links of all 13
clips (label-free), D scoring (labels; the held-out clip's labels are used here and nowhere else).

    python -I fit_loco.py [workers]
      -> cache/loco_raw.pkl      every fold x variant x margin x clip (links, outcomes)
         params_loco.json        per fold (and "all13"): pair model, chosen settings     (method.py reads this)
         out/loco_report.json    operating points under every rule, fixed-setting rows, per-clip results
         out/tradeoff.md         the same as tables

Variants:
  r1   the method: tiers + solid-only position table + optional leaves (linker2)
  r0   round 0's linker for comparison (all pieces in one agglomeration, fuse's linker.py); its
       fixed-margin rows must reproduce fuse's (checked by check_r0.py)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_1")
from reid_lab import data  # noqa: E402
import evidence  # noqa: E402
import linker  # noqa: E402
import linker2  # noqa: E402
import pairmodel  # noqa: E402
import prep  # noqa: E402
import rules  # noqa: E402
from method import LEAF_COLS, N_SOLID, bonus_of, pos_params  # noqa: E402  (the label-free apply path)

HERE = Path(__file__).resolve().parent
MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)
FLOOR, T0, T_ABS = 0.3, 0.5, 0.5       # fixed (not searched): appearance floor, "nobody" rival, m_appear's bar
COLS = ["cos_w", "sink_lo", "plug"]
MIN_ROWS_FIT = 1          # weak labels for the pair model: majority person of >= 1 matched row
ALL = "all13"
CANDS = {"r1": [(lv, m) for lv in (False, True) for m in MARGINS], "r0": [(False, m) for m in MARGINS]}
POINTS = ("strict", "balanced", "recall")


def _scored(score, seq, links, cuts_list, ident=None, leaf_links=None) -> dict:
    r = score.score_clip(seq, links, cuts_list)
    out = {"per_return": r["per_return"], "lk": r["links"], "impure": r["impure_rows"], "rows": r["tracked_rows"]}
    if ident is not None:
        def weak(ls):
            c = Counter()
            for a, b in ls:
                c["none" if (a not in ident or b not in ident) else ("agree" if ident[a][0] == ident[b][0] else "disagree")] += 1
            return {k: int(c[k]) for k in ("agree", "disagree", "none")}
        out["weak"] = weak(links)
        if leaf_links is not None:
            out["weak_leaf"] = weak(leaf_links)
            out["leaf_disagree"] = out["weak_leaf"]["disagree"]
    return out


def run_fold(fold: str, margins=MARGINS, variants=("r1", "r0")):
    """fold = held-out clip, or ALL (fit on all 13 clips: the frozen method, in-sample)."""
    import pandas as pd

    t_start = time.time()
    f = None if fold == ALL else fold
    pp = pos_params(f)
    fit_clips = [s for s in data.DEV_CLIPS if s != fold]
    # ---- A: evidence tables, label-free
    cuts = {s: prep.cuts_for(s, f) for s in data.DEV_CLIPS}
    core = {s: evidence.pair_table(s, cuts[s], pp, N_SOLID) for s in data.DEV_CLIPS}
    full = {s: evidence.pair_table(s, cuts[s], pp, 0) for s in data.DEV_CLIPS}
    # ---- B: fit, labels of the FIT clips only
    import fuse_labels as fitlabels
    models = {}
    lab = pd.concat([fitlabels.label_pairs(core[s], s, cuts[s], MIN_ROWS_FIT) for s in fit_clips], ignore_index=True)
    models["r1"] = pairmodel.fit(lab, COLS)
    if "r0" in variants:
        lab0 = pd.concat([fitlabels.label_pairs(full[s], s, cuts[s], MIN_ROWS_FIT) for s in fit_clips], ignore_index=True)
        models["r0"] = pairmodel.fit(lab0, COLS)
    # ---- C: links of all 13 clips, label-free
    links = {}
    for s in data.DEV_CLIPS:
        bc = bonus_of(core[s], models["r1"])
        bl = bonus_of(full[s], models["r1"], only=LEAF_COLS)
        b0 = bonus_of(full[s], models["r0"]) if "r0" in variants else None
        for m in margins:
            links[("r1", m, s)] = linker2.link(s, cuts[s], bc, bl, margin=m, floor=FLOOR, t0=T0, t_abs=T_ABS, n_solid=N_SOLID, leaves=True)
            if "r0" in variants:
                links[("r0", m, s)] = linker.agglomerate(s, cuts[s], b0, t_abs=T_ABS, margin=m, floor=FLOOR, t0=T0)
    # ---- D: scoring (labels)
    from reid_lab import score
    res = {}
    for s in data.DEV_CLIPS:
        cl = [list(x) for x in cuts[s]]
        ident = fitlabels.identities(s, cuts[s])
        res[("cuts", None, s)] = {"off": _scored(score, s, [], cl)}
        for m in margins:
            c_links, l_links = links[("r1", m, s)]
            res[("r1", m, s)] = {"core": c_links, "leaf": l_links,
                                 "off": _scored(score, s, c_links, cl, ident, []),
                                 "on": _scored(score, s, c_links + l_links, cl, ident, l_links)}
            if "r0" in variants:
                res[("r0", m, s)] = {"core": links[("r0", m, s)], "off": _scored(score, s, links[("r0", m, s)], cl, ident)}
    return {"fold": fold, "cuts": {s: [list(x) for x in cuts[s]] for s in data.DEV_CLIPS}, "models": models, "res": res,
            "seconds": round(time.time() - t_start, 1)}


# ----------------------------------------------------------------------------- report
def per_clip_cells(folds, variant, chosen) -> dict:
    """{clip: counts} with every clip taken from its own fold at the candidate chosen for it."""
    return {c: rules.cell(folds[c]["res"], variant, chosen[c], c) for c in data.DEV_CLIPS}


def report(raw: dict) -> dict:
    folds = raw["folds"]
    rep = {"margins": list(MARGINS), "lambda": rules.LAMBDA, "caps": rules.CAPS, "variants": {}}
    cuts_cells = [rules.cell(folds[c]["res"], "cuts", (False, None), c) for c in data.DEV_CLIPS]
    rep["cuts_only_heldout"] = rules.pool(cuts_cells)
    rep["cuts_only_per_clip"] = {c: x for c, x in zip(data.DEV_CLIPS, cuts_cells)}
    for variant, cands in CANDS.items():
        if (variant, MARGINS[0], data.DEV_CLIPS[0]) not in folds[ALL]["res"]:
            continue
        V = {"rules": {}, "fixed_heldout": {}, "fixed_all13_insample": {}, "fixed_heldout_per_clip": {}}
        for cand in cands:
            key = f"{'on' if cand[0] else 'off'}|{cand[1]}"
            cells = per_clip_cells(folds, variant, {c: cand for c in data.DEV_CLIPS})
            V["fixed_heldout"][key] = rules.pool(cells.values())
            V["fixed_heldout_per_clip"][key] = cells
            V["fixed_all13_insample"][key] = rules.pool(rules.cell(folds[ALL]["res"], variant, cand, c) for c in data.DEV_CLIPS)
        for rule in rules.RULES:
            R = {}
            for point in POINTS:
                chosen, why = {}, {}
                for c in data.DEV_CLIPS:
                    chosen[c], why[c] = rules.choose(folds[c]["res"], variant, cands, [s for s in data.DEV_CLIPS if s != c], point, rule)
                all_c, all_why = rules.choose(folds[ALL]["res"], variant, cands, list(data.DEV_CLIPS), point, rule)
                held = per_clip_cells(folds, variant, chosen)
                fixed = per_clip_cells(folds, variant, {c: all_c for c in data.DEV_CLIPS})
                ins = [rules.cell(folds[ALL]["res"], variant, all_c, c) for c in data.DEV_CLIPS]
                R[point] = {
                    "chosen_per_fold": {c: {"leaves": bool(v[0]), "margin": v[1]} for c, v in chosen.items()},
                    "why_per_fold": why,
                    "chosen_all13": {"leaves": bool(all_c[0]), "margin": all_c[1]}, "why_all13": all_why,
                    "folds_equal_to_all13": int(sum(v == all_c for v in chosen.values())),
                    "heldout": rules.pool(held.values()), "heldout_per_clip": held,
                    "heldout_without_SNGS-162": rules.pool(v for c, v in held.items() if c != "SNGS-162"),
                    "frozen_setting_heldout": rules.pool(fixed.values()), "frozen_setting_heldout_per_clip": fixed,
                    "insample_all13": rules.pool(ins),
                    "bootstrap": rules.bootstrap(held), "bootstrap_frozen_setting": rules.bootstrap(fixed),
                }
            V["rules"][rule] = R
        V["pair_model"] = {c: folds[c]["models"][variant] for c in folds}
        rep["variants"][variant] = V
    return rep


def fmt(p: dict) -> str:
    return (f"{p['right']} / {p['wrong']} / {p['lost']} | {p['links_right']} / {p['links_wrong']} / {p['links_unknown']} | "
            f"{100 * p['impure_row_share']:.2f}%")


def cand_txt(c: dict) -> str:
    return f"{c['margin']:.2f} {'+leaves' if c['leaves'] else 'core'}"


def tables(rep: dict) -> str:
    L = ["# fuse_r1: leave-one-clip-out results (137 scored returns, 13 dev clips)", "",
         "Columns: right / wrong / lost | links right / wrong / unknown | impure rows.",
         "Baselines (score.py): gbf now (split) 0 / 4 / 133 | 0 / 0 / 0 | 2.61%; plugin 30 / 14 / 93 | 33 / 10 / 50 | 3.08%; mutual 26 / 6 / 105 | 60 / 7 / 2 | 3.03%.",
         f"Baseline of the linking step, cuts only (held-out weights): {fmt(rep['cuts_only_heldout'])}.", ""]
    names = {"B": "rule B (PREREG.txt; the METHOD files)", "R0": "R0: round 0's rule (J, no cap, no clip left out)",
             "Rcap": "Rcap: cap only (most right under the cap)", "Rnotrim": "Rnotrim: rule B without leaving a clip out"}
    for variant, V in rep["variants"].items():
        L += [f"## variant {variant}" + (" (the method)" if variant == "r1" else " (round 0's linker, for comparison)"), ""]
        for rule, R in V["rules"].items():
            L += [f"### {names[rule]}", "",
                  "| point | HELD-OUT (rule applied per fold) | frozen setting, held-out weights | in-sample (all 13) | settings chosen per fold | setting on all 13 | folds = all 13 |",
                  "|---|---|---|---|---|---|---|"]
            for point, P in R.items():
                cnt = Counter(cand_txt(v) for v in P["chosen_per_fold"].values())
                L.append(f"| {point} | {fmt(P['heldout'])} | {fmt(P['frozen_setting_heldout'])} | {fmt(P['insample_all13'])} | "
                         f"{', '.join(f'{n}x {k}' for k, n in sorted(cnt.items()))} | {cand_txt(P['chosen_all13'])} | {P['folds_equal_to_all13']} / 13 |")
            L.append("")
        L += ["### fixed setting (nothing chosen; HELD-OUT = each clip with the weights fitted without it)", "",
              "| leaves | margin | HELD-OUT | without SNGS-162 | SNGS-162 | leaf links weak-label disagree | in-sample (all-13 weights) |", "|---|---|---|---|---|---|---|"]
        for key, p in V["fixed_heldout"].items():
            lv, m = key.split("|")
            cells = V["fixed_heldout_per_clip"][key]
            no162 = rules.pool(v for c, v in cells.items() if c != "SNGS-162")
            c162 = cells["SNGS-162"]
            L.append(f"| {lv} | {float(m):.2f} | {fmt(p)} | {no162['right']} / {no162['wrong']} / {no162['lost']} | "
                     f"{c162['right']} / {c162['wrong']} / {c162['lost']} | {p['leaf_disagree']} | {fmt(V['fixed_all13_insample'][key])} |")
        L.append("")
    V = rep["variants"]["r1"]["rules"]["B"]
    L += ["## r1, rule B: held-out result per clip (right / wrong / lost, links right / wrong / unknown, setting used)", "",
          "| clip | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for c in data.DEV_CLIPS:
        cells = []
        for P in V.values():
            x = P["heldout_per_clip"][c]
            cells.append(f"{x['right']} / {x['wrong']} / {x['lost']}, {x['lr']} / {x['lw']} / {x['lu']} ({cand_txt(P['chosen_per_fold'][c])})")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "Clip bootstrap of the held-out shares (13 clips resampled, 5% / 50% / 95%):", ""]
    for point, P in V.items():
        b = P["bootstrap"]
        L.append(f"- {point}: right share {b['right_share_q05_q50_q95']}, wrong share {b['wrong_share_q05_q50_q95']}, "
                 f"P(wrong share <= {rules.CAPS[point]}/137) = {b['P_wrong_share_le_%d_of_137' % rules.CAPS[point]]}")
    return "\n".join(L) + "\n"


def write_outputs(raw: dict) -> dict:
    rep = report(raw)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "loco_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (HERE / "out" / "tradeoff.md").write_text(tables(rep), encoding="utf-8")
    res = raw["folds"]
    B = rep["variants"]["r1"]["rules"]["B"]
    params = {}
    for c in res:
        params[c] = {"pair_model": res[c]["models"]["r1"],
                     "choice": {p: (B[p]["chosen_all13"] if c == ALL else B[p]["chosen_per_fold"][c]) for p in POINTS},
                     "rule": "rule B of PREREG.txt (rules.py)",
                     "cuts": "m_cuts frozen weights" if c == ALL else f"m_cuts loco_params.json[{c}]", "p_cut": prep.P_CUT,
                     "position": "m_space/params_all13.json" if c == ALL else f"m_space/params_cache/excl_{c.split('-')[1]}.json",
                     "n_solid": N_SOLID, "floor": FLOOR, "t0": T0, "t_abs": T_ABS}
    (HERE / "params_loco.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    return rep


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex:
        for r in ex.map(run_fold, [*data.DEV_CLIPS, ALL]):
            res[r["fold"]] = r
            m = r["models"]["r1"]
            print(r["fold"], r["seconds"], "s; r1 pair model", dict(zip(m["cols"], np.round(m["w"], 3).tolist())), "bias", round(m["b"], 3),
                  "n", m["n"], "pos", m["pos"], flush=True)
    if len(res) != len(data.DEV_CLIPS) + 1:
        raise SystemExit("a fold is missing")
    raw = {"folds": res}
    with open(HERE / "cache" / "loco_raw.pkl", "wb") as fh:
        pickle.dump(raw, fh)
    rep = write_outputs(raw)
    print(tables(rep))
    print("done", round(time.time() - t0, 1), "s")
