"""fuse / fit_loco.py -- the leave-one-clip-out (LOCO) run of the fusion method (LABELS: used to fit
the pair model and to choose the margin on the 12 fit clips of each fold, and to score).

For a fold (held-out clip c) every fitted number has been fitted WITHOUT c:
  cuts          m_cuts' detector weights loco_params.json[c], P(switch) >= 0.7     (5 numbers, m_cuts)
  position      m_space's weights params_cache/excl_<c>.json                        (16 numbers, m_space)
  pair model    fitted here on the 12 fit clips: logistic regression of "same person, directly
                following" on cos_w, sink_lo, plug (3 weights + bias; weak labels, >= 1 matched row)
  margin        chosen here on the 12 fit clips, one value per operating point
and all of it is applied unchanged to c. The same numbers are also applied to the 12 fit clips
(that is where the margin is chosen), so nothing fitted in a fold has seen its held-out clip.

Rule for choosing the margin of an operating point (fixed before the pooled held-out result was
looked at; the same objective as m_appear and m_keep, with their penalties):
    J(margin) = sum over the 12 fit clips of  right - lam * (wrong returns + wrong links)
    strict: lam = 10      balanced: lam = 3      recall: lam = 1       ties -> the larger margin
The pooled held-out result of a point = the 13 held-out clips, each scored with its own fold's numbers.

    python -I fit_loco.py [workers]
      -> cache/loco_raw.pkl      every fold x variant x margin x clip (links, outcomes)
         params_loco.json        per fold (and "all13"): pair model, chosen margins     (method.py reads this)
         out/loco_report.json    operating points, fixed-margin trade-off rows, per-clip results
         out/tradeoff.md         the same as tables

Variants (ablations use the same rule):
  fuse        cuts + appearance + position bonus + plugin-join bonus          <- the method
  fuse_pos    the same without the plugin-join bonus
  app_cut     cuts + m_appear's own rule (t_abs 0.5, real rivals only), no bonus
  fuse_nocut  fuse on the uncut pieces (no m_cuts)
  app_nocut   m_appear's own rule on the uncut pieces (re-run of m_appear's margin-only LOCO)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
from reid_lab import data  # noqa: E402
import evidence  # noqa: E402
import linker  # noqa: E402
import pairmodel  # noqa: E402
import prep  # noqa: E402
from method import bonus_of, pos_params  # noqa: E402  (the label-free apply path)

HERE = Path(__file__).resolve().parent
MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)
APP_MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
FLOOR, T0, T_ABS = 0.3, 0.5, 0.5       # fixed (not searched): appearance floor, "nobody" rival, m_appear's bar
FUSE_COLS = ["cos_w", "sink_lo", "plug"]
VARIANTS = {
    "fuse":       dict(cut=True,  cols=FUSE_COLS,             floor=FLOOR, t0=T0,   margins=MARGINS),
    "fuse_pos":   dict(cut=True,  cols=["cos_w", "sink_lo"],  floor=FLOOR, t0=T0,   margins=MARGINS),
    "app_cut":    dict(cut=True,  cols=None,                  floor=None,  t0=None, margins=APP_MARGINS),
    "fuse_nocut": dict(cut=False, cols=FUSE_COLS,             floor=FLOOR, t0=T0,   margins=MARGINS),
    "app_nocut":  dict(cut=False, cols=None,                  floor=None,  t0=None, margins=APP_MARGINS),
}  # fmt: skip
LAMBDA = {"strict": 10.0, "balanced": 3.0, "recall": 1.0}
CAPS = {"strict": 4, "balanced": 7, "recall": 14}
MIN_ROWS_FIT = 1          # weak labels for the pair model: majority person of >= 1 matched row
ALL = "all13"


def run_fold(fold: str):
    """fold = held-out clip, or ALL (fit on all 13 clips: the frozen method, in-sample)."""
    import pandas as pd

    import fuse_labels as fitlabels   # LABELS: pair-model fit on the fit clips, and scoring
    from reid_lab import score

    t0 = time.time()
    f = None if fold == ALL else fold
    pp = pos_params(f)
    fit_clips = [s for s in data.DEV_CLIPS if s != fold]
    out = {"fold": fold, "cuts": {}, "models": {}, "res": {}}
    tabs, cuts = {}, {}
    for cut in (True, False):
        cuts[cut] = {s: (prep.cuts_for(s, f) if cut else ()) for s in data.DEV_CLIPS}
        tabs[cut] = {s: evidence.pair_table(s, cuts[cut][s], pp) for s in data.DEV_CLIPS}
    out["cuts"] = {s: [list(x) for x in cuts[True][s]] for s in data.DEV_CLIPS}
    lab = {cut: pd.concat([fitlabels.label_pairs(tabs[cut][s], s, cuts[cut][s], MIN_ROWS_FIT) for s in fit_clips], ignore_index=True)
           for cut in (True, False)}
    for variant, v in VARIANTS.items():
        cut = v["cut"]
        model = pairmodel.fit(lab[cut], v["cols"]) if v["cols"] else None
        out["models"][variant] = model
        for s in data.DEV_CLIPS:
            bonus = bonus_of(tabs[cut][s], model) if model else None
            cl = [list(x) for x in cuts[cut][s]]
            for m in v["margins"]:
                links = linker.agglomerate(s, cuts[cut][s], bonus, t_abs=T_ABS, margin=m, floor=v["floor"], t0=v["t0"])
                r = score.score_clip(s, links, cl)
                out["res"][(variant, m, s)] = {"links": links, "per_return": r["per_return"], "lk": r["links"],
                                               "impure": r["impure_rows"], "rows": r["tracked_rows"]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ----------------------------------------------------------------------------- pooling / selection
def pool(items) -> dict:
    """items: iterable of per-clip result dicts -> pooled counts."""
    c, lk, imp, rows = Counter(), Counter(), 0, 0
    for x in items:
        c.update(x["per_return"])
        lk.update({k: v for k, v in x["lk"].items() if k != "n"})
        imp += x["impure"]
        rows += x["rows"]
    return {"right": c["right"], "wrong": c["wrong"], "lost": c["lost"], "links_right": lk["right"], "links_wrong": lk["wrong"],
            "links_unknown": lk["unknown"], "impure_row_share": round(imp / rows, 5) if rows else None}


def choose(run: dict, variant: str, fit_clips, lam: float) -> float:
    """The margin with the largest J on the fit clips; ties -> the larger margin."""
    best = None
    for m in VARIANTS[variant]["margins"]:
        p = pool(run["res"][(variant, m, s)] for s in fit_clips)
        j = p["right"] - lam * (p["wrong"] + p["links_wrong"])
        if best is None or j >= best[0]:
            best = (j, m)
    return best[1]


def choose_cap(run: dict, variant: str, fit_clips, cap: int) -> float:
    """SECONDARY rule (sensitivity): most right on the fit clips with wrong <= floor(cap * scored_fit / 137);
    when no margin qualifies: fewest wrong, then most right; ties -> the larger margin."""
    rows = []
    for m in VARIANTS[variant]["margins"]:
        p = pool(run["res"][(variant, m, s)] for s in fit_clips)
        rows.append((m, p["right"], p["wrong"], p["right"] + p["wrong"] + p["lost"]))
    budget = int(np.floor(cap * rows[0][3] / 137.0))
    ok = [r for r in rows if r[2] <= budget]
    if ok:
        return max(ok, key=lambda r: (r[1], r[0]))[0]
    return max(rows, key=lambda r: (-r[2], r[1], r[0]))[0]


def bootstrap(per_clip: dict, n: int = 5000, seed: int = 0) -> dict:
    """Clip bootstrap of the pooled held-out right / wrong shares (13 clips resampled)."""
    rng = np.random.default_rng(seed)
    clips = list(per_clip)
    R = np.array([Counter(per_clip[s]["per_return"])["right"] for s in clips], float)
    Wr = np.array([Counter(per_clip[s]["per_return"])["wrong"] for s in clips], float)
    N = np.array([sum(Counter(per_clip[s]["per_return"])[k] for k in ("right", "wrong", "lost")) for s in clips], float)
    idx = rng.integers(0, len(clips), size=(n, len(clips)))
    rs, ws = R[idx].sum(axis=1) / N[idx].sum(axis=1), Wr[idx].sum(axis=1) / N[idx].sum(axis=1)
    q = lambda v: [round(float(x), 3) for x in np.quantile(v, [0.05, 0.5, 0.95])]  # noqa: E731
    return {"right_share_q05_q50_q95": q(rs), "wrong_share_q05_q50_q95": q(ws)}


def report(raw: dict) -> dict:
    folds = raw["folds"]
    rep = {"margins": {v: list(VARIANTS[v]["margins"]) for v in VARIANTS}, "lambda": LAMBDA, "variants": {}}
    for variant in VARIANTS:
        V = {"points": {}, "points_cap_rule": {}, "fixed_margin_heldout": {}, "fixed_margin_all13_insample": {}}
        for m in VARIANTS[variant]["margins"]:
            V["fixed_margin_heldout"][str(m)] = pool(folds[c]["res"][(variant, m, c)] for c in data.DEV_CLIPS)
            V["fixed_margin_all13_insample"][str(m)] = pool(folds[ALL]["res"][(variant, m, c)] for c in data.DEV_CLIPS)
        for rule, table, chooser, par in (("lam", "points", choose, LAMBDA), ("cap", "points_cap_rule", choose_cap, CAPS)):
            for point, value in par.items():
                chosen = {c: chooser(folds[c], variant, [s for s in data.DEV_CLIPS if s != c], value) for c in data.DEV_CLIPS}
                held = {c: folds[c]["res"][(variant, chosen[c], c)] for c in data.DEV_CLIPS}
                m_all = chooser(folds[ALL], variant, list(data.DEV_CLIPS), value)
                ins = {c: folds[ALL]["res"][(variant, m_all, c)] for c in data.DEV_CLIPS}
                V[table][point] = {
                    "margin_per_fold": chosen, "margin_all13": m_all,
                    "heldout": pool(held.values()), "insample_all13": pool(ins.values()),
                    "heldout_per_clip": {c: {**dict(Counter(held[c]["per_return"])), "links": held[c]["lk"], "impure": held[c]["impure"], "rows": held[c]["rows"]} for c in held},
                    "bootstrap": bootstrap(held),
                }
        V["pair_model"] = {c: folds[c]["models"][variant] for c in folds}
        rep["variants"][variant] = V
    return rep


def fmt(p: dict) -> str:
    return f"{p['right']} / {p['wrong']} / {p['lost']} | {p['links_right']} / {p['links_wrong']} / {p['links_unknown']} | {100 * p['impure_row_share']:.2f}%"


def tables(rep: dict) -> str:
    L = ["# fuse: leave-one-clip-out results (137 scored returns, 13 dev clips)", "",
         "Columns: right / wrong / lost | links right / wrong / unknown | impure rows.",
         "Baselines (score.py): split 0 / 4 / 133 | 0 / 0 / 0 | 2.61%; plugin 30 / 14 / 93 | 33 / 10 / 50 | 3.08%; mutual 26 / 6 / 105 | 60 / 7 / 2 | 3.03%.", ""]
    for variant, V in rep["variants"].items():
        L += [f"## {variant}", "", "Operating points (margin chosen per fold on the 12 fit clips by J = right - lam * (wrong + wrong links)):", "",
              "| point | lam | HELD-OUT | in-sample (all 13) | margins chosen per fold | margin on all 13 |", "|---|---|---|---|---|---|"]
        for point, P in V["points"].items():
            cnt = Counter(P["margin_per_fold"].values())
            L.append(f"| {point} | {LAMBDA[point]:g} | {fmt(P['heldout'])} | {fmt(P['insample_all13'])} | {', '.join(f'{n}x {m}' for m, n in sorted(cnt.items()))} | {P['margin_all13']} |")
        L += ["", "Secondary rule (most right with wrong <= floor(cap * scored_fit / 137) on the fit clips):", "",
              "| point | cap | HELD-OUT | margins chosen per fold |", "|---|---|---|---|"]
        for point, P in V["points_cap_rule"].items():
            cnt = Counter(P["margin_per_fold"].values())
            L.append(f"| {point} | {CAPS[point]} | {fmt(P['heldout'])} | {', '.join(f'{n}x {m}' for m, n in sorted(cnt.items()))} |")
        L += ["", "Fixed margin (nothing chosen; HELD-OUT = each clip with the weights fitted without it; in-sample = weights of all 13):", "",
              "| margin | HELD-OUT | in-sample (all 13) |", "|---|---|---|"]
        for m in VARIANTS[variant]["margins"]:
            L.append(f"| {m:.2f} | {fmt(V['fixed_margin_heldout'][str(m)])} | {fmt(V['fixed_margin_all13_insample'][str(m)])} |")
        L.append("")
    V = rep["variants"]["fuse"]
    L += ["## fuse: held-out result per clip (right / wrong / lost, links right / wrong / unknown)", "",
          "| clip | " + " | ".join(V["points"]) + " |", "|---|" + "---|" * len(V["points"])]
    for c in data.DEV_CLIPS:
        cells = []
        for P in V["points"].values():
            x = P["heldout_per_clip"][c]
            cells.append(f"{x.get('right', 0)} / {x.get('wrong', 0)} / {x.get('lost', 0)}, {x['links']['right']} / {x['links']['wrong']} / {x['links']['unknown']} (margin {P['margin_per_fold'][c]})")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "Clip bootstrap of the held-out shares (13 clips resampled, 5% / 50% / 95%):", ""]
    for point, P in V["points"].items():
        L.append(f"- {point}: right share {P['bootstrap']['right_share_q05_q50_q95']}, wrong share {P['bootstrap']['wrong_share_q05_q50_q95']}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_fold, [*data.DEV_CLIPS, ALL]):
            res[r["fold"]] = r
            m = r["models"]["fuse"]
            print(r["fold"], r["seconds"], "s; fuse pair model", dict(zip(m["cols"], np.round(m["w"], 3).tolist())), "bias", round(m["b"], 3), flush=True)
    raw = {"folds": res}
    with open(HERE / "cache" / "loco_raw.pkl", "wb") as fh:
        pickle.dump(raw, fh)
    rep = report(raw)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "loco_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (HERE / "out" / "tradeoff.md").write_text(tables(rep), encoding="utf-8")
    params = {}
    for c in res:
        params[c] = {"pair_model": res[c]["models"]["fuse"],
                     "margin": {p: (rep["variants"]["fuse"]["points"][p]["margin_all13"] if c == ALL else rep["variants"]["fuse"]["points"][p]["margin_per_fold"][c]) for p in LAMBDA},
                     "cuts": "m_cuts frozen weights" if c == ALL else f"m_cuts loco_params.json[{c}]", "p_cut": prep.P_CUT,
                     "position": "m_space/params_all13.json" if c == ALL else f"m_space/params_cache/excl_{c.split('-')[1]}.json",
                     "floor": FLOOR, "t0": T0, "t_abs": T_ABS}
    (HERE / "params_loco.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    print(tables(rep))
    print("done", round(time.time() - t0, 1), "s")
