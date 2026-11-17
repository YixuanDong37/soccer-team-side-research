"""Leave-one-clip-out fit and held-out evaluation of m_keep's rules.

    python -I loco.py            -> prints everything, writes out/*.json

Labels are used here only through reid_lab.score (score_clip), and only to
  - choose thresholds on the 12 fit clips of a fold, which are then applied unchanged to the
    held-out clip (the 13 held-out link sets are pooled into one METHOD.json and scored), and
  - score fixed-threshold trade-off tables (no fitting; marked in-sample where a choice was made).
The linking code (features.py, method.py) never sees a label.

Objective of every fit, for penalty lam:   J = right - lam * (wrong + links_wrong)
(returns right / wrong as scored by score.py; links_wrong = links between two pieces whose
annotated majority persons differ).  Ties: the median of the maximising grid values for the keep
threshold; for the appearance step the first maximiser in the order of APP_GRID (stricter first).

Fits (per fold, on the 12 fit clips):
  (a)  keep alone: t_keep on T_GRID                                              (1 parameter)
  (b0) keep (t_keep of (a)) + the mutual baseline on appearance_match.py's own candidate pairs
       (margin 0.05, no bar: the baseline's values, not tuned)                   (1 parameter)
  (b1) keep (t_keep of (a)) + mutual best on m_keep's group candidates with the baseline's
       values (margin 0.05, no bar, gap <= 10 s; not tuned)                      (1 parameter)
  (b2) joint: (t_keep, margin, bar, max_gap) on TB_GRID x APP_GRID               (4 parameters)
  (b3) as (b1) plus an absolute bar on the cosine, fitted on BARS                (2 parameters)
  (a-long)  keep only joins whose two pieces both have >= n_min boxes: (t_keep, n_min) on
            T_GRID x N_GRID                                                      (2 parameters)
  (b3-long) (a-long) + the appearance step of (b3)                               (3 parameters)
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
from reid_lab import data, score  # noqa: E402

import method  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

T_GRID = [round(0.70 + 0.01 * i, 2) for i in range(28)]          # 0.70 .. 0.97
MARGINS = [0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
BARS = [0.93, 0.90, 0.85, -1.0]
GAPS = [10.0, None]
APP_GRID = [(m, b, g) for m in MARGINS for g in GAPS for b in BARS]  # stricter first
TB_GRID = [round(0.80 + 0.01 * i, 2) for i in range(18)]        # 0.80 .. 0.97
N_GRID = [5, 10, 15, 20, 25]
LAMS = (1.0, 2.0, 3.0, 6.0)


@lru_cache(maxsize=None)
def _score_links(seq: str, links: tuple) -> dict:
    s = score.score_clip(seq, [list(x) for x in links])
    return {"right": s["returns"]["right"], "wrong": s["returns"]["wrong"], "lost": s["returns"]["lost"],
            "links": s["links"]["n"], "links_right": s["links"]["right"], "links_wrong": s["links"]["wrong"],
            "links_unknown": s["links"]["unknown"], "impure_rows": s["impure_rows"], "tracked_rows": s["tracked_rows"]}  # fmt: skip


@lru_cache(maxsize=None)
def keep(seq: str, t: float, rule: str = "cos", n_min: int = 1) -> tuple:
    return tuple(tuple(x) for x in method.keep_links(seq, t, rule, n_min))


@lru_cache(maxsize=None)
def _cands(seq: str, kept: tuple, gap) -> tuple:
    return tuple(method.group_candidates(seq, [list(x) for x in kept], gap))


@lru_cache(maxsize=None)
def _app(seq: str, kept: tuple, margin: float, bar: float, gap) -> tuple:
    return tuple(tuple(x) for x in method.decide(_cands(seq, kept, gap), bar, margin))


@lru_cache(maxsize=None)
def comb_lab(seq: str, t: float, margin: float = 0.05, bar: float = -1.0) -> tuple:
    k = keep(seq, t)
    return k + tuple(tuple(x) for x in method.decide(method.lab_candidates(seq, [list(x) for x in k]), bar, margin))


def comb(seq: str, t: float, margin: float, bar: float, gap, rule: str = "cos", n_min: int = 1) -> tuple:
    k = keep(seq, t, rule, n_min)
    return k + _app(seq, k, margin, bar, gap)


def pooled(per_clip: dict) -> dict:
    tot = {k: sum(v[k] for v in per_clip.values()) for k in next(iter(per_clip.values()))}
    tot["impure_row_share"] = round(tot["impure_rows"] / tot["tracked_rows"], 5)
    return tot


def J(tot: dict, lam: float) -> float:
    return tot["right"] - lam * (tot["wrong"] + tot["links_wrong"])


def fit_keep(clips, lam: float, rule: str = "cos", grid=T_GRID):
    js = [J(pooled({s: _score_links(s, keep(s, t, rule)) for s in clips}), lam) for t in grid]
    best = max(js)
    arg = [t for t, j in zip(grid, js) if j == best]
    return arg[(len(arg) - 1) // 2] if len(arg) % 2 else arg[len(arg) // 2 - 1], best  # median (lower middle)


def fit_app(clips, t_of: dict, lam: float, rule: str = "cos"):
    """t_of: the keep threshold to use on each fit clip (the fold's own fitted value)."""
    best, arg = -np.inf, None
    for m, b, g in APP_GRID:
        j = J(pooled({s: _score_links(s, comb(s, t_of[s], m, b, g, rule)) for s in clips}), lam)
        if j > best:
            best, arg = j, (m, b, g)
    return arg, best


def _median(xs):
    return xs[(len(xs) - 1) // 2]


def fit_keep_long(clips, lam: float):
    """(t_keep, n_min): ties -> the median maximising n_min, then its median maximising t_keep."""
    per_n = {}
    for n in N_GRID:
        js = [J(pooled({s: _score_links(s, keep(s, t, "cos", n)) for s in clips}), lam) for t in T_GRID]
        top = max(js)
        per_n[n] = (top, _median([t for t, j in zip(T_GRID, js) if j == top]))
    best = max(v[0] for v in per_n.values())
    n = _median([n for n in N_GRID if per_n[n][0] == best])
    return (per_n[n][1], n), best


def fit_bar(clips, t: float, lam: float, n_min: int = 1):
    """The absolute bar of the appearance step (margin 0.05, gap <= 10 s fixed); ties: stricter."""
    best, arg = -np.inf, None
    for b in BARS:
        j = J(pooled({s: _score_links(s, comb(s, t, 0.05, b, 10.0, "cos", n_min)) for s in clips}), lam)
        if j > best:
            best, arg = j, b
    return arg, best


def fit_joint(clips, lam: float):
    """(t_keep, margin, bar, max_gap) maximising J on the clips.  Ties: first (margin, gap, bar) in
    APP_GRID order (stricter first), then the median of that setting's maximising t_keep."""
    best, arg = -np.inf, None
    for m, b, g in APP_GRID:
        js = [J(pooled({s: _score_links(s, comb(s, t, m, b, g)) for s in clips}), lam) for t in TB_GRID]
        top = max(js)
        if top > best:
            ts = [t for t, j in zip(TB_GRID, js) if j == top]
            best, arg = top, (ts[(len(ts) - 1) // 2], m, b, g)
    return arg, best


def fmt(p: dict) -> str:
    return (f"right {p['right']:3d}  wrong {p['wrong']:2d}  lost {p['lost']:3d} | links {p['links']:3d}: right {p['links_right']:3d} "
            f"wrong {p['links_wrong']:2d} unknown {p['links_unknown']:2d} | impure {p['impure_row_share']:.5f}")


def write_method(name: str, links_of: dict, note: str, extra: dict | None = None) -> dict:
    for s, links in links_of.items():
        method.check_no_conflict(s, [list(x) for x in links])
    m = {"name": name, "note": note, **(extra or {}), "clips": {s: {"links": [list(x) for x in links_of[s]]} for s in data.DEV_CLIPS}}
    (OUT / f"{name}.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    res = score.evaluate(m["clips"])  # the official scorer on the written file's content
    (OUT / f"{name}.score.json").write_text(json.dumps({"name": name, **res}, indent=1) + "\n", encoding="utf-8")
    return res["pooled"]


def main() -> None:
    clips = data.DEV_CLIPS
    report = {}
    print("=== baselines (score.py) ===")
    for b in ("split", "plugin", "mutual"):
        p = score.evaluate(score.baseline_links(b))["pooled"]
        print(f"{b:8s} right {p['right']:3d}  wrong {p['wrong']:2d}  lost {p['lost']:3d} | links {p['links']:3d}: right {p['links_right']:3d} wrong {p['links_wrong']:2d} unknown {p['links_unknown']:2d} | impure {p['impure_row_share']:.5f}")

    print()
    print("=== (a) keep alone: trade-off over t_keep (fixed thresholds, all 13 clips, nothing fitted) ===")
    table_a = []
    for t in T_GRID:
        p = pooled({s: _score_links(s, keep(s, t)) for s in clips})
        table_a.append({"t_keep": t, **p})
        print(f"t_keep {t:.2f}  {fmt(p)}")
    report["tradeoff_keep"] = table_a

    for lam in LAMS:
        print()
        print(f"=== LOCO, lam = {lam:g}:  J = right - {lam:g} * (wrong + links_wrong) ===")
        fold_t, fold_joint, fold_bar, fold_long = {}, {}, {}, {}
        L = {k: {} for k in ("keep", "comb_lab", "comb_groups_default", "comb_groups_bar", "comb_joint", "keep_long", "comb_long_bar")}
        for fit, held in data.loco_folds():
            t, _ = fit_keep(fit, lam)
            fold_t[held] = t
            L["keep"][held] = keep(held, t)
            L["comb_lab"][held] = comb_lab(held, t)
            L["comb_groups_default"][held] = comb(held, t, 0.05, -1.0, 10.0)
            fold_bar[held], _ = fit_bar(fit, t, lam)
            L["comb_groups_bar"][held] = comb(held, t, 0.05, fold_bar[held], 10.0)
            (tl, nl), _ = fit_keep_long(fit, lam)
            bl, _ = fit_bar(fit, tl, lam, nl)
            fold_long[held] = (tl, nl, bl)
            L["keep_long"][held] = keep(held, tl, "cos", nl)
            L["comb_long_bar"][held] = comb(held, tl, 0.05, bl, 10.0, "cos", nl)
            (tj, m, b, g), _ = fit_joint(fit, lam)
            fold_joint[held] = (tj, m, b, g)
            L["comb_joint"][held] = comb(held, tj, m, b, g)
        print("fold t_keep (keep-only fit):", fold_t)
        print("fold bar (b3, given the fold's t_keep):", fold_bar)
        print("fold (t_keep, margin, bar, max_gap) (joint fit):", fold_joint)
        print("fold (t_keep, n_min, bar) (long variants):", fold_long)
        notes = {"keep": "(a) selective keep", "comb_lab": "(b0) keep + mutual baseline on appearance_match candidates (margin 0.05, no bar)",
                 "comb_groups_default": "(b1) keep + mutual best on group candidates (margin 0.05, no bar, gap <= 10 s)",
                 "comb_groups_bar": "(b3) keep + mutual best on group candidates (margin 0.05, gap <= 10 s) with a fitted absolute bar",
                 "comb_joint": "(b2) keep + mutual best on group candidates, 4 parameters fitted jointly",
                 "keep_long": "(a-long) selective keep of joins between pieces of >= n_min boxes",
                 "comb_long_bar": "(b3-long) keep-long + mutual best on group candidates (margin 0.05, gap <= 10 s) with a fitted bar"}
        rep = {"fold_t_keep": fold_t, "fold_bar": fold_bar, "fold_joint": {k: list(v) for k, v in fold_joint.items()},
               "fold_long": {k: list(v) for k, v in fold_long.items()}}
        for k in L:
            extra = {"fold_t_keep": fold_t} if k != "comb_joint" else {"fold_tkeep_margin_bar_maxgap": {c: list(v) for c, v in fold_joint.items()}}
            if k == "comb_groups_bar":
                extra["fold_bar"] = fold_bar
            if k in ("keep_long", "comb_long_bar"):
                extra = {"fold_tkeep_nmin_bar": {c: list(v) for c, v in fold_long.items()}}
            p = write_method(f"{k}_loco_lam{lam:g}", L[k], f"{notes[k]}; LOCO HELD-OUT links (each clip linked with parameters fitted on the other 12); objective lam={lam:g}", extra)
            rep[f"{k}_heldout"] = p
            print(f"HELD-OUT  {notes[k]}")
            print(f"          {fmt(p)}")
        # in-sample: the same fits on all 13 clips (the deployable parameters)
        t_all, _ = fit_keep(clips, lam)
        (tj, m, b, g), _ = fit_joint(clips, lam)
        bar_all, _ = fit_bar(clips, t_all, lam)
        (tl, nl), _ = fit_keep_long(clips, lam)
        bl, _ = fit_bar(clips, tl, lam, nl)
        ins = {"keep": ({s: keep(s, t_all) for s in clips}, {"t_keep": t_all}),
               "comb_lab": ({s: comb_lab(s, t_all) for s in clips}, {"t_keep": t_all, "margin": 0.05, "bar": -1.0}),
               "comb_groups_default": ({s: comb(s, t_all, 0.05, -1.0, 10.0) for s in clips}, {"t_keep": t_all, "margin": 0.05, "bar": -1.0, "max_gap_s": 10.0}),
               "comb_groups_bar": ({s: comb(s, t_all, 0.05, bar_all, 10.0) for s in clips}, {"t_keep": t_all, "margin": 0.05, "bar": bar_all, "max_gap_s": 10.0}),
               "comb_joint": ({s: comb(s, tj, m, b, g) for s in clips}, {"t_keep": tj, "margin": m, "bar": b, "max_gap_s": g}),
               "keep_long": ({s: keep(s, tl, "cos", nl) for s in clips}, {"t_keep": tl, "n_min": nl}),
               "comb_long_bar": ({s: comb(s, tl, 0.05, bl, 10.0, "cos", nl) for s in clips}, {"t_keep": tl, "n_min": nl, "margin": 0.05, "bar": bl, "max_gap_s": 10.0})}
        for k, (links, par) in ins.items():
            p = write_method(f"{k}_all13_lam{lam:g}", links, f"{notes[k]}; fitted on all 13 clips (IN-SAMPLE); objective lam={lam:g}", {"params": par})
            rep[f"{k}_insample"] = {"params": par, **p}
            print(f"IN-SAMPLE {notes[k]}  {par}")
            print(f"          {fmt(p)}")
        report[f"lam{lam:g}"] = rep

    print()
    print("=== (b) combined: trade-off over the margin (fixed: t_keep 0.86, bar 0.90; all 13 clips, nothing fitted) ===")
    table_b = []
    for g in GAPS:
        for m in MARGINS:
            p = pooled({s: _score_links(s, comb(s, 0.86, m, 0.90, g)) for s in clips})
            table_b.append({"t_keep": 0.86, "bar": 0.90, "max_gap_s": g, "margin": m, **p})
            print(f"max_gap {str(g):5s} margin {m:.2f}  {fmt(p)}")
    report["tradeoff_comb_margin"] = table_b
    print()
    print("=== (b) combined: trade-off over t_keep (fixed: margin 0.03, bar 0.90, max_gap 10 s; all 13 clips, nothing fitted) ===")
    table_c = []
    for t in (2.0, 0.97, 0.95, 0.94, 0.92, 0.90, 0.88, 0.86, 0.85, 0.84, 0.82, 0.80, 0.78, 0.75, 0.70, -1.0):
        p = pooled({s: _score_links(s, comb(s, t, 0.03, 0.90, 10.0)) for s in clips})
        table_c.append({"t_keep": t, "margin": 0.03, "bar": 0.90, "max_gap_s": 10.0, **p})
        print(f"t_keep {t:5.2f}  {fmt(p)}" + ("   <- no plugin join kept (appearance only)" if t > 1 else "   <- every plugin join kept" if t < 0 else ""))
    report["tradeoff_comb_tkeep"] = table_c
    (OUT / "loco_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
