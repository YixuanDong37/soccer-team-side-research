"""fuse / sensitivity.py -- how much the held-out result depends on the constants that were set by
hand (LABELS: pair-model fit on the fit clips of each fold, and scoring). One constant is changed
at a time; the margin is FIXED (no choice), weights are LOCO as in fit_loco.py.

    python -I sensitivity.py [workers]      -> out/sensitivity.md

Rows: base (the method); no appearance floor; no "nobody" rival; bonus x 0.5 / x 1.5; m_cuts at
P >= 0.5 / 0.9; pair model fitted on firm labels (>= 5 matched rows) instead of weak ones; pair
model with the raw position score instead of its log-odds.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data  # noqa: E402
import evidence  # noqa: E402
import fit_loco  # noqa: E402
import linker  # noqa: E402
import pairmodel  # noqa: E402
import prep  # noqa: E402
from method import bonus_of, pos_params  # noqa: E402

HERE = Path(__file__).resolve().parent
MARGINS = (0.2, 0.25, 0.5, 0.8, 0.9)
BASE = dict(floor=fit_loco.FLOOR, t0=fit_loco.T0, scale=1.0, p_cut=prep.P_CUT, min_rows=1, cols=fit_loco.FUSE_COLS)
ROWS = {
    "base": {},
    "no appearance floor": {"floor": None},
    "floor 0.4": {"floor": 0.4},
    "no 'nobody' rival": {"t0": None},
    "bonus x 0.5": {"scale": 0.5},
    "bonus x 1.5": {"scale": 1.5},
    "cuts at P >= 0.5": {"p_cut": 0.5},
    "cuts at P >= 0.9": {"p_cut": 0.9},
    "pair model on firm labels (>= 5 rows)": {"min_rows": 5},
    "pair model with raw position score": {"cols": ["cos_w", "sink", "plug"]},
}


def build_tables(job):
    seq, cuts = job
    prep.app_table(seq, cuts)
    prep.space_tables(seq, cuts)
    return seq


def run_fold(fold: str):
    import pandas as pd

    import fuse_labels as fitlabels   # LABELS: fit clips only (pair model) and scoring
    from reid_lab import score

    pp = pos_params(fold)
    fit_clips = [s for s in data.DEV_CLIPS if s != fold]
    out = {}
    cache = {}
    for name, change in ROWS.items():
        c = {**BASE, **change}
        key = (c["p_cut"], c["min_rows"], tuple(c["cols"]))
        if key not in cache:
            cuts = {s: prep.cuts_for(s, fold, c["p_cut"]) for s in data.DEV_CLIPS}
            tabs = {s: evidence.pair_table(s, cuts[s], pp) for s in data.DEV_CLIPS}
            lab = pd.concat([fitlabels.label_pairs(tabs[s], s, cuts[s], c["min_rows"]) for s in fit_clips], ignore_index=True)
            cache[key] = (cuts[fold], tabs[fold], pairmodel.fit(lab, c["cols"]))
        cut, tab, model = cache[key]
        bonus = {k: v * c["scale"] for k, v in bonus_of(tab, model).items()}
        for m in MARGINS:
            links = linker.agglomerate(fold, cut, bonus, t_abs=fit_loco.T_ABS, margin=m, floor=c["floor"], t0=c["t0"])
            r = score.score_clip(fold, links, [list(x) for x in cut])
            out[(name, m)] = {"per_return": r["per_return"], "lk": r["links"], "impure": r["impure_rows"], "rows": r["tracked_rows"]}
    return fold, out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t0 = time.time()
    jobs = {}
    for p in sorted({c.get("p_cut", prep.P_CUT) for c in ROWS.values()}):
        for seq in data.DEV_CLIPS:
            for fold in data.DEV_CLIPS:
                jobs[(seq, prep.cuts_for(seq, fold, p))] = None
    prep.save_cuts()
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(build_tables, list(jobs)))          # every table exists before the folds read them
        print(len(jobs), "tables ready", round(time.time() - t0, 1), "s", flush=True)
        res = dict(ex.map(run_fold, data.DEV_CLIPS))
    L = ["# fuse: sensitivity to the hand-set constants (held-out, fixed margin, LOCO weights)", "",
         "Cells: right / wrong / lost | links right / wrong / unknown | impure rows. One constant changed at a time.", "",
         "| setting | " + " | ".join(f"margin {m}" for m in MARGINS) + " |", "|---|" + "---|" * len(MARGINS)]
    for name in ROWS:
        L.append(f"| {name} | " + " | ".join(fit_loco.fmt(fit_loco.pool(res[c][(name, m)] for c in data.DEV_CLIPS)) for m in MARGINS) + " |")
    text = "\n".join(L) + "\n"
    (HERE / "out" / "sensitivity.md").write_text(text, encoding="utf-8")
    print(text)
    print("done", round(time.time() - t0, 1), "s")
