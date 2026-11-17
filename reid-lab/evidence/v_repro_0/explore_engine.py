"""fuse / explore_engine.py -- EXPLORATION engine of one leave-one-clip-out fold (LABELS: fitting on the fit
clips and scoring). Kept for the record: it searched m_appear's threshold range, which turned out to be the wrong
range for the fused similarity (see README). The final run is fit_loco.py.

For a fold (held-out clip c) EVERYTHING that carries fitted numbers has been fitted without c:
  cuts          m_cuts' weights of loco_params.json[c]             (prep.cuts_for(s, c))
  position      m_space's weights of params_cache/excl_<c>.json    (fitted on the other 12 clips)
  pair model    fitted here on the 12 fit clips (pairmodel.fit, weak labels: >= 1 matched row)
and is applied with those numbers to all 13 clips: to the 12 fit clips (where the grid of linker
settings is then scored to choose the operating points) and, unchanged, to the held-out clip c.

    used through t3_onefold.py (one fold -> cache/onefold_<fold>.pkl)

Variants (what enters the similarity besides m_appear's cosine):
  app        nothing (m_appear's rule on the cut pieces)
  pos        + position bonus (m_space's assignment score at the junction)
  posplug    + position bonus + plugin-join bonus            <- the fusion method
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
from reid_lab import data  # noqa: E402
import evidence  # noqa: E402
import linker  # noqa: E402
import pairmodel  # noqa: E402
import prep  # noqa: E402

HERE = Path(__file__).resolve().parent
T_ABS = (0.4, 0.5, 0.6, 0.7, 0.8)
MARGIN = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
GRID = [(t, m) for t in T_ABS for m in MARGIN]
VARIANTS = {"app": None, "pos": ["cos_w", "sink_lo"], "posplug": ["cos_w", "sink_lo", "plug"]}
MIN_ROWS_FIT = 1          # weak labels for the pair model: majority person of >= 1 matched row


def pos_params(fold: str | None) -> dict:
    if fold is None:
        return json.load(open(prep.WORK + "m_space/params_all13.json", encoding="utf-8"))
    return json.load(open(prep.WORK + f"m_space/params_cache/excl_{fold.split('-')[1]}.json", encoding="utf-8"))


def bonus_of(table, model) -> dict:
    """{(a, b): bonus in cosine units} from a pair model with inputs cos_w + evidence columns."""
    cols, w = model["cols"], np.asarray(model["w"])
    wc = w[cols.index("cos_w")]
    rest = [c for c in cols if c != "cos_w"]
    X = pairmodel.design(table, rest)
    v = X @ (w[[cols.index(c) for c in rest]] / wc)
    keep = v > 0
    return {(int(a), int(b)): float(x) for a, b, x in zip(table.a[keep], table.b[keep], v[keep])}


def run_fold(fold: str):
    import fuse_labels as fitlabels   # labels: fit clips (pair model) and scoring
    from reid_lab import score

    t0 = time.time()
    pp = pos_params(fold)
    cuts = {s: prep.cuts_for(s, fold) for s in data.DEV_CLIPS}
    tabs = {s: evidence.pair_table(s, cuts[s], pp) for s in data.DEV_CLIPS}
    fit_clips = [s for s in data.DEV_CLIPS if s != fold]
    import pandas as pd
    lab = pd.concat([fitlabels.label_pairs(tabs[s], s, cuts[s], MIN_ROWS_FIT) for s in fit_clips], ignore_index=True)
    out = {"fold": fold, "cuts": {s: [list(x) for x in cuts[s]] for s in data.DEV_CLIPS}, "models": {}, "res": {}}
    for variant, cols in VARIANTS.items():
        model = pairmodel.fit(lab, cols) if cols else None
        out["models"][variant] = model
        for s in data.DEV_CLIPS:
            bonus = bonus_of(tabs[s], model) if model else None
            for gi, (t_abs, margin) in enumerate(GRID):
                links = linker.agglomerate(s, cuts[s], bonus, t_abs=t_abs, margin=margin)
                r = score.score_clip(s, links, out["cuts"][s])
                out["res"][(variant, gi, s)] = {"links": links, "per_return": r["per_return"], "lk": r["links"],
                                                "impure": r["impure_rows"], "rows": r["tracked_rows"]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_fold, data.DEV_CLIPS):
            res[r["fold"]] = r
            print(r["fold"], r["seconds"], "s;", {v: (np.round(m["w"], 2).tolist(), round(m["b"], 2)) if m else None for v, m in r["models"].items()}, flush=True)
    with open(HERE / "cache" / "loco_raw.pkl", "wb") as fh:
        pickle.dump({"grid": GRID, "variants": list(VARIANTS), "folds": res}, fh)
    print("done", round(time.time() - t0, 1), "s")
