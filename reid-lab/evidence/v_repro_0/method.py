"""fuse / method.py -- the fusion method, LABEL-FREE apply path. Writes the METHOD.json files for
reid_lab/score.py.

    python -I method.py                 # all files below
    python -I method.py heldout         # METHOD_strict.json, METHOD_balanced.json, METHOD_recall.json
    python -I method.py frozen          # METHOD_frozen_strict.json, ... (in-sample on the dev clips)
    python -I method.py margin 0.5      # METHOD_margin_0.50.json: one fixed margin, LOCO weights

What it does for one clip (nothing here reads an annotation or reid_lab.score):
  1. cuts       m_cuts' switch detector (P >= 0.7) cuts pieces where the plugin's track moved onto
                another person; everything below works on the cut pieces (score.py's sub-piece ids).
  2. evidence   per pair of pieces (a earlier, b later): m_appear's whitened team-centred cosine;
                m_space's position assignment score on the cut pieces; "the plugin joined them".
  3. bonus      (pair-model weights) position and plugin evidence expressed in cosine units.
  4. linker     m_appear's constrained agglomeration on  cosine + junction bonus  (linker.py) with
                appearance floor 0.3, a "nobody" rival at 0.5 and ONE threshold, the margin.
Every number comes from params_loco.json (written by fit_loco.py):
  "heldout": a dev clip is processed with the numbers of ITS fold (all fitted without that clip);
  "frozen":  the numbers fitted on all 13 dev clips (key "all13"): the method to apply to a new clip.
"""
from __future__ import annotations

import json
import sys
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
ALL = "all13"
POINTS = ("strict", "balanced", "recall")


def pos_params(fold: str | None) -> dict:
    """m_space's position weights: fitted without `fold`, or (fold None) on all 13 clips."""
    if fold is None:
        return json.load(open(prep.WORK + "m_space/params_all13.json", encoding="utf-8"))
    return json.load(open(prep.WORK + f"m_space/params_cache/excl_{fold.split('-')[1]}.json", encoding="utf-8"))


def bonus_of(table, model) -> dict:
    """{(a, b): bonus in cosine units} = (pair-model weights of the non-appearance inputs / weight of
    cos_w) . inputs, for the time-ordered pairs with a positive value."""
    cols, w = model["cols"], np.asarray(model["w"])
    wc = w[cols.index("cos_w")]
    rest = [c for c in cols if c != "cos_w"]
    v = pairmodel.design(table, rest) @ (w[[cols.index(c) for c in rest]] / wc)
    keep = v > 0
    return {(int(a), int(b)): float(x) for a, b, x in zip(table.a[keep], table.b[keep], v[keep])}


def load_params() -> dict:
    return json.loads((HERE / "params_loco.json").read_text(encoding="utf-8"))


def clip_method(seq: str, margin: float, par: dict, fold: str | None) -> dict:
    """{"links": [...], "cuts": [...]} of one clip for one margin with the numbers of `par`
    (fold: whose m_cuts / m_space weights to use; None = the frozen all-13 ones)."""
    cuts = prep.cuts_for(seq, fold)
    table = evidence.pair_table(seq, cuts, pos_params(fold))
    bonus = bonus_of(table, par["pair_model"])
    links = linker.agglomerate(seq, cuts, bonus, t_abs=par["t_abs"], margin=margin, floor=par["floor"], t0=par["t0"])
    return {"links": links, "cuts": [list(x) for x in cuts]}


def build(mode: str, point: str | None = None, margin: float | None = None) -> dict:
    params = load_params()
    clips = {}
    for seq in data.DEV_CLIPS:
        key, fold = (seq, seq) if mode == "heldout" else (ALL, None)
        par = params[key]
        m = margin if margin is not None else par["margin"][point]
        clips[seq] = clip_method(seq, m, par, fold)
    what = f"point {point}" if margin is None else f"fixed margin {margin}"
    how = ("LOCO held-out: every clip with the numbers fitted without it" if mode == "heldout"
           else "frozen all-13 numbers: IN-SAMPLE on the dev clips")
    return {"name": f"fuse {what} ({how})", "clips": clips}


def write(method: dict, name: str) -> Path:
    path = HERE / name
    path.write_text(json.dumps(method), encoding="utf-8")
    n = sum(len(c["links"]) for c in method["clips"].values())
    k = sum(len(c["cuts"]) for c in method["clips"].values())
    print(f"wrote {path.as_posix()}: {n} links, {k} cuts")
    return path


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "heldout"):
        for p in POINTS:
            write(build("heldout", p), f"METHOD_{p}.json")
    if what in ("all", "frozen"):
        for p in POINTS:
            write(build("frozen", p), f"METHOD_frozen_{p}.json")
    if what == "margin":
        m = float(sys.argv[2])
        write(build("heldout", margin=m), f"METHOD_margin_{m:.2f}.json")
