"""fuse_r1 / method.py -- the fusion method of round 1, LABEL-FREE apply path. Writes the METHOD.json
files for reid_lab/score.py.

    python -I method.py                 # all files below
    python -I method.py heldout         # METHOD_strict.json, METHOD_balanced.json, METHOD_recall.json
    python -I method.py frozen          # METHOD_frozen_<point>.json   (in-sample on the dev clips)
    python -I method.py fixed           # METHOD_fixed_<point>.json    (the FROZEN setting of a point applied to
                                        #   every clip with the weights fitted without that clip)
    python -I method.py cand 0.5 on     # METHOD_cand_m0.50_on.json: one fixed setting, LOCO weights
    python -I method.py cutsonly        # METHOD_cuts_only.json: m_cuts' cuts and no link (the baseline of the linking step)

What it does for one clip (nothing here reads an annotation or reid_lab.score):
  1. cuts      m_cuts' switch detector (P >= 0.7): pieces are cut where the plugin's track moved onto
               another person; everything below works on the cut pieces (score.py's sub-piece ids).
  2. tiers     a (sub-)piece with at least 5 boxes is SOLID, otherwise SHORT.
  3. core      evidence per pair of SOLID pieces: m_appear's whitened team-centred cosine; m_space's
               position assignment score computed on the solid pieces only; "the plugin joined them".
               linker2 stage 1 = fuse's agglomeration on cosine + junction bonus with appearance
               floor 0.3, "nobody" rival 0.5 and ONE threshold, the margin.
  4. leaves    (when the setting says so) a SHORT piece is attached to at most one group, on its own
               appearance + position evidence with the same rule; it never connects two groups.
Every number comes from params_loco.json (written by fit_loco.py):
  "heldout": a dev clip is processed with the numbers of ITS fold (all fitted without that clip);
  "frozen":  the numbers fitted on all 13 dev clips (key "all13"): the method to apply to a new clip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse_r1")
from reid_lab import data  # noqa: E402
import evidence  # noqa: E402
import linker2  # noqa: E402
import pairmodel  # noqa: E402
import prep  # noqa: E402

HERE = Path(__file__).resolve().parent
ALL = "all13"
POINTS = ("strict", "balanced", "recall")
N_SOLID = prep.N_SOLID
LEAF_COLS = ("sink_lo",)          # leaves get the position bonus only (PREREG A3)


def pos_params(fold: str | None) -> dict:
    """m_space's position weights: fitted without `fold`, or (fold None) on all 13 clips."""
    if fold is None:
        return json.load(open(prep.WORK + "m_space/params_all13.json", encoding="utf-8"))
    return json.load(open(prep.WORK + f"m_space/params_cache/excl_{fold.split('-')[1]}.json", encoding="utf-8"))


def bonus_of(table, model, only=None, scale: float = 1.0) -> dict:
    """{(a, b): bonus in cosine units} = (pair-model weights of the non-appearance inputs / weight of
    cos_w) . inputs, for the time-ordered pairs with a positive value. only: use just these inputs."""
    cols, w = model["cols"], np.asarray(model["w"])
    wc = w[cols.index("cos_w")]
    rest = [c for c in cols if c != "cos_w" and (only is None or c in only)]
    if not rest or not len(table):
        return {}
    v = scale * (pairmodel.design(table, rest) @ (w[[cols.index(c) for c in rest]] / wc))
    keep = v > 0
    return {(int(a), int(b)): float(x) for a, b, x in zip(table.a[keep], table.b[keep], v[keep])}


def load_params() -> dict:
    return json.loads((HERE / "params_loco.json").read_text(encoding="utf-8"))


def clip_links(seq: str, margin: float, leaves: bool, par: dict, fold: str | None, n_solid: int = N_SOLID,
               scale: float = 1.0, floor=None, t0=None, p_cut=None, log=None):
    """(cuts, core_links, leaf_links) of one clip for one setting with the numbers of `par`
    (fold: whose m_cuts / m_space weights to use; None = the frozen all-13 ones).
    n_solid, scale, floor, t0, p_cut: only for the sensitivity analysis (defaults = the method)."""
    cuts = prep.cuts_for(seq, fold, prep.P_CUT if p_cut is None else p_cut)
    pp = pos_params(fold)
    model = par["pair_model"]
    bonus_core = bonus_of(evidence.pair_table(seq, cuts, pp, n_solid), model, scale=scale)
    bonus_leaf = bonus_of(evidence.pair_table(seq, cuts, pp, 0), model, only=LEAF_COLS, scale=scale) if leaves else None
    core, leaf = linker2.link(seq, cuts, bonus_core, bonus_leaf, margin=margin,
                              floor=par["floor"] if floor is None else floor, t0=par["t0"] if t0 is None else t0,
                              t_abs=par["t_abs"], n_solid=n_solid, leaves=leaves, log=log)
    return cuts, core, leaf


def clip_method(seq: str, margin: float, leaves: bool, par: dict, fold: str | None) -> dict:
    cuts, core, leaf = clip_links(seq, margin, leaves, par, fold)
    return {"links": core + leaf, "cuts": [list(x) for x in cuts]}


def build(mode: str, point: str | None = None, margin: float | None = None, leaves: bool | None = None) -> dict:
    """mode "heldout": each clip with its fold's numbers and its fold's choice for `point`;
    "frozen": all-13 numbers and the all-13 choice; "fixed": the all-13 CHOICE with each clip's fold weights;
    with margin/leaves given: that setting for every clip (mode heldout or frozen)."""
    params = load_params()
    clips = {}
    for seq in data.DEV_CLIPS:
        key, fold = (ALL, None) if mode == "frozen" else (seq, seq)
        par = params[key]
        if margin is not None:
            m, lv = margin, bool(leaves)
        else:
            ch = params[ALL if mode == "fixed" else key]["choice"][point]
            m, lv = ch["margin"], ch["leaves"]
        clips[seq] = clip_method(seq, m, lv, par, fold)
    what = f"point {point}" if margin is None else f"fixed setting margin {margin}, leaves {'on' if leaves else 'off'}"
    how = {"heldout": "LOCO held-out: every clip with the numbers and the choice fitted without it",
           "frozen": "frozen all-13 numbers and choice: IN-SAMPLE on the dev clips",
           "fixed": "the all-13 choice of the setting, applied to every clip with the weights fitted without it"}[mode]
    return {"name": f"fuse_r1 {what} ({how})", "clips": clips}


def cuts_only() -> dict:
    return {"name": "fuse_r1 cuts only (m_cuts P >= 0.7 with the weights fitted without the clip; no links)",
            "clips": {seq: {"links": [], "cuts": [list(x) for x in prep.cuts_for(seq, seq)]} for seq in data.DEV_CLIPS}}


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
    if what in ("all", "fixed"):
        for p in POINTS:
            write(build("fixed", p), f"METHOD_fixed_{p}.json")
    if what in ("all", "cutsonly"):
        write(cuts_only(), "METHOD_cuts_only.json")
    if what == "cand":
        m, lv = float(sys.argv[2]), sys.argv[3] == "on"
        write(build("heldout", margin=m, leaves=lv), f"METHOD_cand_m{m:.2f}_{'on' if lv else 'off'}.json")
