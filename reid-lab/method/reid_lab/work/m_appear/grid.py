"""Run a family of label-free methods over a parameter grid on the 13 dev clips and cache, per clip and
setting, the links and their score (score.score_clip). loco.py picks settings from this cache.

    python -I grid.py agglo        -> cache/agglo.json
    python -I grid.py mutual       -> cache/mutual.json   (pairwise mutual-best rule, for comparison)

Labels are touched only through score.score_clip (rule 5a); the links themselves are label-free.
"""
import itertools
import json
import os
import sys
import time

sys.path.insert(0, "C:/Research/plugins/tools")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reid_lab import data, score  # noqa: E402

import decide  # noqa: E402
from agglo import agglomerate  # noqa: E402
from features import pair_features  # noqa: E402

T_ABS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
MARGIN = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

FAMILIES = {
    # name -> (parameter grid, function(seq, **params) -> links)
    "agglo": (
        {"kind": ["tc", "w"], "t_abs": T_ABS, "margin": MARGIN},
        lambda seq, kind, t_abs, margin: agglomerate(seq, kind=kind, t_abs=t_abs, margin=margin, covis=1),
    ),
    "agglo_covis5": (   # rivals must share >= 5 frames with the partner (worse; kept for the nested check)
        {"kind": ["tc", "w"], "t_abs": T_ABS, "margin": MARGIN},
        lambda seq, kind, t_abs, margin: agglomerate(seq, kind=kind, t_abs=t_abs, margin=margin, covis=5),
    ),
    "mutual": (
        {"kind": ["tc", "w"], "t_abs": T_ABS, "margin": MARGIN},
        lambda seq, kind, t_abs, margin: decide.mutual(pair_features(seq), kind, t_abs, margin, need_runner_up=False, margin_kind="rival"),
    ),
}


def key(params):
    return json.dumps(params, sort_keys=True)


def one(job):
    family, params = job
    fn = FAMILIES[family][1]
    per = {}
    for seq in data.DEV_CLIPS:
        links = [[int(a), int(b)] for a, b in fn(seq, **params)]
        s = score.score_clip(seq, links)
        per[seq] = {"links": links, "returns": s["returns"], "link_score": s["links"], "impure_rows": s["impure_rows"],
                    "tracked_rows": s["tracked_rows"]}
    return key(params), per


def run(family, workers=6):
    from multiprocessing import Pool
    grid = FAMILIES[family][0]
    names = list(grid)
    out = {}
    t0 = time.time()
    combos = [dict(zip(names, c)) for c in itertools.product(*[grid[k] for k in names])]
    with Pool(workers) as pool:
        for n, (k, per) in enumerate(pool.imap_unordered(one, [(family, c) for c in combos], chunksize=2)):
            out[k] = per
            if n % 20 == 0:
                print(f"{family} {n + 1}/{len(combos)} {round(time.time() - t0)} s", flush=True)
    out = {key(c): out[key(c)] for c in combos}
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    with open(os.path.join(HERE, "cache", f"{family}.json"), "w", encoding="utf-8") as f:
        json.dump({"family": family, "grid": grid, "results": out}, f)
    print("wrote", os.path.join(HERE, "cache", f"{family}.json"))


if __name__ == "__main__":
    for fam in sys.argv[1:] or ["agglo"]:
        run(fam)
