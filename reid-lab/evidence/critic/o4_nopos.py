"""critic / o4_nopos.py -- STRESS (label-free apply path of fuse_r1, scored with labels): what happens to
fuse_r1 when the POSITION evidence is weak or missing, as it may be on the real clip (other stadium,
recompressed video, calibration less often usable)?

fuse_r1's rule needs appearance cosine + position bonus >= 0.5 + margin. At the strict margin 0.60
that is 1.10: no link can be made on appearance alone. Here the bonus is
  scaled        by 1.0 / 0.75 / 0.5 / 0.25 / 0 for every pair, or
  dropped       for every pair that involves a random share q of the solid pieces (their ground
                position counts as unknown), q = 0.25 / 0.5 / 0.75, mean of 3 draws,
at fixed settings (leaves off, margins 0.60 / 0.40 / 0.20), each clip with the weights fitted without it.
Nothing is refitted or re-chosen. fuse_r1's cache is only read (a write raises).
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/critic")
import common as C
from common import data, score


def _no_write(self, path, *a, **k):
    raise RuntimeError(f"critic must not write {path}")


pd.DataFrame.to_pickle = _no_write
sys.path.insert(0, str(C.FUSE))
import evidence  # noqa: E402
import linker2  # noqa: E402
import method  # noqa: E402
import prep  # noqa: E402

P = C.Tee(C.OUT / "o4_nopos.txt")
params = method.load_params()
_TAB = {}


def table(seq):
    if seq not in _TAB:
        cuts = prep.cuts_for(seq, seq)
        pp = method.pos_params(seq)
        _TAB[seq] = (cuts, evidence.pair_table(seq, cuts, pp, method.N_SOLID))
    return _TAB[seq]


def run(margin, scale=1.0, q=0.0, seed=0):
    m = {}
    rng = np.random.default_rng(seed)
    for seq in data.DEV_CLIPS:
        par = params[seq]
        cuts, tab = table(seq)
        bonus = method.bonus_of(tab, par["pair_model"], scale=scale) if scale > 0 else {}
        if q > 0 and bonus:
            ids = sorted({x for ab in bonus for x in ab})
            drop = set(np.asarray(ids)[rng.random(len(ids)) < q].tolist())
            bonus = {ab: v for ab, v in bonus.items() if ab[0] not in drop and ab[1] not in drop}
        core, _ = linker2.link(seq, cuts, bonus, None, margin=margin, floor=par["floor"], t0=par["t0"], t_abs=par["t_abs"],
                               n_solid=method.N_SOLID, leaves=False)
        m[seq] = {"links": core, "cuts": [list(x) for x in cuts]}
    return score.evaluate(m)["pooled"]


def mean(rs):
    keys = ("right", "wrong", "lost", "links_right", "links_wrong", "links_unknown", "impure_row_share")
    return {k: float(np.mean([r[k] for r in rs])) for k in keys}


def line(p):
    return (f"{p['right']:6.1f} / {p['wrong']:4.1f} / {p['lost']:6.1f} | links {p['links_right']:6.1f} / {p['links_wrong']:4.1f} / "
            f"{p['links_unknown']:5.1f} | impure {100 * p['impure_row_share']:.2f}%")


P("right / wrong / lost of 137 | links right / wrong / unknown | impure rows   (cuts only: 0 / 0 / 137, 1.00%)")
for margin in (0.60, 0.40, 0.20):
    P(f"\n-- margin {margin:.2f}, leaves off")
    for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
        P(f"   bonus x {scale:4.2f}                 ", line(mean([run(margin, scale=scale)])))
    for q in (0.25, 0.5, 0.75):
        P(f"   position unknown for {int(100 * q):2d}% pieces", line(mean([run(margin, q=q, seed=s) for s in range(3)])))
