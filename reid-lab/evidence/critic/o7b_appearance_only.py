"""critic / o7b_appearance_only.py -- ABLATION (label-free links, scored with labels; fixed settings, nothing
fitted or chosen): how much of fuse_r1's result needs the POSITION model at all?

Same cuts (m_cuts P >= 0.7, weights fitted without the clip) and the same solid tier (>= 5 boxes) as
fuse_r1, but the links come from appearance only: m_appear's rule (whitened team-centred prototypes,
mutual best, t_abs 0.5, lead over every candidate seen together with the partner), i.e. fuse_r1's
linker2 with no bonus, no appearance floor and no "nobody" rival. One threshold: the lead.
Compared with fuse_r1's fixed-setting rows (leaves off, held-out weights) from its loco_raw.pkl.
The appearance path has no fitted weight except m_cuts' five (held-out); whitening is per clip, label-free.
"""
import pickle
import sys

import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/critic")
import common as C
from common import data, score


def _no_write(self, path, *a, **k):
    raise RuntimeError(f"critic must not write {path}")


pd.DataFrame.to_pickle = _no_write
sys.path.insert(0, str(C.FUSE))
import linker2  # noqa: E402
import method  # noqa: E402
import prep  # noqa: E402
import rules  # noqa: E402

P = C.Tee(C.OUT / "o7b_appearance_only.txt")


def line(p):
    return (f"{p['right']:3d} / {p['wrong']:2d} / {p['lost']:3d} | links {p['links_right']:3d} / {p['links_wrong']:2d} / "
            f"{p['links_unknown']:3d} | impure {100 * p['impure_row_share']:.2f}%")


P("right / wrong / lost of 137 | links right / wrong / unknown | impure rows | without SNGS-162 | SNGS-162")
P("-- appearance only on the cut, solid pieces (lead = the one threshold)")
for lead in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70):
    m = {}
    for seq in data.DEV_CLIPS:
        cuts = prep.cuts_for(seq, seq)
        app, _ = linker2.link(seq, cuts, None, None, margin=lead, floor=None, t0=None, t_abs=0.5, n_solid=method.N_SOLID, leaves=False)
        m[seq] = {"links": app, "cuts": [list(x) for x in cuts]}
    r = score.evaluate(m)
    p = r["pooled"]
    c162 = r["per_clip"]["SNGS-162"]["returns"]
    rest = {k: sum(v["returns"][k] for s, v in r["per_clip"].items() if s != "SNGS-162") for k in ("right", "wrong", "lost")}
    P(f"   lead {lead:.2f}   {line(p)} | {rest['right']:3d} / {rest['wrong']} / {rest['lost']:3d} | {c162['right']} / {c162['wrong']} / {c162['lost']}")

P("-- fuse_r1 (appearance + position bonus), fixed margin, leaves off, held-out weights")
raw = pickle.load(open(C.FUSE / "cache" / "loco_raw.pkl", "rb"))["folds"]
for mg in (0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80):
    cells = {c: rules.cell(raw[c]["res"], "r1", (False, mg), c) for c in data.DEV_CLIPS}
    p = rules.pool(cells.values())
    rest = rules.pool(v for c, v in cells.items() if c != "SNGS-162")
    c162 = cells["SNGS-162"]
    P(f"   margin {mg:.2f} {line(p)} | {rest['right']:3d} / {rest['wrong']} / {rest['lost']:3d} | {c162['right']} / {c162['wrong']} / {c162['lost']}")
