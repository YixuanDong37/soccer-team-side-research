"""critic / o7c_appearance_only_loco.py -- the appearance-only variant of o7b under fuse_r1's OWN protocol,
so that the two can be compared like for like (labels: used to choose the lead on the fit clips of a
fold with fuse_r1's rule B, and to score).

Variant "app": m_cuts' cuts (weights fitted without the held-out clip, as in fuse_r1's folds), solid
pieces (>= 5 boxes), m_appear's rule with no position bonus (linker2 with bonus None, floor None,
t0 None, t_abs 0.5). ONE threshold, the lead, chosen per fold on the 12 fit clips by rules.choose
(rule B, caps 4 / 7 / 14, lam 10 / 3 / 1), applied unchanged to the held-out clip; pooled over the 13
held-out clips. Also: the lead chosen without any clip of the held-out clip's match.

This is a diagnosis by the critic (the variant was thought of after seeing the dev clips), not a
proposed result; it answers "what does the position model add under the same rule?".
"""
import sys
from collections import Counter

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

P = C.Tee(C.OUT / "o7c_appearance_only_loco.txt")
LEADS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80)
CANDS = [(False, m) for m in LEADS]
game = {s: C.game(s) for s in data.DEV_CLIPS}
_cache = {}


def cell_for(seq, fold, lead):
    cuts = prep.cuts_for(seq, fold)
    key = (seq, cuts, lead)
    if key not in _cache:
        links, _ = linker2.link(seq, cuts, None, None, margin=lead, floor=None, t0=None, t_abs=0.5, n_solid=method.N_SOLID, leaves=False)
        r = score.score_clip(seq, links, [list(x) for x in cuts])
        _cache[key] = {"off": {"per_return": r["per_return"], "lk": r["links"], "impure": r["impure_rows"], "rows": r["tracked_rows"]}}
    return _cache[key]


def show(cells):
    p = rules.pool(cells)
    return (f"{p['right']:3d} / {p['wrong']:2d} / {p['lost']:3d} | links {p['links_right']:3d} / {p['links_wrong']:2d} / {p['links_unknown']:3d} | "
            f"impure {100 * p['impure_row_share']:.2f}%")


res = {}
for fold in data.DEV_CLIPS:
    res[fold] = {("app", m, s): cell_for(s, fold, m) for s in data.DEV_CLIPS for m in LEADS}
    print("fold", fold, "done", flush=True)

P("right / wrong / lost of 137 | links right / wrong / unknown | impure rows")
for tag, fit_of in (("leave one CLIP out", lambda c: [s for s in data.DEV_CLIPS if s != c]),
                    ("lead chosen without the held-out clip's MATCH (cuts still leave-one-clip-out)", lambda c: [s for s in data.DEV_CLIPS if game[s] != game[c]])):
    P(f"\n== appearance only, rule B, {tag} ==")
    for point in ("strict", "balanced", "recall"):
        chosen = {c: rules.choose(res[c], "app", CANDS, fit_of(c), point, "B")[0] for c in data.DEV_CLIPS}
        cells = {c: rules.cell(res[c], "app", chosen[c], c) for c in data.DEV_CLIPS}
        cnt = Counter(f"{v[1]:.2f}" for v in chosen.values())
        no162 = rules.pool(v for c, v in cells.items() if c != "SNGS-162")
        P(f"{point:9s} {show(cells.values())} | leads chosen {dict(sorted(cnt.items()))} | without SNGS-162 {no162['right']} / {no162['wrong']} / {no162['lost']}"
          f" | SNGS-162 {cells['SNGS-162']['right']} / {cells['SNGS-162']['wrong']} / {cells['SNGS-162']['lost']}")
P("\nfuse_r1 under the same rule (its report): strict 67 / 3 / 67 | 312 / 7 / 25 | 1.61%; balanced 85 / 5 / 47 | 378 / 16 / 30 | 2.12%; recall 106 / 7 / 24 (with leaves) | 414 / 17 / 260 | 2.18%")
