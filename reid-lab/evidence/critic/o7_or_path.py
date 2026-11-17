"""critic / o7_or_path.py -- PROBE (label-free links, scored with labels; fixed settings, nothing fitted or
chosen here, but the idea comes from looking at these clips: diagnosis, not a result).

fuse_r1's rule is ADDITIVE: appearance cosine + position bonus >= 0.5 + margin. At the strict margin
(0.60) no pair can pass on appearance alone, so returns after a long absence, where position says
little, are mostly lost although appearance is still good there (o5b: 38 of 41 top-1 for gaps > 10 s).
Probe: an OR path. Keep the strict links and add links that m_appear's own rule (mutual best, lead over
every candidate seen together with the partner; no position) makes with a HIGH lead on the same cut,
solid pieces. A link of the appearance path is dropped when it would put two boxes of one frame into
one identity together with the strict links.

Settings: strict = fuse_r1 margin 0.60, leaves off, weights fitted without the clip. Appearance path:
t_abs 0.5, lead m in (0.30, 0.40, 0.50, 0.60, 0.70).
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
import linker2  # noqa: E402
import method  # noqa: E402
import prep  # noqa: E402

P = C.Tee(C.OUT / "o7_or_path.txt")
params = method.load_params()
strict = C.method_file("fixed_strict")


def frames_of(seq, cuts):
    t = data.tracks(seq)
    ids = prep.ids_after_cuts(t.track_id.to_numpy(), t.frame.to_numpy(), cuts)
    out = {}
    fr = t.frame.to_numpy()
    for i in np.unique(ids[ids >= 0]):
        out[int(i)] = set(fr[ids == i].tolist())
    return out


def union_safe(seq, cuts, base, extra):
    """base links + those of extra that keep every identity free of two boxes in one frame."""
    fo = frames_of(seq, cuts)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    fr = {}
    for a, b in base:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for p in fo:
        r = find(p)
        fr.setdefault(r, set()).update(fo[p])
    kept, dropped = [], 0
    for a, b in extra:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if fr[ra] & fr[rb]:
            dropped += 1
            continue
        parent[ra] = rb
        fr[rb] = fr[rb] | fr.pop(ra)
        kept.append([a, b])
    return base + kept, len(kept), dropped


def settled_long(m):
    """Settled verdicts (o2's measure, W = 50) for the returns after more than 10 s."""
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        import o2_settled as S
    o, _ = S.settled(m, 50, 0)
    return o


def line(p):
    return (f"{p['right']:3d} / {p['wrong']:2d} / {p['lost']:3d} | links {p['links_right']:3d} / {p['links_wrong']:2d} / "
            f"{p['links_unknown']:3d} | impure {100 * p['impure_row_share']:.2f}%")


base_p = score.evaluate(strict)["pooled"]
o = settled_long(strict)
P("right / wrong / lost of 137 | links right / wrong / unknown | impure rows | settled, all 167: same / wrong | settled, gap > 10 s (44): same / wrong")
P(f"strict alone (margin 0.60)            {line(base_p)} | {o['same']:3d} / {o['wrong']:2d} | {o[('long', 'same')]:2d} / {o[('long', 'wrong')]}")
for m_app in (0.30, 0.40, 0.50, 0.60, 0.70):
    alone, both = {}, {}
    added = dropped = 0
    for seq in data.DEV_CLIPS:
        cuts = prep.cuts_for(seq, seq)
        cl = [list(x) for x in cuts]
        app, _ = linker2.link(seq, cuts, None, None, margin=m_app, floor=None, t0=None, t_abs=0.5, n_solid=method.N_SOLID, leaves=False)
        alone[seq] = {"links": app, "cuts": cl}
        u, k, d = union_safe(seq, cuts, [list(map(int, x)) for x in strict[seq]["links"]], [list(map(int, x)) for x in app])
        added += k
        dropped += d
        both[seq] = {"links": u, "cuts": cl}
    pa, pb = score.evaluate(alone)["pooled"], score.evaluate(both)["pooled"]
    oa, ob = settled_long(alone), settled_long(both)
    P(f"appearance path alone, lead {m_app:.2f}     {line(pa)} | {oa['same']:3d} / {oa['wrong']:2d} | {oa[('long', 'same')]:2d} / {oa[('long', 'wrong')]}")
    P(f"strict OR appearance path, lead {m_app:.2f} {line(pb)} | {ob['same']:3d} / {ob['wrong']:2d} | {ob[('long', 'same')]:2d} / {ob[('long', 'wrong')]}   (+{added} links, {dropped} dropped for a shared frame)")
    per = score.evaluate(both)["per_clip"]
    P("      wrong returns per clip:", {s: v["returns"]["wrong"] for s, v in per.items() if v["returns"]["wrong"]},
      "| wrong links per clip:", {s: v["links"]["wrong"] for s, v in per.items() if v["links"]["wrong"]})
