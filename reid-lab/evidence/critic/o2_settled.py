"""critic / o2_settled.py -- DIAGNOSIS (labels): two measures the lab's scorer does not report.

1. SETTLED identity of a return, on all 167 annotated returns (the scorer's 30 "undetected" included).
   The scorer takes ONE matched row on each side of the gap (within 12 frames). Here the identity
   before the gap is the system identity that holds most of the person's matched rows in the last
   W frames before he left, and the identity after the gap the one that holds most of his matched
   rows in the first W frames after he came back (W = 50 frames = 2 s; also 125).
     same      before == after                                   (re-recognised)
     earlier   after-identity's earlier rows are mostly this person, but it is not the before-identity
     wrong     after-identity's earlier rows are mostly another person (>= 5 rows)
     lost      after-identity is new
     nobox     no matched row of the person in a window
2. WHOLE-CLIP consistency per annotated player / goalkeeper: share of his matched rows that sit in
   his main identity (the identity holding most of his rows), and how many identities hold >= 5 of
   his rows. This is what "he keeps his number through the clip" means; returns are a small part.
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/critic")
import common as C
from common import data, score

P = C.Tee(C.OUT / "o2_settled.txt")


def groups(seq, links, cuts):
    c = score.clip(seq)
    ids = score._ids_after_cuts(c, cuts)
    uf = score._Groups()
    for a, b in links:
        uf.union(int(a), int(b))
    alone = -1 - np.arange(len(ids), dtype=np.int64)
    g = np.where(ids >= 0, np.array([uf.find(int(x)) if x >= 0 else -1 for x in ids], dtype=np.int64), alone)
    return c, ids, g


def settled(method, W, skip=0):
    out = Counter()
    per = []
    for seq in data.DEV_CLIPS:
        m = method.get(seq) or {}
        c, ids, g = groups(seq, m.get("links") or [], m.get("cuts") or [])
        lab = C.labels(seq)
        for r in c.returns:
            me = (c.refs == r["ref_id"]) & (ids >= 0)
            wa = me & (c.frames <= r["last"]) & (c.frames > r["last"] - W)
            wb = me & (c.frames >= r["first"] + skip) & (c.frames < r["first"] + skip + W)
            if not wa.any() or not wb.any():
                o = "nobox"
            else:
                ga = Counter(g[wa].tolist()).most_common(1)[0][0]
                gb = Counter(g[wb].tolist()).most_common(1)[0][0]
                if ga == gb:
                    o = "same"
                else:
                    cnt = Counter(c.refs[(c.frames < r["first"]) & (g == gb)].tolist())
                    if not cnt:
                        o = "lost"
                    else:
                        top, n = cnt.most_common(1)[0]
                        o = "earlier" if top == r["ref_id"] else ("wrong" if n >= 5 else "lost")
            scored = r["a"] is not None and r["b"] is not None
            out[o] += 1
            out[("scored" if scored else "undet", o)] += 1
            gap_s = r["gap"] / 25.0
            out[("long" if gap_s > 10 else "short", o)] += 1
            per.append((seq, r["ref_id"], r["last"], r["first"], o, scored))
    return out, per


def consistency(method):
    main_rows = rows = 0
    n_ids = []
    whole = 0
    persons = 0
    for seq in data.DEV_CLIPS:
        m = method.get(seq) or {}
        c, ids, g = groups(seq, m.get("links") or [], m.get("cuts") or [])
        lab = C.labels(seq)
        for rid, v in lab.items():
            if v["role"] not in ("player", "goalkeeper"):
                continue
            me = (c.refs == rid) & (ids >= 0)
            if me.sum() < 25:
                continue
            cnt = Counter(g[me].tolist())
            top = cnt.most_common(1)[0][1]
            main_rows += top
            rows += int(me.sum())
            n_ids.append(sum(1 for x in cnt.values() if x >= 5))
            persons += 1
            whole += int(top >= 0.95 * me.sum())
    return {"main_share": main_rows / rows, "ids_per_person": float(np.mean(n_ids)), "persons": persons,
            "persons_one_identity_95": whole}


methods = {"gbf now (split)": score.baseline_links("split"),
           "plugin": score.baseline_links("plugin"),
           "mutual": score.baseline_links("mutual"),
           "cuts only": C.method_file("cuts_only"),
           "fuse_r1 strict": C.method_file("strict"),
           "fuse_r1 balanced": C.method_file("balanced"),
           "fuse_r1 recall": C.method_file("recall"),
           "fuse_r1 frozen strict setting": C.method_file("fixed_strict")}

for W, skip in ((50, 0), (50, 12), (125, 0)):
    P(f"\n== settled identity, window {W} frames, first {skip} frames after the return skipped; all 167 annotated returns ==")
    P(f"{'method':32s} same earlier wrong lost nobox | scored 137: same earlier wrong lost nobox | undetected 30: same earlier wrong lost nobox | gap > 10 s: same earlier wrong lost nobox")
    for name, m in methods.items():
        o, _ = settled(m, W, skip)
        f = lambda key: " ".join(f"{o[(key, k)] if key else o[k]:5d}" for k in ("same", "earlier", "wrong", "lost", "nobox"))  # noqa: E731
        P(f"{name:32s} {f(None)} | {f('scored')} | {f('undet')} | {f('long')}")

P("\n== whole-clip consistency per annotated player / goalkeeper (>= 25 matched rows) ==")
P(f"{'method':32s} rows in main identity | identities (>= 5 rows) per person | persons with >= 95% of rows in one identity")
for name, m in methods.items():
    r = consistency(m)
    P(f"{name:32s} {100 * r['main_share']:6.1f}%               | {r['ids_per_person']:.2f}                              | {r['persons_one_identity_95']} of {r['persons']}")

# oracle ceiling for the consistency measure: link every piece to its majority person
orc = {}
for seq in data.DEV_CLIPS:
    cuts = C.method_file("cuts_only")[seq]["cuts"]
    maj = C.piece_majority(seq, cuts, 1)
    by = {}
    for p, r in maj.items():
        by.setdefault(r, []).append(p)
    links = []
    for r, ps in by.items():
        links += [[ps[0], x] for x in ps[1:]]
    orc[seq] = {"links": links, "cuts": cuts}
r = consistency(orc)
P(f"{'ORACLE on the cut pieces':32s} {100 * r['main_share']:6.1f}%               | {r['ids_per_person']:.2f}                              | {r['persons_one_identity_95']} of {r['persons']}")
