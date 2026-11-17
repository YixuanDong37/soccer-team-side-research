"""critic / o1_jersey.py -- ORACLE (labels): what would a better jersey reader buy?

The annotation gives a jersey number per person (when the annotators could read it at some point of
the 30 s clip, at 1080p). A piece "has a read" when its majority annotated person (weak: >= 1 matched
row) has an annotated number and the piece passes a readability model:
  all      every such piece (upper bound of ANY reader)
  known5   majority person with >= 5 matched rows
  h60x10   >= 10 boxes at least 60 px tall (720p) that do not touch the image edge
  h80x10   the same with 80 px
  h100x10  the same with 100 px
  p50/p25/p10   a random 50 / 25 / 10 % of the "known5" pieces (mean of 20 draws)
  now      LABEL-FREE: the reads gbf keeps today (tracks.jersey, EasyOCR): key = (tracks team, number)
Uses on top of a base method (links + cuts of a fuse_r1 METHOD file, or none):
  veto     no identity may hold two different (team, number) keys: a conflicting group is split
           (pieces without a read go with the nearest piece that has one, by link distance)
  confirm  pieces with the same key are chained in time order
  both     veto, then confirm
Nothing here is a method result: the "read" is the annotated number (except "now").
"""
import sys
from collections import Counter, defaultdict, deque

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/critic")
import common as C
from common import data, score

P = C.Tee(C.OUT / "o1_jersey.txt")
EDGE = data.EDGE_PX


def tall_rows(seq, ids, h_min):
    t = data.tracks(seq)
    x1, y1, x2, y2 = (t[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
    ok = ((y2 - y1) >= h_min) & (x1 > EDGE) & (y1 > EDGE) & (x2 < data.WIDTH - 1 - EDGE) & (y2 < data.HEIGHT - 1 - EDGE)
    return Counter(ids[ok & (ids >= 0)].tolist())


def clean_number(x):
    if x is None or x != x:
        return None
    s = str(x).strip()
    if s in ("", "nan", "None"):
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def keys_for(seq, cuts, model, rng=None):
    """{piece: (team, jersey)} under a readability model."""
    lab = C.labels(seq)
    span, ids = C.piece_span(seq, cuts)
    if model == "now":
        t = data.tracks(seq)
        out = {}
        j = t.jersey.to_numpy(object)
        team = t.team.to_numpy(object)
        for p in span:
            m = ids == p
            js = [v for v in (clean_number(x) for x in j[m]) if v is not None]
            if not js:
                continue
            tm = [x for x in team[m] if x is not None and x == x]
            out[p] = (Counter(tm).most_common(1)[0][0] if tm else None, Counter(js).most_common(1)[0][0])
        return out
    weak = C.piece_majority(seq, cuts, 1)
    strong = C.piece_majority(seq, cuts, 5)
    base = {p: (lab[r]["team"], lab[r]["jersey"]) for p, r in weak.items()
            if lab.get(r, {}).get("role") in ("player", "goalkeeper") and lab[r]["jersey"] is not None}
    if model == "all":
        return base
    if model == "known5":
        return {p: k for p, k in base.items() if p in strong}
    if model.startswith("h"):
        h, n = model[1:].split("x")
        tall = tall_rows(seq, ids, float(h))
        return {p: k for p, k in base.items() if tall.get(p, 0) >= int(n)}
    if model.startswith("p"):
        share = int(model[1:]) / 100
        cand = sorted(p for p in base if p in strong)
        pick = set(np.asarray(cand)[rng.random(len(cand)) < share].tolist()) if cand else set()
        return {p: base[p] for p in pick}
    raise ValueError(model)


def veto(links, keys):
    """Split every linked group that holds two different keys (multi-source BFS from keyed pieces)."""
    adj = defaultdict(set)
    for a, b in links:
        adj[a].add(b)
        adj[b].add(a)
    owner = {}
    q = deque()
    for p in sorted(adj):
        if p in keys:
            owner[p] = keys[p]
            q.append(p)
    while q:
        x = q.popleft()
        for y in sorted(adj[x]):
            if y not in owner:
                owner[y] = owner[x]
                q.append(y)
    return [[a, b] for a, b in links if owner.get(a) is None or owner.get(b) is None or owner[a] == owner[b]]


def confirm(seq, cuts, keys):
    span, _ = C.piece_span(seq, cuts)
    by = defaultdict(list)
    for p, k in keys.items():
        by[k].append(p)
    links = []
    for k, ps in by.items():
        ps = sorted(ps, key=lambda p: span[p][0])
        links += [[ps[i], ps[i + 1]] for i in range(len(ps) - 1)]
    return links


def build(base, model, use, rng=None, skip=()):
    m = {}
    for seq in data.DEV_CLIPS:
        if seq in skip:
            continue
        cuts = [tuple(x) for x in base.get(seq, {}).get("cuts", [])]
        links = [[int(a), int(b)] for a, b in base.get(seq, {}).get("links", [])]
        keys = keys_for(seq, cuts, model, rng)
        if use in ("veto", "both"):
            links = veto(links, keys)
        if use in ("confirm", "both"):
            links = links + confirm(seq, cuts, keys)
        m[seq] = {"links": links, "cuts": [list(x) for x in cuts]}
    return m


def run(base, model, use, seeds=1, skip=()):
    tot = Counter()
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        m = build(base, model, use, rng, skip)
        p = score.evaluate(m, clips=[c for c in data.DEV_CLIPS if c not in skip])["pooled"]
        for k in ("right", "wrong", "lost", "links_right", "links_wrong", "links_unknown"):
            tot[k] += p[k]
        tot["imp"] += p["impure_row_share"]
    return {k: v / seeds for k, v in tot.items()}


def line(r):
    return (f"{r['right']:6.1f} / {r['wrong']:4.1f} / {r['lost']:6.1f} | links {r['links_right']:6.1f} / {r['links_wrong']:4.1f} / "
            f"{r['links_unknown']:6.1f} | impure {100 * r['imp']:.2f}%")


if __name__ == "__main__":
    # ---------------------------------------------------------------- coverage
    P("== coverage of the annotated jersey number ==")
    tp = tj = 0
    ret = Counter()
    for seq in data.DEV_CLIPS:
        lab = C.labels(seq)
        pl = [v for v in lab.values() if v["role"] in ("player", "goalkeeper")]
        tp += len(pl)
        tj += sum(v["jersey"] is not None for v in pl)
        c = score.clip(seq)
        for r in c.returns:
            kind = "scored" if (r["a"] is not None and r["b"] is not None) else "undetected"
            ret[(kind, lab[r["ref_id"]]["jersey"] is not None)] += 1
    P(f"annotated players + goalkeepers with a number: {tj} of {tp} ({100 * tj / tp:.0f}%)")
    P(f"scored returns whose person has a number: {ret[('scored', True)]} of {ret[('scored', True)] + ret[('scored', False)]}; "
      f"undetected returns: {ret[('undetected', True)]} of {ret[('undetected', True)] + ret[('undetected', False)]}")
    for model in ("all", "known5", "h60x10", "h80x10", "h100x10", "now"):
        n = k5 = 0
        for seq in data.DEV_CLIPS:
            n += len(keys_for(seq, (), model))
            k5 += len(C.piece_majority(seq, (), 5))
        P(f"pieces with a read under '{model}': {n} (pieces of known identity: {k5})")

    ok = bad = unk = 0
    for seq in data.DEV_CLIPS:
        lab = C.labels(seq)
        now = keys_for(seq, (), "now")
        weak = C.piece_majority(seq, (), 1)
        for p, (tm, j) in now.items():
            r = weak.get(p)
            if r is None or lab[r]["jersey"] is None:
                unk += 1
            elif lab[r]["jersey"] == j:
                ok += 1
            else:
                bad += 1
    P(f"today's reads (tracks.jersey) against the annotation, per piece: agree {ok}, differ {bad}, cannot judge {unk}")

    # ---------------------------------------------------------------- oracle table
    bases = {"gbf now (no cuts, no links)": {},
             "cuts only": C.method_file("cuts_only"),
             "fuse_r1 strict (held-out)": C.method_file("strict"),
             "fuse_r1 balanced (held-out)": C.method_file("balanced"),
             "fuse_r1 recall (held-out)": C.method_file("recall")}
    P("\n== right / wrong / lost of 137 | links right / wrong / unknown | impure rows ==")
    for bname, base in bases.items():
        P(f"\n-- base: {bname}")
        P(f"   {'base':28s}", line(run(base, "all", "none")))
        for model in ("all", "known5", "h60x10", "h80x10", "h100x10", "p50", "p25", "p10", "now"):
            seeds = 20 if model.startswith("p") else 1
            uses = ("confirm",) if not base else ("veto", "confirm", "both")
            for use in uses:
                P(f"   {model:8s} {use:8s}           ", line(run(base, model, use, seeds)))

    P("\n== the same without SNGS-162 (119 scored returns) ==")
    for bname in ("fuse_r1 strict (held-out)", "fuse_r1 balanced (held-out)", "fuse_r1 recall (held-out)"):
        base = bases[bname]
        P(f"-- base: {bname}")
        P(f"   {'base':28s}", line(run(base, "all", "none", skip=("SNGS-162",))))
        for model in ("all", "h80x10", "p25"):
            seeds = 20 if model.startswith("p") else 1
            P(f"   {model:8s} both               ", line(run(base, model, "both", seeds, skip=("SNGS-162",))))
