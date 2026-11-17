"""fuse_r1 / check_linker2.py -- LABEL-FREE checks of the two-stage linker.

1. With n_solid = 0 (every piece solid, so no leaves) linker2.link returns exactly fuse's
   linker.agglomerate links, with and without a bonus.
2. With n_solid = 5: no core link touches a short piece; every leaf link joins one short piece and
   one solid piece; no short piece has two links; the groups (union of all links) never hold two
   pieces that share a frame, never mix appearance teams and never hold a referee.
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse_r1")
from collections import Counter

from reid_lab import data
import evidence, linker, linker2, prep
from method import LEAF_COLS, bonus_of, pos_params

model = {"cols": ["cos_w", "sink_lo", "plug"], "w": [7.2, 0.68, 0.95], "b": -7.15}
bad = 0
for seq in data.DEV_CLIPS:
    cuts = prep.cuts_for(seq, None)
    pp = pos_params(None)
    full = evidence.pair_table(seq, cuts, pp)
    bonus = bonus_of(full, model)
    for b in (None, bonus):
        for m in (0.25, 0.5, 0.9):
            x = linker.agglomerate(seq, cuts, b, t_abs=0.5, margin=m, floor=0.3, t0=0.5)
            y, z = linker2.link(seq, cuts, b, None, margin=m, floor=0.3, t0=0.5, t_abs=0.5, n_solid=0, leaves=True)
            same = sorted(map(tuple, x)) == sorted(map(tuple, y)) and not z
            bad += not same
            print(seq, "check 1", "bonus" if b else "no bonus", m, len(x), "links", "equal" if same else "DIFFERENT")
    # check 2
    af, core = linker._appear()
    c = core.ctx(seq, cuts)
    pr = af.protos(seq, cuts)
    rows = prep.piece_rows(seq, cuts)
    solid = prep.solid_ids(seq, cuts)
    bc = bonus_of(evidence.pair_table(seq, cuts, pp, prep.N_SOLID), model)
    bl = bonus_of(full, model, only=LEAF_COLS)
    for m in (0.15, 0.5):
        cl, ll = linker2.link(seq, cuts, bc, bl, margin=m, floor=0.3, t0=0.5, t_abs=0.5)
        problems = []
        if any(a not in solid or b not in solid for a, b in cl):
            problems.append("core link with a short piece")
        if any((a in solid) == (b in solid) for a, b in ll):
            problems.append("leaf link not short-solid")
        cnt = Counter(p for l in ll for p in l if p not in solid)
        if any(v > 1 for v in cnt.values()):
            problems.append("short piece with two links")
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in cl + ll:
            parent[find(a)] = find(b)
        groups = {}
        for x in list(parent):
            groups.setdefault(find(x), []).append(x)
        for g in groups.values():
            fr = Counter(f for p in g for f in c.frames_of[p])
            if max(fr.values()) > 1:
                problems.append("two pieces of a group in one frame")
            if len({pr.ateam[p] for p in g}) > 1:
                problems.append("mixed teams")
            if any(c.P.role[c.index[p]] == "referee" for p in g):
                problems.append("referee")
        bad += bool(problems)
        print(seq, "check 2 margin", m, len(cl), "core links,", len(ll), "leaf links,", len(solid), "solid of", len(rows), "pieces:",
              "ok" if not problems else sorted(set(problems)))
print("ALL CHECKS PASS" if not bad else f"{bad} PROBLEMS")
