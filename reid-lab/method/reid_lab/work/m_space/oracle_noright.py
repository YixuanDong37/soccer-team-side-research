"""ORACLE / DIAGNOSIS ONLY. Later pieces that start at an image border: how many have an
earlier piece of the same person at all, and how many of those are admissible to the linker?"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import features  # noqa: E402
import geom  # noqa: E402
import labels  # noqa: E402
import linker  # noqa: E402

pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 500)
tot = {"edge_starts": 0, "known": 0, "has_earlier": 0, "has_admissible": 0, "first_appearance": 0}
why = {}
for seq in data.DEV_CLIPS:
    e = geom.piece_ends(seq)
    weak = labels.identity_map(seq, 1)
    strong = labels.identity_map(seq, 5)
    p = features.pair_features(seq)
    adm = linker.admissible(p)
    for b in e[(e.start_side != "") & (e["first"] > 0)].itertuples():
        tot["edge_starts"] += 1
        ib = strong.get(int(b.piece))
        if ib is None:
            continue
        tot["known"] += 1
        earlier = e[(e["last"] < b.first) & (e.piece.map(weak) == ib)]
        if not len(earlier):
            tot["first_appearance"] += 1
            continue
        tot["has_earlier"] += 1
        ok = adm[(adm.b == b.piece) & adm.a.isin(earlier.piece)]
        if len(ok):
            tot["has_admissible"] += 1
        else:
            rows = p[(p.b == b.piece) & p.a.isin(earlier.piece)]
            for r in rows.itertuples():
                reason = "unreachable" if not r.reach else ("role" if r.same_role != 1 else ("team" if r.same_team == 0 else "nan"))
                why[reason] = why.get(reason, 0) + 1
                print(seq, "b", b.piece, "person", ib, "earlier", r.a, "gap", r.gap, "d", round(r.d, 1), "reason", reason, "a_rows", r.a_rows, "b_rows", r.b_rows, "teams", r.a_team, r.b_team, "roles", r.a_role, r.b_role)
print(tot)
print("reasons (pairs):", why)
