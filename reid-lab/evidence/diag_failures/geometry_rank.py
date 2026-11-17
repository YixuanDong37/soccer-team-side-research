"""Does pitch position alone find the earlier own piece of a returning player? For the 155 returns
with an own piece on both sides: rank the true earlier piece among every earlier piece of the same
plugin team label (or none), not referee, that ended before the later own piece starts, by
(a) pitch distance between its last and the later piece's first position, (b) the speed that
distance implies, (c) cosine, (d) cosine among the pieces within reach. DIAGNOSIS ONLY (rule 5c).
-> out/geometry_rank.json

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I geometry_rank.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FPS, ctx  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
ret = pd.read_csv(OUT / "returns.csv")
ok = ret[ret.anchor_a.notna() & ret.anchor_b.notna()]
rows = []
for r in ok.itertuples():
    c = ctx(r.clip)
    a, b = c.P[int(r.anchor_a)], c.P[int(r.anchor_b)]
    if a["last_pos"] is None or b["first_pos"] is None or a["last"] >= b["first"]:
        continue
    pool = [q for q in c.P.values() if q["piece"] not in (a["piece"], b["piece"]) and q["last"] < b["first"] and q["last_pos"] is not None
            and q["role"] != "referee" and (q["team"] is None or b["team"] is None or q["team"] == b["team"]) and q["ident_any"] != r.ref_id]

    def dist(q):
        return float(np.hypot(b["first_pos"][1] - q["last_pos"][1], b["first_pos"][2] - q["last_pos"][2]))

    def speed(q):
        return dist(q) / max((b["first_pos"][0] - q["last_pos"][0]) / FPS, 0.04)

    def cs(q):
        return -float(q["desc_any"] @ b["desc_any"]) if q["desc_any"] is not None and b["desc_any"] is not None else 1.0

    reach = [q for q in pool if dist(q) <= 8.0 * (b["first_pos"][0] - q["last_pos"][0]) / FPS + 4.0]
    edge = [q for q in pool if q["ends_at_edge"]]
    rows.append({"clip": r.clip, "ret": r.ret, "pool": len(pool), "in_reach": len(reach), "true_dist": dist(a), "true_speed": speed(a),
                 "rank_dist": 1 + sum(dist(q) < dist(a) for q in pool), "rank_speed": 1 + sum(speed(q) < speed(a) for q in pool),
                 "rank_cos": 1 + sum(cs(q) < cs(a) for q in pool), "rank_cos_in_reach": 1 + sum(cs(q) < cs(a) for q in reach),
                 "rank_dist_edge_enders": 1 + sum(dist(q) < dist(a) for q in edge), "edge_enders": len(edge), "a_ends_at_edge": a["ends_at_edge"]})  # fmt: skip
d = pd.DataFrame(rows)
res = {"pairs": len(d), "median other-person pieces to choose from": float(d.pool.median()), "median of them within reach (8 m/s + 4 m)": float(d.in_reach.median()),
       "true piece is rank 1 by distance": int((d.rank_dist == 1).sum()), "rank <= 3 by distance": int((d.rank_dist <= 3).sum()),
       "rank 1 by implied speed": int((d.rank_speed == 1).sum()), "rank 1 by cosine": int((d.rank_cos == 1).sum()),
       "rank 1 by cosine among those within reach": int((d.rank_cos_in_reach == 1).sum()),
       "rank 1 by distance among pieces that ended at an image edge (true piece ends at an edge in %d)" % int(d.a_ends_at_edge.sum()): int(((d.rank_dist_edge_enders == 1) & d.a_ends_at_edge).sum()),
       "true distance m quantiles 25/50/75/95": d.true_dist.quantile([.25, .5, .75, .95]).round(1).tolist(),
       "true speed m/s quantiles 25/50/75/95": d.true_speed.quantile([.25, .5, .75, .95]).round(2).tolist()}  # fmt: skip
(OUT / "geometry_rank.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
