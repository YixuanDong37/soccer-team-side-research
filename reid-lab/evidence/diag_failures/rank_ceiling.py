"""How well the piece descriptors rank the true earlier piece of a later piece, with and without
simple label-free restrictions. Labels are used to say which answer is true, so every number here
is an in-sample DIAGNOSIS (rule 5c), not a method result. -> out/rank_ceiling.json

Later pieces B: every piece of a known annotated player/goalkeeper (>= 5 matched rows) that
starts after frame 5 and has a descriptor. Earlier candidates: pieces that ended before B starts,
with a descriptor, not referee, plugin team label equal or missing.
  returning B: the person has an own earlier piece among the candidates;
  new B: he has none (first appearance, or only fragments before) -> any link is wrong.

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I rank_ceiling.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FPS, ctx, data  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def speed_ok(q, b, vmax=8.0, err=4.0):
    if q["last_pos"] is None or b["first_pos"] is None:
        return True
    d = float(np.hypot(b["first_pos"][1] - q["last_pos"][1], b["first_pos"][2] - q["last_pos"][2]))
    return d <= vmax * (b["first_pos"][0] - q["last_pos"][0]) / FPS + err


VARIANTS = {
    "all earlier pieces": lambda q, b: True,
    "strict descriptors only (>= 5 clean rows both)": lambda q, b: q["desc_kind"] == "strict" and b["desc_kind"] == "strict",
    "earlier piece ended at an image edge": lambda q, b: q["ends_at_edge"],
    "ended at an edge, B starts at the same side": lambda q, b: q["ends_at_edge"] and b["starts_at_edge"] and bool(set(q["end_side"].split("+")) & set(b["start_side"].split("+"))),
    "within reach (8 m/s + 4 m)": lambda q, b: speed_ok(q, b),
    "gap <= 10 s": lambda q, b: (b["first"] - q["last"]) / FPS <= 10.0,
}
res = {}
for name, keep in VARIANTS.items():
    tot = Counter()
    margins = {"right": [], "wrong": [], "new": []}
    coss = {"right": [], "wrong": [], "new": []}
    for seq in data.DEV_CLIPS:
        c = ctx(seq)
        P = c.P
        for b in P.values():
            rid = b["ident"]
            if rid is None or b["first"] <= 5 or b["desc_any"] is None or c.att[rid]["role"] not in ("player", "goalkeeper"):
                continue
            pool = [q for q in P.values() if q["last"] < b["first"] and q["desc_any"] is not None and q["role"] != "referee"
                    and (q["team"] is None or b["team"] is None or q["team"] == b["team"])]
            has_true_any = any(q["ident"] == rid for q in pool)
            pool = [q for q in pool if keep(q, b)]
            has_true = any(q["ident"] == rid for q in pool)
            kind = "returning" if has_true_any else "new"
            tot[f"{kind}: pieces B"] += 1
            if not pool:
                tot[f"{kind}: no candidate"] += 1
                continue
            s = sorted(((float(b["desc_any"] @ q["desc_any"]), q) for q in pool), key=lambda x: -x[0])
            top = s[0][1]
            # the runner-up that is NOT another piece of the top piece's person
            rest = [v for v, q in s[1:] if q["ident_any"] is None or q["ident_any"] != top["ident_any"]]
            margin = s[0][0] - (rest[0] if rest else 0.0)
            if kind == "new":
                tot["new: would link (top-1 exists)"] += 1
                margins["new"].append(margin); coss["new"].append(s[0][0])
                continue
            if not has_true:
                tot["returning: true piece removed by the restriction"] += 1
            if top["ident"] == rid:
                tot["returning: top-1 is the person"] += 1
                margins["right"].append(margin); coss["right"].append(s[0][0])
            elif top["ident"] is None:
                tot["returning: top-1 is a fragment (unknown)"] += 1
            else:
                tot["returning: top-1 is another person"] += 1
                margins["wrong"].append(margin); coss["wrong"].append(s[0][0])
    out = dict(tot)
    for t, m in ((0.90, 0.0), (0.93, 0.0), (0.95, 0.0), (0.0, 0.03), (0.0, 0.05), (0.90, 0.03), (0.93, 0.03), (0.93, 0.05), (0.95, 0.05)):
        out[f"cos>={t}, margin>={m}"] = {k: int(sum(1 for cv, mv in zip(coss[k], margins[k]) if cv >= t and mv >= m)) for k in ("right", "wrong", "new")}
    res[name] = out
    print(name); print(json.dumps(out, indent=1))
(OUT / "rank_ceiling.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
