"""Probe for question 3 (IN-SAMPLE, thresholds read off the 13 clips: a diagnosis, not a method
result): keep a plugin re-entry join only when the two pieces look alike. Uses the label-free
columns of out/plugin_joins.csv (cos_any, pieces_scoring_higher, a_rows, b_rows); labels only score.
With and without the label-free hole cuts of probe_hole_cuts.py (H = 10).
-> out/probe_plugin_filter.json

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I probe_plugin_filter.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
pj = pd.read_csv(OUT / "plugin_joins.csv")
res = {}
for name, keep in {
    "all plugin joins (= plugin baseline)": pj.index == pj.index,
    "cos_any >= 0.88": pj.cos_any >= 0.88,
    "cos_any >= 0.90": pj.cos_any >= 0.90,
    "cos_any >= 0.92": pj.cos_any >= 0.92,
    "cos_any >= 0.90, both pieces >= 5 rows": (pj.cos_any >= 0.90) & (pj.a_rows >= 5) & (pj.b_rows >= 5),
    "cos_any >= 0.90, both >= 5 rows, <= 1 other piece scores higher": (pj.cos_any >= 0.90) & (pj.a_rows >= 5) & (pj.b_rows >= 5) & (pj.pieces_scoring_higher <= 1),
    "cos_any >= 0.90, both >= 10 rows, no other piece scores higher": (pj.cos_any >= 0.90) & (pj.a_rows >= 10) & (pj.b_rows >= 10) & (pj.pieces_scoring_higher == 0),
}.items():
    k = pj[keep]
    m = {seq: {"links": [[int(a), int(b)] for a, b in zip(k[k["clip"] == seq].earlier, k[k["clip"] == seq].later)]} for seq in data.DEV_CLIPS}
    p = score.evaluate(m)["pooled"]
    res[name] = {x: p[x] for x in ("right", "wrong", "lost", "links", "links_right", "links_wrong", "links_unknown", "impure_row_share")}
    print(name, json.dumps(res[name]))
(OUT / "probe_plugin_filter.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
