"""Probe (label-free rule, no fitted parameter; labels only score it): cut every gbf piece at every
hole of >= H frames, wherever it is in the image (gbf cuts only at holes next to the image edge),
make no link. Shows how much of split's 4 wrong returns and of the impure rows comes from
re-used plugin ids after a hole. H = 10 is gbf's own re-entry gap; the others are shown for
context only. -> out/probe_hole_cuts.json, out/METHOD_probe_hole_cuts_H10.json

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I probe_hole_cuts.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
res = {}
for H in (10, 15, 25, 50):
    m = {}
    n = 0
    for seq in data.DEV_CLIPS:
        t = data.tracks(seq)
        t = t[t.track_id >= 0]
        cuts = []
        for pid, g in t.groupby("track_id"):
            f = np.sort(g.frame.unique())
            cuts += [[int(pid), int(f[i + 1])] for i in np.where(np.diff(f) - 1 >= H)[0]]
        m[seq] = {"links": [], "cuts": cuts}
        n += len(cuts)
    res[f"H={H}"] = {"cuts": n, **score.evaluate(m)["pooled"]}
    if H == 10:
        (OUT / "METHOD_probe_hole_cuts_H10.json").write_text(json.dumps({"name": "probe: cut at every hole >= 10 frames, no links", "clips": m}), encoding="utf-8")
    print(H, json.dumps(res[f"H={H}"]))
(OUT / "probe_hole_cuts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
