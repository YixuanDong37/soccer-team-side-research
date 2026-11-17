"""Re-run gbf's stitch audit (path G, CPU, pure function, nothing written to the runs) on the 13 dev
clips with every abstention listed, so that the reason for each pair of pieces is known.
The stored refine.json lists only the first 1000 abstentions per clip. Label-free.
-> out/pathg_pairs.json  {seq: {"a,b": reason}}, and a check that the counts equal refine.json's.

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I pathg_audit.py
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
from gbf import stitch  # noqa: E402
from reid_lab import data  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
params = replace(stitch.DEFAULT_PARAMS, max_listed_abstentions=10**8)
res, ok = {}, True
for seq in data.DEV_CLIPS:
    r = data.refine(seq)
    shots = pd.read_parquet(data.run_dir(seq) / "gbf/shots.parquet")
    rep = stitch.audit(data.tracks(seq), data.calib(seq), shots, fps=25.0, team_report=r["teams"], role_report=r["roles"], reads=None, params=params)
    old = r["stitch_audit"]["counts"]
    same = all(rep["counts"][k] == old[k] for k in ("pairs", "decisions", "geometry", "vetoes"))
    ok &= same
    print(seq, "pairs", rep["counts"]["pairs"], "same counts as refine.json:", same, rep["counts"]["geometry"], rep["counts"]["vetoes"])
    d = {}
    for e in rep["abstentions"]:
        why = ("veto:" + e["veto"]) if e.get("veto") else ("G:" + str(e.get("geometry")))
        ev = e.get("geometry_evidence") or {}
        if e.get("geometry") == "contradiction":
            why += " (" + ",".join(sorted({c.get("why", "other_box") for c in ev.get("contradictions", [])})) + ")"
        d[f"{e['earlier']},{e['later']}"] = why
    for e in rep["stitches"] + rep["conflicts_with_plugin"] + rep["ambiguous"] + rep["implied_by_chain"]:
        d[f"{e['earlier']},{e['later']}"] = "G:" + e["decision"]
    res[seq] = d
(OUT / "pathg_pairs.json").write_text(json.dumps(res), encoding="utf-8")
print("all counts reproduced:", ok)
