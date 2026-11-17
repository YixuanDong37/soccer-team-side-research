"""ORACLE on top of the label-free hole cuts (probe_hole_cuts.py, H = 10): link the cut pieces by
their annotated majority person. Ceiling of "cut at every hole, then link perfectly".
DIAGNOSIS ONLY (rule 5c). -> out/probe_hole_cuts_oracle.json

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I probe_hole_cuts_oracle.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ctx, data, score  # noqa: E402
from diagnose_returns import oracle_cut_links  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
m = json.loads((OUT / "METHOD_probe_hole_cuts_H10.json").read_text(encoding="utf-8"))["clips"]
res = {}
for name, k in (("links between pieces of known identity (>= 5 matched rows)", 5), ("links of every piece with a matched row", 1)):
    mm = {seq: {"cuts": m[seq]["cuts"], "links": oracle_cut_links(ctx(seq), m[seq]["cuts"], k)} for seq in data.DEV_CLIPS}
    res[name] = score.evaluate(mm)["pooled"]
    print(name, json.dumps(res[name]))
(OUT / "probe_hole_cuts_oracle.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
