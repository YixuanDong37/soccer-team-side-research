"""Counts of gbf pieces on the 13 dev clips, for the report's context paragraph.
DIAGNOSIS ONLY (rule 5c). -> out/piece_stats.json

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I piece_stats.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ctx, data, eligible_why  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
n = dict.fromkeys(["pieces", "known identity (>= 5 matched rows)", "1-4 matched rows", "no matched row", "<= 4 rows", "strict descriptor (>= 5 clean rows)",
                   "no embedding at all", "player with verified team", "eligible for appearance_match", "known and eligible", "known, not eligible: team unverified",
                   "known, not eligible: clean rows < 5", "known, not eligible: role", "rows", "rows in pieces <= 4 rows"], 0)  # fmt: skip
for seq in data.DEV_CLIPS:
    for p in ctx(seq).P.values():
        why = eligible_why(p)
        n["pieces"] += 1
        n["rows"] += p["rows"]
        n["rows in pieces <= 4 rows"] += p["rows"] if p["rows"] <= 4 else 0
        n["known identity (>= 5 matched rows)"] += p["ident"] is not None
        n["1-4 matched rows"] += p["ident"] is None and p["matched"] > 0
        n["no matched row"] += p["matched"] == 0
        n["<= 4 rows"] += p["rows"] <= 4
        n["strict descriptor (>= 5 clean rows)"] += p["desc"] is not None
        n["no embedding at all"] += p["desc_any"] is None
        n["player with verified team"] += p["role"] == "player" and p["vteam"] is not None
        n["eligible for appearance_match"] += not why
        if p["ident"] is not None:
            n["known and eligible"] += not why
            n["known, not eligible: team unverified"] += any("team" in w for w in why)
            n["known, not eligible: clean rows < 5"] += any("clean" in w for w in why)
            n["known, not eligible: role"] += any("role" in w for w in why)
n = {k: int(v) for k, v in n.items()}
(OUT / "piece_stats.json").write_text(json.dumps(n, indent=1), encoding="utf-8")
print(json.dumps(n, indent=1))
