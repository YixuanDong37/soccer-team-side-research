"""DIAGNOSIS (labels): which match each dev clip comes from, and the annotated jersey coverage."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
TRAIN = Path(r"C:\Research\gbf-data\external\sn-gsr-2025-ce84ee7\train")
rows = []
for seq in data.DEV_CLIPS:
    d = json.loads((TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    info = d["info"]
    per = {}
    for a in d["annotations"]:
        at = a.get("attributes") or {}
        if at.get("role") in ("player", "goalkeeper"):
            per.setdefault(a["track_id"], set()).add((at.get("role"), at.get("team"), at.get("jersey")))
    multi = sum(1 for v in per.values() if len(v) > 1)
    nj = sum(1 for v in per.values() if any(x[2] not in (None, "") for x in v))
    # duplicate (team, jersey) among different track ids
    tj = Counter()
    for v in per.values():
        for x in v:
            if x[2] not in (None, ""):
                tj[(x[1], x[2])] += 1
    dup = {k: n for k, n in tj.items() if n > 1}
    print(seq, "game", info.get("game_id"), info.get("game_time_start"), info.get("game_time_stop"), "|", info.get("action_class"),
          "| persons", len(per), "with jersey", nj, "attr changes", multi, "dup team+jersey", dup)
    rows.append((seq, info.get("game_id")))
print(Counter(g for _, g in rows))
