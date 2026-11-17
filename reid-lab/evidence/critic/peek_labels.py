"""DIAGNOSIS (labels): structure of the annotation's attributes (jersey, team, role)."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
TRAIN = Path(r"C:\Research\gbf-data\external\sn-gsr-2025-ce84ee7\train")
seq = sys.argv[1] if len(sys.argv) > 1 else "SNGS-162"
d = json.loads((TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
print(d.keys())
print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in d.items()})
print(d["images"][0])
anns = d["annotations"]
print(anns[0]); 
persons = [a for a in anns if a.get("attributes")]
print(len(anns), len(persons))
print(persons[0])
byid = defaultdict(lambda: Counter())
for a in persons:
    at = a["attributes"]
    byid[a["track_id"]][(at.get("role"), at.get("team"), at.get("jersey"))] += 1
for k in sorted(byid):
    print(k, dict(byid[k]))
print(d.get("info"))
