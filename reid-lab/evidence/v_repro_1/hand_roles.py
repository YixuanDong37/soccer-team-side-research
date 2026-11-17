"""v_repro_1 / hand_roles.py -- does any annotated return come from a role change instead of an absence?
score.py keeps the annotation rows whose role is player or goalkeeper and looks for gaps in what is
left; a person annotated as "other" or "referee" for a while would produce a gap although he is in
the picture. LABELS (check of the scorer only). Imports neither reid_lab nor gbf.

    python -I -B hand_roles.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_1")
import hand_check as h  # noqa: E402

total = mixed = 0
gaps_with_other_role = 0
for seq in h.DEV:
    doc = json.loads((h.TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    fr = {str(im["image_id"]): int(Path(im["file_name"]).stem) - 1 for im in doc["images"]}
    roles = defaultdict(dict)                     # person -> frame -> role (every role, also "other" / "ball")
    for a in doc["annotations"]:
        if a.get("supercategory") != "object" or "track_id" not in a:
            continue
        roles[int(a["track_id"])][fr[str(a["image_id"])]] = (a.get("attributes") or {}).get("role")
    for person, by_frame in roles.items():
        rs = set(by_frame.values())
        if not rs & {"player", "goalkeeper"}:
            continue
        total += 1
        if len(rs) > 1:
            mixed += 1
            print(f"  {seq} person {person}: roles {sorted(str(r) for r in rs)}")
        frames = sorted(f for f, r in by_frame.items() if r in ("player", "goalkeeper"))
        for u, v in zip(frames[:-1], frames[1:]):
            if v - u - 1 >= 10 and any(u < f < v for f in by_frame):
                gaps_with_other_role += 1
                print(f"  {seq} person {person}: gap {u}>{v} holds annotation rows with another role")
print(f"annotated players / goalkeepers: {total}; with more than one role over the clip: {mixed}; "
      f"gaps of >= 10 frames that hold rows of the same person with another role: {gaps_with_other_role}")
