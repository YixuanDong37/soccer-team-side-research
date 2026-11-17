"""critic / o5_roster.py -- ORACLE (annotation only): how far would IDENTITY BOOKKEEPING narrow a return?

Idea not tried by the methods: a team has at most 10 outfield players. If every other identity were
tracked correctly, the candidates for a player who walks into the picture are only the team-mates who
are NOT in the picture at that moment (and have been seen before). The methods compare pieces; this
counts people.

For each of the 167 annotated returns (frame `first` = the re-entry):
  seen      same-team outfield players annotated at any frame before `first` (the returner included)
  absent    those of `seen` with no annotated box at frame `first` - 1 (the frame before he is back)
            = the candidates under perfect bookkeeping (the returner is one of them)
  same side those of `absent` whose last box before `first` was in the same image half as the re-entry box
  together  other same-team players who re-enter within +-25 frames (they must be told apart jointly)
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/critic")
import common as C
from common import data, score

P = C.Tee(C.OUT / "o5_roster.txt")
rows = []
for seq in data.DEV_CLIPS:
    d = json.loads((C.TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    img_frame = {im["image_id"]: int(im["file_name"].split(".")[0]) - 1 for im in d["images"]}
    frames = defaultdict(dict)       # ref -> {frame: x centre}
    team = {}
    for a in d["annotations"]:
        at = a.get("attributes") or {}
        if at.get("role") != "player":
            continue
        rid = int(a["track_id"])
        team[rid] = at.get("team")
        frames[rid][img_frame[a["image_id"]]] = float(a["bbox_image"]["x_center"])
    c = score.clip(seq)
    rets = [r for r in c.returns if r["ref_id"] in team]
    for r in rets:
        p, f = r["ref_id"], r["first"]
        mates = [q for q in team if team[q] == team[p]]
        seen = [q for q in mates if min(frames[q]) < f]
        absent = [q for q in seen if (f - 1) not in frames[q]]
        side = frames[p][f] < 960
        same_side = []
        for q in absent:
            before = [x for x in frames[q] if x < f]
            lastx = frames[q][max(before)]
            if (lastx < 960) == side:
                same_side.append(q)
        together = [q["ref_id"] for q in rets if q["ref_id"] != p and team[q["ref_id"]] == team[p] and abs(q["first"] - f) <= 25]
        rows.append({"seq": seq, "p": p, "gap": r["gap"], "seen": len(seen), "absent": len(absent), "same_side": len(same_side),
                     "together": len(together), "scored": r["a"] is not None and r["b"] is not None})

n = len(rows)
P(f"annotated returns of outfield players: {n}")
for key in ("seen", "absent", "same_side", "together"):
    v = np.array([r[key] for r in rows])
    P(f"{key:9s}: median {np.median(v):.0f}, quartiles {np.percentile(v, 25):.0f}-{np.percentile(v, 75):.0f}, max {v.max()};  distribution {dict(sorted(Counter(v.tolist()).items()))}")
v = np.array([r["absent"] for r in rows])
s = np.array([r["same_side"] for r in rows])
P(f"returns with the returner as the ONLY absent team-mate: {int((v == 1).sum())} ({100 * np.mean(v == 1):.0f}%); at most 2 absent: {int((v <= 2).sum())} ({100 * np.mean(v <= 2):.0f}%); at most 3: {int((v <= 3).sum())} ({100 * np.mean(v <= 3):.0f}%)")
P(f"... counting only absent team-mates last seen in the same image half: only 1: {int((s == 1).sum())} ({100 * np.mean(s == 1):.0f}%); at most 2: {int((s <= 2).sum())} ({100 * np.mean(s <= 2):.0f}%); at most 3: {int((s <= 3).sum())} ({100 * np.mean(s <= 3):.0f}%)")
P(f"chance level of a blind guess among the absent team-mates: mean 1/absent = {np.mean(1 / v):.2f}; same image half: {np.mean(1 / np.maximum(s, 1)):.2f}")
P("per clip, median absent team-mates at a return:", {q: float(np.median([r['absent'] for r in rows if r['seq'] == q])) for q in data.DEV_CLIPS})
P("per clip, returns with another team-mate re-entering within 1 s:", {q: int(sum(r['together'] > 0 for r in rows if r['seq'] == q)) for q in data.DEV_CLIPS})
