"""DIAGNOSIS (labels): annotated jersey / team per identity and stretch, against the tracker's OCR reads."""
import sys, os, json; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
import numpy as np, pandas as pd
seq = sys.argv[1]
refs = [int(x) for x in sys.argv[2].split(",")]
J = json.loads((score.TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
print(J.keys()); print(J["annotations"][0]); print(J["images"][0])
img = {im["image_id"]: int(im["file_name"].split(".")[0]) - 1 for im in J["images"]}
rows = []
for a in J["annotations"]:
    if a.get("track_id") in refs and a.get("bbox_image"):
        at = a.get("attributes") or {}
        rows.append((a["track_id"], img[a["image_id"]], at.get("role"), at.get("team"), at.get("jersey")))
d = pd.DataFrame(rows, columns=["ref", "frame", "role", "team", "jersey"]).sort_values(["ref", "frame"])
for rid, g in d.groupby("ref"):
    f = g.frame.to_numpy(); brk = np.where(np.diff(f) > 1)[0]
    for s in np.split(np.arange(len(f)), brk + 1):
        gg = g.iloc[s]
        print(rid, int(gg.frame.iloc[0]), int(gg.frame.iloc[-1]), gg.role.value_counts(dropna=False).to_dict(), gg.team.value_counts(dropna=False).to_dict(), gg.jersey.value_counts(dropna=False).to_dict())
