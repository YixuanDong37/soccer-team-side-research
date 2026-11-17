import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score
import pandas as pd, numpy as np
seq = "SNGS-162"
doc = json.loads((score.TRAIN/seq/"Labels-GameState.json").read_text(encoding="utf-8"))
print(doc.keys())
print(json.dumps(doc["images"][0]))
anns = [a for a in doc["annotations"] if a.get("supercategory") == "object"]
print(json.dumps(anns[0], indent=1))
print(json.dumps(anns[5000], indent=1))
print({k: v for k, v in doc.items() if k not in ("images", "annotations")}.__repr__()[:1500])
t = data.tracks(seq)
j = t[t.jersey.notna()]
print(len(j), j.groupby("track_id").jersey.agg(lambda s: s.value_counts().to_dict()).head(30))
print(j.jersey_conf.describe())
print(t.calib_status.value_counts(), t.pitch_usable.mean(), t.in_pitch.mean(), t.role.value_counts(dropna=False), t.team.value_counts(dropna=False))
print(data.refine(seq)["teams"]["counts"], data.refine(seq)["teams"]["params"])
vs = pd.DataFrame(data.refine(seq)["teams"]["tracks"])
print(vs.verdict.value_counts(dropna=False)); print(vs.head(10)); print(vs[vs.track_id>=100000])
