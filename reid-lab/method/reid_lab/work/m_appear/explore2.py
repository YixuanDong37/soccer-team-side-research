import sys, os, time; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
import numpy as np, pandas as pd, json
seq="SNGS-107"
z = np.load(data.run_dir(seq) / "reid-embeddings.npz")
for k in z.files: print(k, z[k].shape, z[k].dtype)
print(np.linalg.norm(z["emb"], axis=1)[:10])
t=data.tracks(seq)
print(t.calib_status.value_counts().to_dict(), t.pitch_usable.mean(), t.in_pitch.mean(), t.jersey.notna().sum(), (t.track_id<0).sum())
print(json.dumps(data.refine(seq)["tracks"], indent=0)[:1500])
print(json.dumps(data.refine(seq)["teams"]["counts"]), json.dumps(data.refine(seq)["teams"]["params"]))
print(json.dumps(data.refine(seq)["roles"])[:800])
t0=time.time(); score.score_clip(seq, []); print("first", time.time()-t0)
t0=time.time(); 
for _ in range(10): score.score_clip(seq, [])
print("10x", time.time()-t0)
