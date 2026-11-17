import sys
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
import pandas as pd, numpy as np
for seq in data.DEV_CLIPS:
    imp = pd.read_parquet(data.run_dir(seq) / "gbf/import/tracks.parquet")
    t = data.tracks(seq)
    key = lambda d: set(zip(d.frame, d.x1.round(2), d.y1.round(2), d.x2.round(2), d.y2.round(2)))
    print(seq, "import rows", len(imp), "refine rows", len(t), "import-only boxes", len(key(imp) - key(t)), "refine-only", len(key(t) - key(imp)), "untracked refine", int((t.track_id < 0).sum()), "conf<0.3 rows", int((t.conf < 0.3).sum()), "conf==0", int((t.conf == 0).sum()))
print(imp.columns.tolist())
