"""DIAGNOSIS (labels): are some annotated identities in SNGS-162 swapped across a gap?"""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
from gbf.evaluate import gsr
import numpy as np, pandas as pd
import fitlabels
seq = sys.argv[1] if len(sys.argv) > 1 else "SNGS-162"
refs_of_interest = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [12, 14, 9, 18, 19, 21]
ref = gsr.load_reference(score.TRAIN / seq / "Labels-GameState.json")
pe = ref.people
print(pe.columns.tolist())
c = score.clip(seq); t = data.tracks(seq)
pd.set_option("display.width", 250)
for rid in refs_of_interest:
    g = pe[pe.ref_id == rid].sort_values("frame")
    f = g.frame.to_numpy(); brk = np.where(np.diff(f) > 1)[0]
    segs = np.split(np.arange(len(f)), brk + 1)
    print(f"\nref {rid}: role {g.role.iloc[0]} team {g.team.iloc[0] if 'team' in g else None} jersey {g.jersey.dropna().unique().tolist() if 'jersey' in g else None} rows {len(g)}")
    for s in segs:
        a, b = g.iloc[s[0]], g.iloc[s[-1]]
        cols = [k for k in ("pitch_x", "pitch_y", "x", "y") if k in g.columns]
        # pieces matched in this stretch
        m = (c.refs == rid) & (c.frames >= a.frame) & (c.frames <= b.frame)
        pcs = pd.Series(c.piece[m]).value_counts().head(4).to_dict()
        jr = {}
        for p in pcs:
            tt = t[(t.track_id == p) & t.jersey.notna()]
            if len(tt): jr[p] = tt.jersey.value_counts().head(3).to_dict()
        print(f"  frames {int(a.frame):4d}-{int(b.frame):4d}  start box ({a.x1:.0f},{a.y1:.0f},{a.x2:.0f},{a.y2:.0f}) end box ({b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f})"
              + "".join(f" {k} {a[k]:.1f}->{b[k]:.1f}" for k in cols) + f" | pieces {pcs} | OCR {jr}")
