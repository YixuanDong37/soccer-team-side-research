import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import pandas as pd
def spans(f):
    f = np.sort(np.asarray(f)); brk = np.where(np.diff(f) > 1)[0]
    s = np.concatenate([[f[0]], f[brk+1]]); e = np.concatenate([f[brk], [f[-1]]])
    return [(int(a), int(b)) for a, b in zip(s, e)]
ml = pd.read_csv("out/mutual_links.csv")
for r in ml[ml.label == "wrong"].itertuples():
    c = ctx(r.clip)
    pa, pb = int(r.a_person), int(r.b_person)
    ga, gb = c.people_by[pa], c.people_by[pb]
    ov = len(set(ga.frame) & set(gb.frame))
    print(r.clip, r.earlier, "->", r.later, "cos", r.cos, "| person", pa, c.att[pa], spans(ga.frame), "| person", pb, c.att[pb], spans(gb.frame), "| co-annotated frames:", ov)
    print("    piece A", c.P[r.earlier]["first"], c.P[r.earlier]["last"], "piece B", c.P[r.later]["first"], c.P[r.later]["last"])
