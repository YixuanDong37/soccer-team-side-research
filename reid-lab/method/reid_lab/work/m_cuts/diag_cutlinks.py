"""DIAGNOSIS: links a cut suggests (the best alternative origin of the after side -> the after
sub-piece; the before sub-piece -> its best alternative continuation): how often are they right?
Grid over the two thresholds (margin over the cut's own cosine, absolute cosine). Uses the
held-out cuts of make_methods.py (methods/heldout_p{p0}_cuts.csv).

    python -I diag_cutlinks.py [p0]
"""
import sys

sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np
import pandas as pd
from reid_lab import data, score
import features as F

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
p0 = sys.argv[1] if len(sys.argv) > 1 else "0.7"
cuts = pd.read_csv(HERE + f"methods/heldout_p{p0}_cuts.csv")


def links_for(seq, df, margin, floor, sides=("d", "r")):
    cl = [[int(r.piece), int(r.f1)] for r in df.itertuples()]
    sub = F.subpiece_ids(cl)
    P = F.piece_table(seq)
    out = []
    for r in df.itertuples():
        if "d" in sides and r.alt_d_pid >= 0 and r.alt_d - r.cos >= margin and r.alt_d >= floor:
            q = P[int(r.alt_d_pid)]
            fq = int(q.frame[q.frame < r.f1][-1])
            out.append([F.sub_of(sub, q.pid, fq), F.sub_of(sub, int(r.piece), int(r.f1))])
        if "r" in sides and r.alt_r_pid >= 0 and r.alt_r - r.cos >= margin and r.alt_r >= floor:
            q = P[int(r.alt_r_pid)]
            fq = int(q.frame[q.frame > r.f0][0])
            out.append([F.sub_of(sub, int(r.piece), int(r.f0)), F.sub_of(sub, q.pid, fq)])
    return cl, out


for sides in (("d",), ("r",), ("d", "r")):
    print("sides", sides)
    for margin in (0.0, 0.05, 0.1, 0.15, 0.2):
        for floor in (0.0, 0.85, 0.9, 0.93, 0.95):
            m = {}
            for seq in data.DEV_CLIPS:
                cl, lk = links_for(seq, cuts[cuts.seq == seq], margin, floor, sides)
                m[seq] = {"links": lk, "cuts": cl}
            r = score.evaluate(m)["pooled"]
            print(f"  margin {margin:.2f} floor {floor:.2f}: links {r['links']:3d} right {r['links_right']:3d} wrong {r['links_wrong']:2d} unknown {r['links_unknown']:3d} | returns R/W/L {r['right']}/{r['wrong']}/{r['lost']} impure {r['impure_row_share']:.4f}")
