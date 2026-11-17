"""DIAGNOSIS (labels): pieces that share frames. How many shared frames / what box overlap do same-person pairs have?"""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
import numpy as np, pandas as pd
from core import ctx
from features import protos, candidate_ids
import fitlabels
rows = []
for s in data.DEV_CLIPS:
    c = ctx(s); pr = protos(s); ids = candidate_ids(s); ident = fitlabels.piece_identity(s)
    t = c.t
    box = {}
    for p in ids:
        r = c.rows_of[p]; box[p] = {int(f): (c.x1[i], c.y1[i], c.x2[i], c.y2[i]) for f, i in zip(c.frame[r], r)}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sh = c.frames_of[a] & c.frames_of[b]
            if not sh or pr.ateam[a] != pr.ateam[b]: continue
            ious = []
            for f in sh:
                A, B = box[a][f], box[b][f]
                iw = min(A[2], B[2]) - max(A[0], B[0]); ih = min(A[3], B[3]) - max(A[1], B[1])
                inter = max(iw, 0) * max(ih, 0); ua = (A[2]-A[0])*(A[3]-A[1]) + (B[2]-B[0])*(B[3]-B[1]) - inter
                ious.append(inter / ua)
            lab = None if a not in ident or b not in ident else ident[a][0] == ident[b][0]
            rows.append(dict(seq=s, a=a, b=b, shared=len(sh), iou_med=float(np.median(ious)), iou_max=float(np.max(ious)), same=lab,
                             cos=float(pr.tc[a] @ pr.tc[b]), rows_a=len(c.rows_of[a]), rows_b=len(c.rows_of[b])))
d = pd.DataFrame(rows)
k = d[d.same.notna()]
print("same-team pairs sharing a frame:", len(d), "labelled", len(k), "same person", int(k.same.sum()))
print(k.groupby([pd.cut(k.shared, [0, 1, 2, 4, 9, 19, 49, 1000]), "same"]).size().unstack(fill_value=0))
sp = k[k.same == True]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(sp.sort_values("shared", ascending=False).head(40).to_string())
print("same-person pairs: iou_med quantiles", sp.iou_med.quantile([0, .1, .25, .5, .75, .9, 1]).round(2).to_dict())
dp = k[k.same == False]
print("different pairs: iou_med quantiles", dp.iou_med.quantile([0, .5, .9, .99, 1]).round(2).to_dict())
print("different pairs with shared>=5:", (dp.shared >= 5).sum(), " same-person with shared>=5:", (sp.shared >= 5).sum(), " with shared>=5 and iou_med<0.1:", ((sp.shared >= 5) & (sp.iou_med < 0.1)).sum())
