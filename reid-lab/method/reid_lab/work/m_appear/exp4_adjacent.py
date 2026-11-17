"""IN-SAMPLE exploration (not appearance): links between a piece's end and a piece starting right after at the same place.
Adds [a, b] when b starts 1..max_gap frames after a ends and IoU(a's last box, b's first box) >= min_iou, each piece used once per side."""
import sys, os, json; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
import numpy as np
from core import ctx
from exp2_agglo import show

def adjacent_links(seq, max_gap=3, min_iou=0.5):
    c = ctx(seq); P = c.P
    ids = [int(p) for p, r in zip(P.piece, P.role) if r != "referee"]
    first = {p: c.rows_of[p][0] for p in ids}; last = {p: c.rows_of[p][-1] for p in ids}
    cand = []
    for a in ids:
        ra = last[a]
        for b in ids:
            if a == b: continue
            rb = first[b]; gap = c.frame[rb] - c.frame[ra]
            if gap < 1 or gap > max_gap: continue
            iw = min(c.x2[ra], c.x2[rb]) - max(c.x1[ra], c.x1[rb]); ih = min(c.y2[ra], c.y2[rb]) - max(c.y1[ra], c.y1[rb])
            inter = max(iw, 0) * max(ih, 0); u = c.w[ra] * c.h[ra] + c.w[rb] * c.h[rb] - inter
            if inter / u >= min_iou: cand.append((inter / u, a, b))
    used_a, used_b, out = set(), set(), []
    for iou, a, b in sorted(cand, reverse=True):
        if a in used_a or b in used_b: continue
        used_a.add(a); used_b.add(b); out.append([a, b])
    return out

if __name__ == "__main__":
    base = json.load(open("out/agglo-lam3.method.json", encoding="utf-8"))["clips"]
    show("LOCO agglo lam=3", base)
    for max_gap in (1, 3, 10):
        for min_iou in (0.3, 0.5, 0.7):
            adj = {s: adjacent_links(s, max_gap, min_iou) for s in data.DEV_CLIPS}
            show(f"adjacent only gap<={max_gap} iou>={min_iou}", {s: {"links": adj[s]} for s in adj})
            show(f"  + LOCO agglo lam=3", {s: {"links": base[s]["links"] + adj[s]} for s in adj})
