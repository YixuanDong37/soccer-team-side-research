"""v_repro_1 / trace_picks.py -- a second, deliberately different recomputation of the picked returns,
printed as raw numbers so that each outcome can be followed by eye. LABELS (check of the scorer only).

Differences from hand_check.py (so that a shared mistake is less likely):
- no Hungarian matching: a predicted box belongs to the annotated person whose box overlaps it most
  (IoU >= 0.5), and row A / B is the predicted box that overlaps the person's annotated box most;
- no union-find: the identity of a sub-piece is the set reached by a breadth-first walk over the links;
- the tracks table is used row by row (pandas filters), not through a matched-pair list.
Imports neither reid_lab nor gbf.

    python -I -B trace_picks.py METHOD.json PICKS.json
"""
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("C:/Research/gbf-data/gsr-runs")
TRAIN = Path("C:/Research/gbf-data/external/sn-gsr-2025-ce84ee7/train")
SUB = 10 ** 7


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    un = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / un if un > 0 else 0.0


def load(seq):
    doc = json.loads((TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    fr = {str(im["image_id"]): int(Path(im["file_name"]).stem) - 1 for im in doc["images"]}
    W, H = int(doc["images"][0]["width"]), int(doc["images"][0]["height"])
    ann = defaultdict(dict)        # frame -> person -> (box, role)
    for a in doc["annotations"]:
        if a.get("supercategory") != "object":
            continue
        role = (a.get("attributes") or {}).get("role")
        if role not in ("player", "goalkeeper", "referee"):
            continue
        b = a["bbox_image"]
        ann[fr[str(a["image_id"])]][int(a["track_id"])] = ((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]), role)
    t = pd.read_parquet(RUNS / seq / "gbf/refine/tracks.parquet")[["frame", "track_id", "x1", "y1", "x2", "y2"]]
    return ann, t, W, H


def person_of_row(ann, frame, box):
    """annotated person whose box overlaps this predicted box most (IoU >= 0.5), else None"""
    best, who = 0.0, None
    for p, (b, _) in ann.get(frame, {}).items():
        v = iou(b, box)
        if v > best:
            best, who = v, p
    return who if best >= 0.5 else None


def main():
    method = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["clips"]
    picks = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    cache = {}
    agree = 0
    for p in picks:
        seq, person, last, first = p["clip"], p["person"], p["last"], p["first"]
        if seq not in cache:
            cache[seq] = load(seq)
        ann, t, W, H = cache[seq]
        m = method[seq]
        cuts = defaultdict(list)
        for q, f in m.get("cuts") or []:
            cuts[int(q)].append(int(f))
        sub = lambda piece, frame: piece if piece < 0 else piece + SUB * sum(1 for c in cuts.get(piece, ()) if frame >= c)  # noqa: E731
        nb = defaultdict(set)
        for a, b in m.get("links") or []:
            nb[int(a)].add(int(b))
            nb[int(b)].add(int(a))

        def component(x):
            seen, dq = {x}, deque([x])
            while dq:
                y = dq.popleft()
                for z in nb[y]:
                    if z not in seen:
                        seen.add(z)
                        dq.append(z)
            return seen

        frames = sorted(f for f in ann if person in ann[f] and ann[f][person][1] in ("player", "goalkeeper"))
        assert last in frames and first in frames and not any(last < f < first for f in frames)
        bl, bf = ann[last][person][0], ann[first][person][0]
        dist = lambda b: min(b[0], b[1], W - 1 - b[2], H - 1 - b[3])  # noqa: E731
        print(f"== {seq} person {person} ({ann[last][person][1]}): annotated in frame {last}, next in frame {first}: {first - last - 1} frames missing")
        print(f"   box in frame {last}: {[round(v, 1) for v in bl]}, nearest border {dist(bl):.1f} px; "
              f"box in frame {first}: {[round(v, 1) for v in bf]}, nearest border {dist(bf):.1f} px (limit 15, image {W}x{H})")

        def find(start, step):
            for k in range(13):
                f = start + step * k
                if person not in ann.get(f, {}):
                    continue
                g = t[t.frame == f]
                best, row = 0.0, None
                for r in g.itertuples():
                    box = (1.5 * r.x1, 1.5 * r.y1, 1.5 * r.x2, 1.5 * r.y2)
                    v = iou(ann[f][person][0], box)
                    if v > best:
                        best, row = v, r
                if row is not None and best >= 0.5:
                    box = (1.5 * row.x1, 1.5 * row.y1, 1.5 * row.x2, 1.5 * row.y2)
                    owner = person_of_row(ann, f, box)
                    return {"frame": f, "piece": int(row.track_id), "iou": best, "owner": owner}
            return None

        A, B = find(last, -1), find(first, +1)
        if A is None or B is None:
            out = "undetected"
            print("   no predicted box overlaps the person within 12 frames of the gap ->", out)
        else:
            sa, sb = sub(A["piece"], A["frame"]), sub(B["piece"], B["frame"])
            print(f"   row A: frame {A['frame']}, IoU {A['iou']:.2f}, piece {A['piece']} -> sub-piece {sa} (cut frames of the piece: {sorted(cuts.get(A['piece'], []))}); "
                  f"that box overlaps person {A['owner']} most")
            print(f"   row B: frame {B['frame']}, IoU {B['iou']:.2f}, piece {B['piece']} -> sub-piece {sb} (cut frames of the piece: {sorted(cuts.get(B['piece'], []))}); "
                  f"that box overlaps person {B['owner']} most")
            ca, cb = component(sa), component(sb)
            if sa < 0 or sb < 0:
                out = "untracked"
            elif sb in ca:
                out = "right"
                print(f"   linked: {sorted(ca)} contains both -> right")
            else:
                print(f"   not linked. identity of A = {sorted(ca)}; identity of B = {sorted(cb)}")
                cnt = Counter()
                pieces = {x % SUB for x in cb}
                g = t[t.track_id.isin(pieces) & (t.frame < B["frame"])]
                for r in g.itertuples():
                    if sub(int(r.track_id), int(r.frame)) in cb:
                        who = person_of_row(ann, int(r.frame), (1.5 * r.x1, 1.5 * r.y1, 1.5 * r.x2, 1.5 * r.y2))
                        if who is not None:
                            cnt[who] += 1
                print(f"   boxes of the identity of B before frame {B['frame']}, by annotated person: {dict(cnt.most_common())}")
                if not cnt:
                    out = "lost"
                else:
                    top, n = cnt.most_common(1)[0]
                    out = "right" if top == person else ("wrong" if n >= 5 else "lost")
                print(f"   -> {out}")
        ok = out == p["outcome"]
        agree += ok
        print(f"   trace: {out}; hand_check.py: {p['outcome']}  {'AGREE' if ok else 'DIFFERENT'}")
    print(f"\n{agree} of {len(picks)} traced outcomes equal hand_check.py's")
    sys.exit(0 if agree == len(picks) else 1)


if __name__ == "__main__":
    main()
