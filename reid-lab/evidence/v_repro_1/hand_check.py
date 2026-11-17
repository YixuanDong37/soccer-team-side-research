"""v_repro_1 / hand_check.py -- INDEPENDENT recomputation of the outcome of every annotated return for a
METHOD file, from refine/tracks.parquet and Labels-GameState.json only.

LABELS: this is a check of the scorer, never a method. It imports neither reid_lab nor gbf: the
annotation and the tracks table are read directly, boxes are matched per frame here (IoU >= 0.5 on
1080p coordinates, Hungarian), returns are found here, and the outcome rules of the task text are
applied here.

    python -I -B hand_check.py METHOD.json OUT.json [seed n]
      OUT.json: every annotated return with its outcome and the evidence behind it
      seed n  : also print the full trace of n random SCORED returns (numpy default_rng(seed))
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

RUNS = Path("C:/Research/gbf-data/gsr-runs")
TRAIN = Path("C:/Research/gbf-data/external/sn-gsr-2025-ce84ee7/train")
DEV = ["SNGS-107", "SNGS-163", "SNGS-106", "SNGS-097", "SNGS-067", "SNGS-100", "SNGS-066",
       "SNGS-161", "SNGS-169", "SNGS-162", "SNGS-151", "SNGS-102", "SNGS-112"]
PEOPLE = {"player", "goalkeeper", "referee"}
SUB = 10 ** 7


def read_labels(seq):
    doc = json.loads((TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    frame_of, sizes = {}, set()
    for im in doc["images"]:
        frame_of[str(im["image_id"])] = int(Path(im["file_name"]).stem) - 1     # image N.jpg = frame N - 1
        sizes.add((int(im["width"]), int(im["height"])))
    assert len(sizes) == 1, sizes
    (W, H), = sizes
    rows = []
    for a in doc["annotations"]:
        if a.get("supercategory") != "object":
            continue
        role = (a.get("attributes") or {}).get("role")
        if role not in PEOPLE:
            continue
        b = a["bbox_image"]
        rows.append((frame_of[str(a["image_id"])], int(a["track_id"]), role, float(b["x"]), float(b["y"]),
                     float(b["x"]) + float(b["w"]), float(b["y"]) + float(b["h"])))
    return rows, W, H


def iou_matrix(A, B):
    ix = np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0])
    iy = np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    un = aa[:, None] + bb[None, :] - inter
    return np.where(un > 0, inter / np.where(un > 0, un, 1), 0.0)


def match_clip(seq):
    """-> list of matched (frame, person, piece id, iou), the annotation rows, W, H"""
    lab, W, H = read_labels(seq)
    t = pd.read_parquet(RUNS / seq / "gbf/refine/tracks.parquet")
    by_frame_ref = defaultdict(list)
    for r in sorted(lab, key=lambda r: (r[0], r[1])):
        by_frame_ref[r[0]].append(r)
    by_frame_pred = defaultdict(list)
    for f, tid, x1, y1, x2, y2 in zip(t.frame.to_numpy(), t.track_id.to_numpy(), t.x1.to_numpy(float), t.y1.to_numpy(float),
                                      t.x2.to_numpy(float), t.y2.to_numpy(float)):
        by_frame_pred[int(f)].append((int(tid), 1.5 * x1, 1.5 * y1, 1.5 * x2, 1.5 * y2))      # 720p -> 1080p
    matched = []
    for f in sorted(set(by_frame_ref) & set(by_frame_pred)):
        R, P = by_frame_ref[f], by_frame_pred[f]
        iou = iou_matrix(np.array([r[3:7] for r in R]), np.array([p[1:5] for p in P]))
        rr, cc = linear_sum_assignment(np.where(iou >= 0.5, iou, 0.0), maximize=True)
        for i, j in zip(rr, cc):
            if iou[i, j] >= 0.5:
                matched.append((f, R[i][1], P[j][0], float(iou[i, j])))
    return matched, lab, W, H


def sub_id(piece, frame, cuts_by_piece):
    """the rows of a piece from its j-th cut frame on are piece + 10**7 * j (j = 1, 2, ...)"""
    if piece < 0:
        return piece
    j = sum(1 for c in cuts_by_piece.get(piece, ()) if frame >= c)
    return piece + SUB * j


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def hand_clip(seq, links, cuts):
    matched, lab, W, H = match_clip(seq)
    cuts_by_piece = defaultdict(list)
    for p, f in cuts:
        cuts_by_piece[int(p)].append(int(f))
    cuts_by_piece = {p: sorted(set(fs)) for p, fs in cuts_by_piece.items()}
    uf = UF()
    for a, b in links:
        uf.union(int(a), int(b))
    # matched rows with their identity (group of sub-pieces); untracked rows have no identity
    rows = []
    for f, person, piece, iou in matched:
        sid = sub_id(piece, f, cuts_by_piece)
        rows.append({"frame": f, "person": person, "piece": piece, "sub": sid, "group": uf.find(sid) if sid >= 0 else None})
    at = {(r["frame"], r["person"]): r for r in rows}
    by_group = defaultdict(list)
    for r in rows:
        if r["group"] is not None:
            by_group[r["group"]].append(r)

    def border(b):
        return b[3] <= 15 or b[4] <= 15 or b[5] >= W - 1 - 15 or b[6] >= H - 1 - 15

    per = defaultdict(list)
    for r in lab:
        if r[2] in ("player", "goalkeeper"):
            per[r[1]].append(r)
    out = []
    for person in sorted(per):
        boxes = sorted(per[person], key=lambda r: r[0])
        for u, v in zip(boxes[:-1], boxes[1:]):
            gap = v[0] - u[0] - 1
            if gap < 10 or not (border(u) or border(v)):
                continue
            A = next((at[(u[0] - k, person)] for k in range(13) if (u[0] - k, person) in at), None)
            B = next((at[(v[0] + k, person)] for k in range(13) if (v[0] + k, person) in at), None)
            rec = {"clip": seq, "person": person, "last": u[0], "first": v[0], "gap": gap, "A": A, "B": B}
            if A is None or B is None:
                rec["outcome"] = "undetected"
            elif A["group"] is None or B["group"] is None:
                rec["outcome"] = "untracked"
            elif A["group"] == B["group"]:
                rec["outcome"] = "right"
                rec["why"] = "A and B have the same identity"
            else:
                before = Counter(r["person"] for r in by_group[B["group"]] if r["frame"] < B["frame"])
                rec["before"] = {str(k): v for k, v in before.most_common()}
                if not before:
                    rec["outcome"], rec["why"] = "lost", "the identity of B has no matched box before B"
                else:
                    ranked = before.most_common()
                    top, n = ranked[0]
                    rec["tie"] = len(ranked) > 1 and ranked[1][1] == n
                    if top == person:
                        rec["outcome"], rec["why"] = "right", "the identity of B was mostly this person before"
                    elif n >= 5:
                        rec["outcome"], rec["why"] = "wrong", f"the identity of B was mostly person {top} before ({n} boxes)"
                    else:
                        rec["outcome"], rec["why"] = "lost", f"the identity of B held only {n} boxes of person {top} before"
            out.append(rec)
    # link verdicts and impure rows, also recomputed here
    ident = {}
    by_sub = defaultdict(list)
    for r in rows:
        if r["sub"] >= 0:
            by_sub[r["sub"]].append(r["person"])
    for s, ps in by_sub.items():
        top, n = Counter(ps).most_common(1)[0]
        if n >= 5:
            ident[s] = top
    lk = Counter()
    for a, b in links:
        a, b = int(a), int(b)
        lk["unknown" if (a not in ident or b not in ident) else ("right" if ident[a] == ident[b] else "wrong")] += 1
    impure = 0
    for g, rs in by_group.items():
        c = Counter(r["person"] for r in rs)
        impure += len(rs) - c.most_common(1)[0][1]
    tracked = sum(1 for r in rows if r["group"] is not None)
    return out, {"links_right": lk["right"], "links_wrong": lk["wrong"], "links_unknown": lk["unknown"], "impure": impure, "tracked": tracked,
                 "matched": len(rows)}


def main():
    method = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["clips"]
    allr, extra = [], {}
    for seq in DEV:
        m = method.get(seq) or {}
        r, e = hand_clip(seq, m.get("links") or [], m.get("cuts") or [])
        allr += r
        extra[seq] = e
    c = Counter(r["outcome"] for r in allr)
    tot = {k: sum(e[k] for e in extra.values()) for k in ("links_right", "links_wrong", "links_unknown", "impure", "tracked")}
    print("annotated returns:", len(allr), dict(c))
    print("links right / wrong / unknown:", tot["links_right"], tot["links_wrong"], tot["links_unknown"],
          "| impure rows", tot["impure"], "of", tot["tracked"], "=", round(tot["impure"] / tot["tracked"], 5))
    print("ties in the majority of the identity of B:", sum(1 for r in allr if r.get("tie")))
    for seq in DEV:
        cc = Counter(r["outcome"] for r in allr if r["clip"] == seq)
        print(f"  {seq}: right {cc['right']} wrong {cc['wrong']} lost {cc['lost']} undetected {cc['undetected']} untracked {cc['untracked']} | "
              f"links {extra[seq]['links_right']} / {extra[seq]['links_wrong']} / {extra[seq]['links_unknown']} | impure {extra[seq]['impure']}")
    Path(sys.argv[2]).write_text(json.dumps({"returns": allr, "per_clip": extra}, indent=1), encoding="utf-8")
    if len(sys.argv) > 4:
        seed, n = int(sys.argv[3]), int(sys.argv[4])
        scored = [i for i, r in enumerate(allr) if r["outcome"] in ("right", "wrong", "lost")]
        pick = sorted(np.random.default_rng(seed).choice(len(scored), size=n, replace=False).tolist())
        print(f"\n{n} random scored returns (default_rng({seed}) over the {len(scored)} scored returns in clip / person / frame order):")
        for k in pick:
            r = allr[scored[k]]
            A, B = r["A"], r["B"]
            print(f"- #{k} {r['clip']} person {r['person']}: annotated until frame {r['last']}, again from {r['first']} (gap {r['gap']})")
            print(f"    row A: frame {A['frame']}, piece {A['piece']}, sub-piece {A['sub']}, identity {A['group']}")
            print(f"    row B: frame {B['frame']}, piece {B['piece']}, sub-piece {B['sub']}, identity {B['group']}")
            if "before" in r:
                print(f"    matched boxes of the identity of B before frame {B['frame']} by person: {r['before']}")
            print(f"    -> {r['outcome']}: {r['why']}")
        Path(sys.argv[2]).with_suffix(".picks.json").write_text(json.dumps(
            [{"k": k, **{x: allr[scored[k]][x] for x in ("clip", "person", "last", "first", "outcome")}} for k in pick], indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
