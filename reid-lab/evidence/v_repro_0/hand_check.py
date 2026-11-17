"""v_repro / hand_check.py -- INDEPENDENT recomputation of return outcomes (LABELS; a check of the scorer).

Imports neither reid_lab (data.py, score.py) nor gbf: only json, numpy, pandas, scipy. It reads
  - the annotation file Labels-GameState.json of a dev clip,
  - gbf's refine/tracks.parquet of the clip,
  - a METHOD.json (links between pieces, cuts)
and recomputes, from the rules as written in the task text and in score.py's docstring, the outcome
of every annotated return (right / wrong / lost / undetected / untracked).

    python -I -B hand_check.py METHOD.json OUT.json [seed] [n]

Writes OUT.json: every return of the 13 dev clips with its outcome and the evidence behind it, and
the n (default 10) scored returns drawn at random (numpy default_rng(seed), default 20261115) from
the pooled list of scored returns in (clip order, person id, gap start) order.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

RUNS = Path("C:/Research/gbf-data/gsr-runs")
TRAIN = Path("C:/Research/gbf-data/external/sn-gsr-2025-ce84ee7/train")
DEV = ("SNGS-107", "SNGS-163", "SNGS-106", "SNGS-097", "SNGS-067", "SNGS-100", "SNGS-066",
       "SNGS-161", "SNGS-169", "SNGS-162", "SNGS-151", "SNGS-102", "SNGS-112")
SCALE = 1.5          # 720p boxes -> 1080p annotation
IOU_MIN = 0.5
GAP_MIN = 10         # frames missing
EDGE = 15.0          # px at 1080p
SLACK = 12           # frames next to the gap in which row A / row B may lie
MIN_ROWS = 5         # rows of another person that make a "wrong"
SUB = 10**7


def read_labels(seq: str):
    doc = json.loads((TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    frame_of, sizes = {}, set()
    for im in doc["images"]:
        frame_of[str(im["image_id"])] = int(Path(im["file_name"]).stem) - 1      # image N.jpg is frame N - 1
        sizes.add((int(im["width"]), int(im["height"])))
    assert len(sizes) == 1, sizes
    W, H = sizes.pop()
    rows = []
    for a in doc["annotations"]:
        if a.get("supercategory") != "object":
            continue
        role = (a.get("attributes") or {}).get("role")
        if role not in ("player", "goalkeeper", "referee"):
            continue
        b = a["bbox_image"]
        rows.append((frame_of[str(a["image_id"])], int(a["track_id"]), role,
                     float(b["x"]), float(b["y"]), float(b["x"]) + float(b["w"]), float(b["y"]) + float(b["h"])))
    people = pd.DataFrame(rows, columns=["frame", "ref", "role", "x1", "y1", "x2", "y2"])
    return people.sort_values(["frame", "ref"]).reset_index(drop=True), W, H


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((len(a), len(b)))
    for i, (ax1, ay1, ax2, ay2) in enumerate(a):
        for j, (bx1, by1, bx2, by2) in enumerate(b):
            w = min(ax2, bx2) - max(ax1, bx1)
            h = min(ay2, by2) - max(ay1, by1)
            if w <= 0 or h <= 0:
                continue
            inter = w * h
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            out[i, j] = inter / union if union > 0 else 0.0
    return out


def match_clip(people: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Per frame: Hungarian assignment maximising the summed IoU (IoU < 0.5 counts 0); pairs with
    IoU >= 0.5 are kept. One row per matched pair: frame, ref, piece (track_id), iou."""
    out = []
    pg = {f: g for f, g in people.groupby("frame")}
    tg = {f: g for f, g in tracks.groupby("frame")}
    for f in sorted(set(pg) & set(tg)):
        p, t = pg[f], tg[f]
        iou = iou_matrix(p[["x1", "y1", "x2", "y2"]].to_numpy(float), t[["x1", "y1", "x2", "y2"]].to_numpy(float) * SCALE)
        score = np.where(iou >= IOU_MIN, iou, 0.0)
        r, c = linear_sum_assignment(score, maximize=True)
        for i, j in zip(r, c):
            if iou[i, j] >= IOU_MIN:
                out.append((int(f), int(p.ref.iloc[i]), int(t.track_id.iloc[j]), float(iou[i, j])))
    return pd.DataFrame(out, columns=["frame", "ref", "piece", "iou"])


def find_returns(people: pd.DataFrame, W: int, H: int) -> list[dict]:
    near = lambda r: bool(r.x1 <= EDGE or r.y1 <= EDGE or r.x2 >= W - 1 - EDGE or r.y2 >= H - 1 - EDGE)  # noqa: E731
    out = []
    pk = people[people.role.isin(["player", "goalkeeper"])]
    for ref, g in pk.groupby("ref"):
        g = g.sort_values("frame")
        fr = g.frame.to_numpy()
        for i in range(len(fr) - 1):
            gap = int(fr[i + 1] - fr[i] - 1)
            if gap < GAP_MIN:
                continue
            before, after = g.iloc[i], g.iloc[i + 1]
            if not (near(before) or near(after)):
                continue
            out.append({"ref": int(ref), "last": int(fr[i]), "first": int(fr[i + 1]), "gap": gap,
                        "edge_before": near(before), "edge_after": near(after)})
    return out


class Groups:
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
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def clip_outcomes(seq: str, links, cuts) -> list[dict]:
    people, W, H = read_labels(seq)
    tracks = pd.read_parquet(RUNS / seq / "gbf/refine/tracks.parquet").reset_index(drop=True)
    m = match_clip(people, tracks)
    # identities after the method: cuts give the rows of a piece from the cut frame on a new id
    ident = m.piece.to_numpy().copy()
    by = {}
    for p, f in cuts or []:
        by.setdefault(int(p), []).append(int(f))
    for p, fs in by.items():
        for j, f in enumerate(sorted(fs)):
            ident[(m.piece.to_numpy() == p) & (m.frame.to_numpy() >= f)] = p + SUB * (j + 1)
    G = Groups()
    for a, b in links or []:
        G.union(int(a), int(b))
    # group id per matched row; an untracked row (piece -1) is its own group
    grp = np.array([G.find(int(x)) if x >= 0 else -(k + 1) for k, x in enumerate(ident)], dtype=np.int64)
    m = m.assign(ident=ident, grp=grp)
    at = {(int(f), int(r)): k for k, (f, r) in enumerate(zip(m.frame, m.ref))}
    res = []
    for r in find_returns(people, W, H):
        a = next((at[(r["last"] - k, r["ref"])] for k in range(SLACK + 1) if (r["last"] - k, r["ref"]) in at), None)
        b = next((at[(r["first"] + k, r["ref"])] for k in range(SLACK + 1) if (r["first"] + k, r["ref"]) in at), None)
        d = {"clip": seq, **r}
        if a is None or b is None:
            d.update(outcome="undetected", row_a=None if a is None else int(m.frame[a]), row_b=None if b is None else int(m.frame[b]))
            res.append(d)
            continue
        d.update(row_a={"frame": int(m.frame[a]), "piece": int(m.piece[a]), "id_after_cuts": int(m.ident[a]), "group": int(m.grp[a])},
                 row_b={"frame": int(m.frame[b]), "piece": int(m.piece[b]), "id_after_cuts": int(m.ident[b]), "group": int(m.grp[b])})
        if m.piece[a] < 0 or m.piece[b] < 0:
            d["outcome"] = "untracked"
        elif m.grp[a] == m.grp[b]:
            d["outcome"] = "right"
            d["why"] = "row A and row B have the same identity"
        else:
            earlier = m[(m.frame < m.frame[b]) & (m.grp == m.grp[b])]
            cnt = Counter(earlier.ref.tolist())
            d["earlier_rows_of_B_identity"] = {str(k): int(v) for k, v in cnt.most_common()}
            if not cnt:
                d["outcome"], d["why"] = "lost", "row B's identity has no matched row before B: new identity"
            else:
                best = max(cnt.values())
                tops = sorted(k for k, v in cnt.items() if v == best)
                d["tie_for_majority"] = len(tops) > 1
                if r["ref"] in tops and len(tops) == 1:
                    d["outcome"], d["why"] = "right", "row B's identity was mostly this person before (joins an earlier stretch)"
                elif r["ref"] in tops:
                    d["outcome"], d["why"] = "ambiguous", "tie for the majority person that includes this person"
                elif best >= MIN_ROWS:
                    d["outcome"], d["why"] = "wrong", f"row B's identity was mostly person {tops[0]} before ({best} rows)"
                else:
                    d["outcome"], d["why"] = "lost", f"row B's identity had only {best} rows of another person before"
        res.append(d)
    return res


if __name__ == "__main__":
    method = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["clips"]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261115
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    allr = []
    for seq in DEV:
        mm = method.get(seq) or {}
        rr = clip_outcomes(seq, mm.get("links") or [], mm.get("cuts") or [])
        allr += rr
        c = Counter(x["outcome"] for x in rr)
        print(seq, dict(c), flush=True)
    c = Counter(x["outcome"] for x in allr)
    print("pooled", dict(c))
    scored = [k for k, x in enumerate(allr) if x["outcome"] in ("right", "wrong", "lost", "ambiguous")]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(scored), size=n, replace=False).tolist())
    sample = [scored[k] for k in pick]
    Path(sys.argv[2]).write_text(json.dumps({"method": sys.argv[1], "seed": seed, "pooled": dict(c), "sample_index": sample,
                                             "returns": allr}, indent=1), encoding="utf-8")
    print(f"\n{n} random scored returns (seed {seed}):")
    for k in sample:
        x = allr[k]
        print(json.dumps({kk: x[kk] for kk in ("clip", "ref", "last", "first", "gap", "row_a", "row_b", "outcome", "why") if kk in x}
                         | ({"earlier": x["earlier_rows_of_B_identity"]} if "earlier_rows_of_B_identity" in x else {})))
