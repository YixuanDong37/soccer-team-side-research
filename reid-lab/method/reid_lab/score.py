"""Label-based scoring of re-identification methods on the 13 SN-GSR dev clips.

Same rules as analysis/gsr_reentry_eval.py (checked by check_score.py): gbf's predicted boxes are
scaled x1.5 and matched to the annotation with gbf.evaluate.gsr.match; a return is an annotated
player or goalkeeper missing for >= 10 frames whose box before or after the gap is within 15 px
(1080p) of the image border; row A / row B are the person's matched rows next to the gap (within
12 frames). Outcome under an identity assignment g: right if g(A) == g(B); else among g(B)'s
matched rows before B: none -> lost; mostly the same person -> right; mostly another person with
>= 5 rows -> wrong; else lost. Returns without both rows are "undetected", with an untracked row
"untracked"; neither is scored.

A method is links between gbf pieces (tracks.track_id), optionally after cuts: a cut (piece,
frame) gives the piece's rows from that frame on a new id, piece + 10**7 * (j + 1) for the j-th cut
of that piece by frame (link those ids like any other). The assignment is union-find over links.

Also reported, so that a method cannot buy returns with wrong merges elsewhere:
- link precision: a link between two (sub)pieces whose identities are known (>= 5 matched rows,
  majority person) is right when they are the same person;
- impure_rows: share of matched tracked rows whose group's majority person is someone else.

CLI (gbf's .venv): python -I score.py METHOD.json [OUT.json]
METHOD.json = {"name": str, "clips": {"SNGS-107": {"links": [[a, b], ...], "cuts": [[piece, frame], ...]}, ...}}
A dev clip missing from "clips" is scored with no links and listed under "clips_without_entry".
Python: from reid_lab import score; score.evaluate({"SNGS-107": {"links": [...]}}) -> dict.
Baselines: score.baseline_links(name) for "split", "plugin", "mutual" (appearance mutual best,
0.05 margin, no bar).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from gbf.evaluate import gsr

try:
    from . import data
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reid_lab import data

TRAIN = Path(r"C:\Research\gbf-data\external\sn-gsr-2025-ce84ee7\train")
SCALE, GAP_MIN, EDGE_PX, MIN_MATCHED, MATCH_SLACK = 1.5, 10, 15.0, 5, 12
SUB = 10**7


@dataclass
class Clip:
    seq: str
    frames: np.ndarray      # per matched pair
    refs: np.ndarray
    piece: np.ndarray       # gbf piece id (-1 untracked)
    src: np.ndarray         # plugin id
    returns: list[dict]     # with "a"/"b" pair indices (None when missing)


@lru_cache(maxsize=None)
def clip(seq: str) -> Clip:
    ref = gsr.load_reference(TRAIN / seq / "Labels-GameState.json")
    t = data.tracks(seq)
    pred = t[["frame", "track_id", "x1", "y1", "x2", "y2"]].copy()
    for c in ("x1", "y1", "x2", "y2"):
        pred[c] = pred[c] * SCALE
    pairs, _ = gsr.match(ref, pred)
    pred_row = pairs["pred_row"].to_numpy(dtype=np.int64)
    frames = pairs["frame"].to_numpy(dtype=np.int64)
    refs = pairs["ref_id"].to_numpy(dtype=np.int64)
    piece = pairs["track_id"].to_numpy(dtype=np.int64)
    src = t["source_track_id"].to_numpy(dtype=np.int64)[pred_row] if len(pred_row) else np.zeros(0, np.int64)
    row_of = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(frames, refs))}

    def find_row(rid: int, frame: int, step: int):
        for k in range(MATCH_SLACK + 1):
            i = row_of.get((frame + step * k, rid))
            if i is not None:
                return i
        return None

    people, W, H = ref.people, ref.width, ref.height
    near = lambda r: bool(r.x1 <= EDGE_PX or r.y1 <= EDGE_PX or r.x2 >= W - 1 - EDGE_PX or r.y2 >= H - 1 - EDGE_PX)  # noqa: E731
    returns = []
    for rid, g in people[people["role"].isin(["player", "goalkeeper"])].groupby("ref_id"):
        g = g.sort_values("frame")
        f = g["frame"].to_numpy()
        for i in np.where(np.diff(f) - 1 >= GAP_MIN)[0]:
            before, after = g.iloc[i], g.iloc[i + 1]
            if not (near(before) or near(after)):
                continue
            returns.append({"ref_id": int(rid), "last": int(f[i]), "first": int(f[i + 1]), "gap": int(f[i + 1] - f[i] - 1),
                            "a": find_row(int(rid), int(f[i]), -1), "b": find_row(int(rid), int(f[i + 1]), +1)})  # fmt: skip
    return Clip(seq, frames, refs, piece, src, returns)


class _Groups:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _ids_after_cuts(c: Clip, cuts) -> np.ndarray:
    ids = c.piece.copy()
    by_piece: dict[int, list[int]] = {}
    for p, f in cuts or []:
        by_piece.setdefault(int(p), []).append(int(f))
    for p, fs in by_piece.items():
        if not (c.piece == p).any() and p not in set(data.tracks(c.seq).track_id.tolist()):
            raise ValueError(f"{c.seq}: cut names unknown piece {p}")
        for j, f in enumerate(sorted(fs)):
            ids[(c.piece == p) & (c.frames >= f)] = p + SUB * (j + 1)
    return ids


def _identities(ids: np.ndarray, refs: np.ndarray) -> dict[int, int]:
    ident = {}
    for tid in np.unique(ids[ids >= 0]):
        top, n = Counter(refs[ids == tid].tolist()).most_common(1)[0]
        if n >= MIN_MATCHED:
            ident[int(tid)] = int(top)
    return ident


def score_clip(seq: str, links=(), cuts=()) -> dict:
    c = clip(seq)
    ids = _ids_after_cuts(c, cuts)
    known = set(np.unique(ids[ids >= 0]).tolist())
    valid = set(data.tracks(seq).track_id[data.tracks(seq).track_id >= 0].tolist())
    for p, f in cuts or []:
        valid.update(int(p) + SUB * (j + 1) for j in range(len(cuts)))
    groups = _Groups()
    for a, b in links or []:
        a, b = int(a), int(b)
        if a not in valid or b not in valid:
            raise ValueError(f"{seq}: link names unknown piece {a if a not in valid else b}")
        groups.union(a, b)
    tracked = ids >= 0
    alone = -1 - np.arange(len(ids), dtype=np.int64)
    g = np.where(tracked, np.array([groups.find(int(x)) if x >= 0 else -1 for x in ids], dtype=np.int64), alone)

    out = Counter()
    per_return = []
    for r in c.returns:
        a, b = r["a"], r["b"]
        if a is None or b is None:
            out["undetected"] += 1
            per_return.append("undetected")
            continue
        if not (tracked[a] and tracked[b]):
            out["untracked"] += 1
            per_return.append("untracked")
            continue
        if g[a] == g[b]:
            o = "right"
        else:
            counts = Counter(c.refs[(c.frames < c.frames[b]) & (g == g[b])].tolist())
            if not counts:
                o = "lost"
            else:
                top, n = counts.most_common(1)[0]
                o = "right" if top == r["ref_id"] else ("wrong" if n >= MIN_MATCHED else "lost")
        out[o] += 1
        per_return.append(o)

    ident = _identities(ids, c.refs)
    lk = Counter()
    for a, b in links or []:
        a, b = int(a), int(b)
        if a in ident and b in ident:
            lk["right" if ident[a] == ident[b] else "wrong"] += 1
        else:
            lk["unknown"] += 1
    # impurity: matched tracked rows whose group's majority person is someone else
    gt = g[tracked]
    rt = c.refs[tracked]
    df = pd.DataFrame({"g": gt, "r": rt})
    major = df.groupby("g")["r"].agg(lambda s: s.value_counts().index[0])
    impure = int((df["r"].to_numpy() != major.loc[df["g"]].to_numpy()).sum())
    return {"returns": {k: int(out.get(k, 0)) for k in ("right", "wrong", "lost", "undetected", "untracked")},
            "per_return": per_return, "links": {"n": len(links or []), **{k: int(lk.get(k, 0)) for k in ("right", "wrong", "unknown")}},
            "impure_rows": impure, "tracked_rows": int(tracked.sum()), "unknown_link_ids": sorted(set().union(*[set(map(int, l)) for l in links or [[]]]) - known)}  # fmt: skip


def evaluate(method: dict, clips=data.DEV_CLIPS) -> dict:
    """method: {seq: {"links": [...], "cuts": [...]}} -> pooled and per-clip scores."""
    per = {}
    for seq in clips:
        m = method.get(seq) or {}
        per[seq] = score_clip(seq, m.get("links") or [], m.get("cuts") or [])
    tot = Counter()
    lk = Counter()
    imp = rows = 0
    for s in per.values():
        tot.update(s["returns"])
        lk.update({k: v for k, v in s["links"].items()})
        imp += s["impure_rows"]
        rows += s["tracked_rows"]
    scored = tot["right"] + tot["wrong"] + tot["lost"]
    judged = lk["right"] + lk["wrong"]
    pooled = {"scored": scored, **{k: int(tot[k]) for k in ("right", "wrong", "lost", "undetected", "untracked")},
              "right_share": round(tot["right"] / scored, 4) if scored else None,
              "wrong_share": round(tot["wrong"] / scored, 4) if scored else None,
              "links": int(lk["n"]), "links_right": int(lk["right"]), "links_wrong": int(lk["wrong"]), "links_unknown": int(lk["unknown"]),
              "link_precision": round(lk["right"] / judged, 4) if judged else None,
              "impure_row_share": round(imp / rows, 5) if rows else None}  # fmt: skip
    return {"pooled": pooled, "per_clip": {k: {kk: v[kk] for kk in ("returns", "links", "impure_rows")} for k, v in per.items()},
            "clips_without_entry": [s for s in clips if s not in method]}  # fmt: skip


def _decide(pairs, t_abs: float, margin: float):
    later, earlier = {}, {}
    for a, b, s in pairs:
        later.setdefault(a, []).append((s, b))
        earlier.setdefault(b, []).append((s, a))
    out = []
    for a, b, s in pairs:
        ob, oa = sorted(later[a], reverse=True), sorted(earlier[b], reverse=True)
        if ob[0][1] != b or oa[0][1] != a or s < t_abs or len(ob) < 2 or len(oa) < 2:
            continue
        if min(s - ob[1][0], s - oa[1][0]) >= margin:
            out.append((a, b))
    return out


def baseline_links(name: str, clips=data.DEV_CLIPS) -> dict:
    """"split" (gbf now: no links), "plugin" (re-join every piece of a plugin track),
    "mutual" (appearance mutual best with a 0.05 margin, no absolute bar)."""
    out = {}
    for seq in clips:
        if name == "split":
            out[seq] = {"links": []}
        elif name == "plugin":
            p = data.pieces(seq)
            links = []
            for _, g in p.groupby("plugin"):
                ids = g.sort_values("first")["piece"].tolist()
                links += [[ids[i], ids[i + 1]] for i in range(len(ids) - 1)]
            out[seq] = {"links": links}
        elif name == "mutual":
            cand = [(int(a), int(b), float(s)) for a, b, s in data.appearance(seq).get("candidate_pairs", [])]
            out[seq] = {"links": [list(x) for x in _decide(cand, -1.0, 0.05)]}
        else:
            raise ValueError(name)
    return out


def main() -> int:
    method = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    res = {"name": method.get("name"), **evaluate(method["clips"])}
    text = json.dumps(res, indent=1)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"name": res["name"], "pooled": res["pooled"], "clips_without_entry": res["clips_without_entry"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
