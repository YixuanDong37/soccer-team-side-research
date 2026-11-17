"""critic / common.py -- DIAGNOSIS helpers. Reads the annotation (rule 5c): nothing in this
directory is a method; every number produced here is an oracle / diagnosis number.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
WORK = Path("C:/Research/plugins/tools/reid_lab/work")
FUSE = WORK / "fuse_r1"
TRAIN = score.TRAIN
GAME = {}  # seq -> game id


@lru_cache(maxsize=None)
def labels(seq: str) -> dict:
    """{ref_id: {"role", "team", "jersey"}} of the annotation, plus the match id in GAME."""
    d = json.loads((TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    GAME[seq] = d["info"].get("game_id")
    out = {}
    for a in d["annotations"]:
        at = a.get("attributes") or {}
        if at.get("role") in ("player", "goalkeeper", "referee"):
            j = at.get("jersey")
            out.setdefault(int(a["track_id"]), {"role": at.get("role"), "team": at.get("team"),
                                                "jersey": None if j in (None, "") else str(j)})
    return out


def game(seq: str) -> str:
    labels(seq)
    return GAME[seq]


def ids_after_cuts(seq: str, cuts) -> np.ndarray:
    """Sub-piece id per matched pair (score.py's own function; diagnosis use only)."""
    return score._ids_after_cuts(score.clip(seq), cuts)


def piece_people(seq: str, cuts=()) -> dict:
    """{(sub)piece id: Counter(ref_id -> matched rows)}."""
    c = score.clip(seq)
    ids = ids_after_cuts(seq, cuts)
    out: dict[int, Counter] = {}
    for i, r in zip(ids.tolist(), c.refs.tolist()):
        if i >= 0:
            out.setdefault(i, Counter())[r] += 1
    return out


def piece_majority(seq: str, cuts=(), min_rows: int = 5) -> dict:
    """{piece: ref_id} for pieces whose majority person has >= min_rows matched rows."""
    out = {}
    for p, cnt in piece_people(seq, cuts).items():
        top, n = cnt.most_common(1)[0]
        if n >= min_rows:
            out[p] = top
    return out


def piece_span(seq: str, cuts=()) -> dict:
    """{(sub)piece id: (first frame, last frame, rows)} from the tracks table (label-free)."""
    t = data.tracks(seq)
    tid, fr = t.track_id.to_numpy(), t.frame.to_numpy()
    ids = tid.copy()
    by = {}
    for p, f in cuts or ():
        by.setdefault(int(p), []).append(int(f))
    for p, fs in by.items():
        for j, f in enumerate(sorted(fs)):
            ids[(tid == p) & (fr >= f)] = p + score.SUB * (j + 1)
    out = {}
    for i in np.unique(ids[ids >= 0]):
        m = ids == i
        out[int(i)] = (int(fr[m].min()), int(fr[m].max()), int(m.sum()))
    return out, ids


def method_file(name: str) -> dict:
    return json.loads((FUSE / f"METHOD_{name}.json").read_text(encoding="utf-8"))["clips"]


def pooled(method: dict) -> dict:
    return score.evaluate(method)["pooled"]


def fmt(p: dict) -> str:
    return (f"{p['right']:3d} / {p['wrong']:2d} / {p['lost']:3d} | links {p['links_right']:3d} / {p['links_wrong']:2d} / "
            f"{p['links_unknown']:3d} | impure {100 * p['impure_row_share']:.2f}%")


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
