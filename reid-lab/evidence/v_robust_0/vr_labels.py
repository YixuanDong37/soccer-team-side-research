"""v_robust_0 / vr_labels.py -- LABELS. Diagnosis and scoring helpers only; never imported by code
that produces links or cuts. Identities come from reid_lab.score's own matching, so they agree
with the scorer; the annotated team / role / jersey of a person are read from the annotation file
(the lab's loader does not carry them) for the cross-team / cross-role diagnosis only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from functools import lru_cache

sys.dont_write_bytecode = True
import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402


@lru_cache(maxsize=None)
def ref_attrs(seq: str) -> dict:
    """annotated person -> {"role", "team", "jersey"} (most frequent value over the person's boxes)."""
    doc = json.loads((score.TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    acc: dict[int, dict[str, Counter]] = {}
    for row in doc.get("annotations") or []:
        if row.get("supercategory") != "object":
            continue
        at = row.get("attributes") or {}
        d = acc.setdefault(int(row["track_id"]), {"role": Counter(), "team": Counter(), "jersey": Counter()})
        for k in d:
            d[k][str(at.get(k))] += 1
    return {rid: {k: c.most_common(1)[0][0] for k, c in d.items()} for rid, d in acc.items()}


def assign(seq: str, links, cuts):
    """(clip, sub-piece id per matched pair, group per matched pair) exactly as score.score_clip."""
    c = score.clip(seq)
    ids = score._ids_after_cuts(c, [list(x) for x in cuts])
    groups = score._Groups()
    for a, b in links or []:
        groups.union(int(a), int(b))
    tracked = ids >= 0
    alone = -1 - np.arange(len(ids), dtype=np.int64)
    g = np.where(tracked, np.array([groups.find(int(x)) if x >= 0 else -1 for x in ids], dtype=np.int64), alone)
    return c, ids, g


def identities(seq: str, cuts) -> dict:
    """sub-piece id -> (majority annotated person, its matched rows, all matched rows of the piece)."""
    c = score.clip(seq)
    ids = score._ids_after_cuts(c, [list(x) for x in cuts])
    out = {}
    for tid in np.unique(ids[ids >= 0]):
        cnt = Counter(c.refs[ids == tid].tolist())
        top, n = cnt.most_common(1)[0]
        out[int(tid)] = (int(top), int(n), int(sum(cnt.values())))
    return out


def returns_table(seq: str, links, cuts) -> list[dict]:
    """One dict per annotated return: outcome (score.py's rule, re-derived here and checked against
    score.score_clip by the callers), gap, the pieces of rows A and B, B's group."""
    c, ids, g = assign(seq, links, cuts)
    tracked = ids >= 0
    out = []
    for r in c.returns:
        a, b = r["a"], r["b"]
        d = {"seq": seq, "ref_id": r["ref_id"], "last": r["last"], "first": r["first"], "gap": r["gap"]}
        if a is None or b is None:
            d["outcome"] = "undetected"
        elif not (tracked[a] and tracked[b]):
            d["outcome"] = "untracked"
        else:
            d.update(piece_a=int(ids[a]), piece_b=int(ids[b]), frame_a=int(c.frames[a]), frame_b=int(c.frames[b]))
            if g[a] == g[b]:
                d["outcome"] = "right"
            else:
                counts = Counter(c.refs[(c.frames < c.frames[b]) & (g == g[b])].tolist())
                if not counts:
                    d["outcome"] = "lost"
                else:
                    top, n = counts.most_common(1)[0]
                    d["outcome"] = "right" if top == r["ref_id"] else ("wrong" if n >= score.MIN_MATCHED else "lost")
                    d["b_group_before"] = dict(counts.most_common(4))
        out.append(d)
    return out
