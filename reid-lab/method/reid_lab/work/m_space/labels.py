"""LABELS. Not a method module: never import this from features.py or from code that produces
links for a held-out clip.

Used only (b) to fit thresholds / weights inside the leave-one-clip-out loop (fit clips only)
and (c) in oracle / diagnosis scripts. It reads the annotation through reid_lab.score.
"""
from __future__ import annotations

import sys
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402


@lru_cache(maxsize=None)
def piece_identity(seq: str) -> pd.DataFrame:
    """Per piece: the annotated person most of its matched rows belong to (ref), how many
    matched rows (n), how many of them are that person (top), and the first / last matched
    frame of that person in the piece."""
    c = score.clip(seq)
    rows = []
    for pid in np.unique(c.piece[c.piece >= 0]):
        m = c.piece == pid
        cnt = Counter(c.refs[m].tolist())
        top, n = cnt.most_common(1)[0]
        rows.append({"piece": int(pid), "ref": int(top), "top": int(n), "n": int(m.sum())})
    return pd.DataFrame(rows)


def identity_map(seq: str, min_rows: int = 5, min_share: float = 0.0) -> dict[int, int]:
    """piece -> annotated person, for pieces with at least ``min_rows`` matched rows of their
    majority person (score.py's rule is min_rows = 5 on the majority count)."""
    d = piece_identity(seq)
    d = d[(d.top >= min_rows) & (d.top >= min_share * d.n)]
    return dict(zip(d.piece.astype(int), d.ref.astype(int)))
