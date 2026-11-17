"""LABEL ACCESS (not method code). Piece identities from the annotation, through score.clip.

Used ONLY for (b) fitting thresholds/weights inside leave-one-clip-out cross-validation and
(c) diagnosis / oracle analyses. Method code (features.py, method.py link builders) never imports
this module.
"""
import sys
from collections import Counter
from functools import lru_cache

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402

MIN_MATCHED = 5


@lru_cache(maxsize=None)
def piece_identity(seq: str) -> dict:
    """piece id -> (majority annotated person, matched rows of that person, matched rows) for pieces
    with >= 5 matched rows of the majority person (score.py's rule for a known identity)."""
    c = score.clip(seq)
    out = {}
    for pid in set(c.piece[c.piece >= 0].tolist()):
        cnt = Counter(c.refs[c.piece == pid].tolist())
        top, n = cnt.most_common(1)[0]
        if n >= MIN_MATCHED:
            out[int(pid)] = (int(top), int(n), int(sum(cnt.values())))
    return out


def pair_label(seq: str, a: int, b: int):
    """1 same person, 0 different people, None unknown (a piece without a known identity)."""
    ident = piece_identity(seq)
    if a not in ident or b not in ident:
        return None
    return int(ident[a][0] == ident[b][0])
