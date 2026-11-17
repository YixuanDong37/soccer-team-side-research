"""fuse / fuse_labels.py -- LABELS. Used only (b) to fit weights and choose thresholds on the FIT
clips of a leave-one-clip-out fold and (c) in diagnosis scripts. Never imported by the code that
produces links (prep.py, evidence.py, linker.py, method.py's apply path).

Identities come from reid_lab.score's own matching (score.clip) so that they agree with the scorer.
"""
from __future__ import annotations

import sys
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import score  # noqa: E402


@lru_cache(maxsize=None)
def identities(seq: str, cuts: tuple) -> dict:
    """sub-piece id -> (majority annotated person, its matched rows, all matched rows)."""
    c = score.clip(seq)
    ids = score._ids_after_cuts(c, [list(x) for x in cuts])
    out = {}
    for tid in np.unique(ids[ids >= 0]):
        cnt = Counter(c.refs[ids == tid].tolist())
        top, n = cnt.most_common(1)[0]
        out[int(tid)] = (int(top), int(n), int(sum(cnt.values())))
    return out


def label_pairs(df: pd.DataFrame, seq: str, cuts: tuple, min_rows: int = 5) -> pd.DataFrame:
    """Adds a_ref, b_ref, known (both sub-pieces have >= min_rows matched rows of their majority
    person: score.py's rule for 5), same, imm (same and a is the latest earlier piece of that person
    among the rows of df for b)."""
    ident = identities(seq, cuts)
    ref = {k: v[0] for k, v in ident.items() if v[1] >= min_rows}
    df = df.copy()
    df["a_ref"] = df.a.map(ref)
    df["b_ref"] = df.b.map(ref)
    df["known"] = df.a_ref.notna() & df.b_ref.notna()
    df["same"] = df.known & (df.a_ref == df.b_ref)
    latest = df[df.same].groupby("b").last_a.transform("max")
    df["imm"] = False
    df.loc[latest.index, "imm"] = df.loc[latest.index, "last_a"] == latest
    df["seq"] = seq
    return df
