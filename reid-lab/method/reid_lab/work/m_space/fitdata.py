"""LABELS (fitting / diagnosis only): pair features joined with whether the two pieces are the
same annotated person. Used only on the fit clips of a leave-one-clip-out fold and in oracle
scripts; never to produce the links of a held-out clip."""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
import features  # noqa: E402
import labels  # noqa: E402


@lru_cache(maxsize=None)
def labelled_pairs(seq: str) -> pd.DataFrame:
    """features.pair_features(seq) plus: known (both pieces have an identity by score.py's rule,
    >= 5 matched rows of the majority person), same (known and equal), imm (same, and a is the
    latest earlier piece of that person before b), a_ref / b_ref."""
    p = features.pair_features(seq).copy()
    ident = labels.identity_map(seq, 5)
    p["a_ref"] = p.a.map(ident)
    p["b_ref"] = p.b.map(ident)
    p["known"] = p.a_ref.notna() & p.b_ref.notna()
    p["same"] = p.known & (p.a_ref == p.b_ref)
    latest = p[p.same].groupby("b").a_last.transform("max")
    p["imm"] = False
    p.loc[latest.index, "imm"] = p.loc[latest.index, "a_last"] == latest
    p["seq"] = seq
    return p


def pooled(clips) -> pd.DataFrame:
    return pd.concat([labelled_pairs(s) for s in clips], ignore_index=True)
