"""fuse_r1 / evidence.py (fuse's evidence.py + the n_solid argument) -- LABEL-FREE: one table per clip with every agent's evidence for a pair of
(sub-)pieces (a earlier, b later), on the pieces as they are after m_cuts' cuts.

    pair_table(seq, cuts, pos_params) -> DataFrame, one row per candidate pair of m_appear
        (neither a referee, a ends before b starts, both have a prototype, same appearance team):
      a, b, gap                       sub-piece ids (reid_lab.score's convention), frames between
      cos_w, cos_tc, cos_raw          m_appear's cosines (whitened / team-centred / raw prototypes)
      low                             1 when either prototype rests on < 5 clean rows
      plug                            1 when the plugin tracked a and b as ONE id, b is the next piece
                                      of that id after a, and gbf (not m_cuts) separated them: the
                                      plugin's own re-entry join (m_keep's subject)
      pos_known                       both have a ground position (m_space could measure the pair)
      pos_adm                         m_space admits the pair (within reach at 8 m/s + 3 m, same role,
                                      team labels not different)
      pos_unreach                     pos_known and not within reach
      pos_z, pos_sink, pos_p_in, pos_p_out   m_space's position model on the cut pieces: log evidence,
                                      doubly normalised assignment score, P(a | b), P(b | a) (0 when
                                      not admissible)
      kind                            0 / 1 / 2: how many of (a ends, b starts) are at an image border
      rows_a, rows_b, clean_a, clean_b, first_a, last_a, first_b, last_b
    pieces(seq, cuts) -> DataFrame per sub-piece (m_space's geom.piece_ends columns)

n_solid > 0 (fuse_r1): the same table for the SOLID pieces only (>= n_solid boxes); m_space's position
model is then run on the solid pieces alone, so a 1-4 box fragment takes no part in the position
competition, and "plug" means consecutive SOLID pieces of one plugin id.

Nothing here reads an annotation or reid_lab.score.
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse_r1")
import prep  # noqa: E402

SUB = prep.SUB


def pieces(seq: str, cuts, n_solid: int = 0) -> pd.DataFrame:
    return prep.space_tables(seq, cuts, n_solid)[1]


def position_scores(seq: str, cuts, pos_params: dict, n_solid: int = 0) -> pd.DataFrame:
    """m_space's posscore.position_scores on the cut pieces (same columns), for given weights."""
    prep._space_modules()
    linker = prep._import("m_space", "linker")
    p, _ = prep.space_tables(seq, cuts, n_solid)
    q = linker.posteriors(p, pos_params, "sinkhorn")
    q["pos_kind"] = linker.kind(q)
    q = q.rename(columns={"z": "pos_z", "p_in": "pos_p_in", "p_out": "pos_p_out", "score": "pos_sink"})
    return q[["a", "b", "pos_z", "pos_p_in", "pos_p_out", "pos_sink", "pos_kind"]]


def plugin_joins(seq: str, cuts, n_solid: int = 0) -> set:
    """{(a, b)}: consecutive (sub-)pieces of one plugin id that gbf's re-entry split separated
    (the two stretches next to a re-entry gap of the plugin track)."""
    e = pieces(seq, cuts, n_solid).sort_values(["plugin", "first"])
    out = set()
    for _, g in e.groupby("plugin", sort=False):
        ids, last, first = g.piece.to_numpy(), g["last"].to_numpy(), g["first"].to_numpy()
        for i in range(len(ids) - 1):
            if ids[i] % SUB != ids[i + 1] % SUB and last[i] < first[i + 1]:
                out.add((int(ids[i]), int(ids[i + 1])))
    return out


def pair_table(seq: str, cuts, pos_params: dict, n_solid: int = 0) -> pd.DataFrame:
    a = prep.app_table(seq, cuts)
    if n_solid:
        solid = prep.solid_ids(seq, cuts, n_solid)
        a = a[a.a.isin(solid) & a.b.isin(solid)]
    s, e = prep.space_tables(seq, cuts, n_solid)
    keep = ["a", "b", "gap", "cos_w", "cos_tc", "cos_raw", "low_a", "low_b", "rows_a", "rows_b", "clean_a", "clean_b",
            "medh_a", "medh_b", "same_plugin"]
    df = a[keep].copy()
    df["low"] = (df.low_a | df.low_b).astype(int)
    sp = s[["a", "b", "reach", "r", "a_end_edge", "b_start_edge", "dt", "d"]]
    df = df.merge(sp, on=["a", "b"], how="left")
    df["pos_known"] = df.r.notna().astype(int)
    df["pos_unreach"] = ((df.pos_known == 1) & ~df.reach.fillna(False).astype(bool)).astype(int)
    df["kind"] = df.a_end_edge.fillna(False).astype(int) + df.b_start_edge.fillna(False).astype(int)
    q = position_scores(seq, cuts, pos_params, n_solid)
    df = df.merge(q.drop(columns=["pos_kind"]), on=["a", "b"], how="left")
    df["pos_adm"] = df.pos_z.notna().astype(int)
    for c in ("pos_p_in", "pos_p_out", "pos_sink"):
        df[c] = df[c].fillna(0.0)
    pj = plugin_joins(seq, cuts, n_solid)
    df["plug"] = [int((int(x), int(y)) in pj) for x, y in zip(df.a, df.b)]
    ee = e.set_index("piece")
    for side in ("a", "b"):
        df[f"first_{side}"] = ee["first"].reindex(df[side]).to_numpy()
        df[f"last_{side}"] = ee["last"].reindex(df[side]).to_numpy()
    return df
