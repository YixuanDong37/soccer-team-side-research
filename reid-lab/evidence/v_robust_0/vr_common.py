"""v_robust_0 / vr_common.py -- shared helpers of the robustness check of work/fuse.

Imports fuse's modules READ-ONLY: no bytecode is written, and the two cached table builders of
fuse/prep.py are replaced in this process by versions that read fuse/cache when the table exists
there and otherwise build the table into THIS directory's cache (never into fuse's).
Nothing here reads an annotation; scripts that do say so in their first line.

SPACE_PATCH: {"MAX_SPEED": 8.8, ...} temporarily changes hand-set constants of m_space/features.py
while a position table is built (sensitivity runs only); such tables get their own cache names.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data  # noqa: E402
import prep  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MYCACHE = HERE / "cache"
FUSE = Path("C:/Research/plugins/tools/reid_lab/work/fuse")
FUSE_CACHE = FUSE / "cache"
SUB = 10**7
POINTS = ("strict", "balanced", "recall")
OUT.mkdir(exist_ok=True)
MYCACHE.mkdir(exist_ok=True)
SPACE_PATCH: dict = {}


def _find(name: str, fuse_ok: bool = True):
    for d in ((FUSE_CACHE, MYCACHE) if fuse_ok else (MYCACHE,)):
        if (d / name).exists():
            return d / name
    return None


def app_table(seq: str, cuts) -> pd.DataFrame:
    name = f"app_{seq}_{prep.cut_key(cuts)}.pkl"
    p = _find(name)
    if p is not None:
        return pd.read_pickle(p)
    af = prep._import("m_appear", "features")
    df = af.pair_features(seq, cuts=prep.norm_cuts(cuts)).copy()
    df.to_pickle(MYCACHE / name)
    return df


def space_tables(seq: str, cuts):
    tag = "".join(f"_{k}{v:g}" for k, v in sorted(SPACE_PATCH.items()))
    n1, n2 = f"space_{seq}_{prep.cut_key(cuts)}{tag}.pkl", f"ends_{seq}_{prep.cut_key(cuts)}{tag}.pkl"
    p1, p2 = _find(n1, not tag), _find(n2, not tag)
    if p1 is not None and p2 is not None:
        return pd.read_pickle(p1), pd.read_pickle(p2)
    sf, geom = prep._space_modules()
    t = data.tracks(seq)
    cuts = prep.norm_cuts(cuts)
    t2 = t.assign(track_id=prep.ids_after_cuts(t.track_id.to_numpy(), t.frame.to_numpy(), cuts)) if cuts else t
    orig = data.tracks
    saved = {k: getattr(sf, k) for k in SPACE_PATCH}
    prep._clear_space_caches()
    data.tracks = lambda s: t2 if s == seq else orig(s)
    for k, v in SPACE_PATCH.items():
        setattr(sf, k, v)
    try:
        df = sf.pair_features(seq)
        ends = geom.piece_ends(seq).copy()
    finally:
        data.tracks = orig
        for k, v in saved.items():
            setattr(sf, k, v)
        prep._clear_space_caches()
    df.to_pickle(MYCACHE / n1)
    ends.to_pickle(MYCACHE / n2)
    return df, ends


def _no_save():
    raise RuntimeError("v_robust must not write fuse/cache/cuts.json")


prep.app_table = app_table
prep.space_tables = space_tables
prep.save_cuts = _no_save

import evidence  # noqa: E402
import linker  # noqa: E402
import method  # noqa: E402
import pairmodel  # noqa: E402


def load_method(name: str) -> dict:
    """clips dict of fuse/METHOD_<name>.json."""
    return json.loads((FUSE / f"METHOD_{name}.json").read_text(encoding="utf-8"))["clips"]


def params() -> dict:
    return json.loads((FUSE / "params_loco.json").read_text(encoding="utf-8"))


def piece_ids_after_cuts(seq: str, cuts) -> np.ndarray:
    """Sub-piece id per tracks row (score.py's convention), label-free."""
    t = data.tracks(seq)
    return prep.ids_after_cuts(t.track_id.to_numpy(), t.frame.to_numpy(), cuts)


def groups_of(links) -> dict:
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        parent[find(int(a))] = find(int(b))
    out: dict[int, list[int]] = {}
    for x in list(parent):
        out.setdefault(find(x), []).append(x)
    return out


def counts(per_return) -> dict:
    c = Counter(per_return)
    return {k: int(c.get(k, 0)) for k in ("right", "wrong", "lost")}


def pool(items) -> dict:
    """items: per-clip dicts {"per_return", "links" or "lk", "impure_rows" or "impure", "tracked_rows" or "rows"}."""
    c, lk, imp, rows = Counter(), Counter(), 0, 0
    for x in items:
        c.update(x["per_return"])
        d = x["lk"] if "lk" in x else x["links"]      # fuse's raw results keep the link LIST under "links"
        lk.update({k: v for k, v in d.items() if k != "n"})
        imp += x.get("impure_rows", x.get("impure"))
        rows += x.get("tracked_rows", x.get("rows"))
    return {"right": c["right"], "wrong": c["wrong"], "lost": c["lost"], "links_right": lk["right"], "links_wrong": lk["wrong"],
            "links_unknown": lk["unknown"], "impure_row_share": round(imp / rows, 5) if rows else None}


def fmt_pool(p: dict) -> str:
    return (f"{p['right']} / {p['wrong']} / {p['lost']} | {p['links_right']} / {p['links_wrong']} / {p['links_unknown']} | "
            f"{100 * p['impure_row_share']:.2f}%")
