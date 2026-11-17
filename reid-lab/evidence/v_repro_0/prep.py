"""fuse / prep.py -- LABEL-FREE evidence tables for the fusion step, cached on disk.

For a clip and a cut list (m_cuts) it builds, on the sub-pieces that the cuts produce:
  app    m_appear's pair table (features.pair_features(seq, cuts)): cos_raw / cos_tc / cos_w ...
  space  m_space's pair table (features.pair_features) computed on the cut pieces. m_space has
         no `cuts` argument, so its modules are handed a tracks table whose track_id column holds
         the sub-piece ids (reid_lab.data.tracks is substituted in this process only while the
         table is built; no file of the lab or of another agent is changed).
  ends   m_space's geom.piece_ends on the cut pieces (first / last frame, border sides ...).

Nothing here reads an annotation or reid_lab.score.

    python -I prep.py            # builds every table the LOCO run and the frozen method need
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True   # importing the other agents' modules must not leave .pyc files in their folders
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
WORK = "C:/Research/plugins/tools/reid_lab/work/"
SUB = 10**7
P_CUT = 0.7          # m_cuts' default probability level (not tuned here)


def _import(agent: str, name: str):
    """Import module `name` of another agent under a private name (their modules are all
    called features.py, so a plain import would collide)."""
    import importlib.util

    key = f"{agent}__{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = WORK + agent
    # the agent's modules import their siblings by bare name: make those resolve to the same agent
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k in _SIBLINGS}
    sys.path.insert(0, path)
    try:
        spec = importlib.util.spec_from_file_location(key, f"{path}/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
        # keep the siblings it pulled in under private names too
        for k in list(sys.modules):
            if k in _SIBLINGS:
                sys.modules[f"{agent}__{k}"] = sys.modules.pop(k)
    finally:
        # the agents' modules put their own folder at the front of sys.path; take it out again so
        # that a later bare import here (method, linker, ...) can never resolve to their files
        norm = lambda q: str(q).replace("\\", "/").rstrip("/").lower()  # noqa: E731
        sys.path[:] = [q for q in sys.path if norm(q) != norm(path)]
        sys.modules.update(saved)
    return mod


_SIBLINGS = {"features", "core", "agglo", "decide", "method", "geom", "linker", "posscore", "fitdata", "labels",
             "relink", "cutlinks", "truth"}


def norm_cuts(cuts) -> tuple:
    return tuple(sorted({(int(p), int(f)) for p, f in (cuts or ())}))


def cut_key(cuts) -> str:
    c = norm_cuts(cuts)
    return "nocut" if not c else hashlib.sha1(json.dumps(c).encode()).hexdigest()[:12]


def ids_after_cuts(tid: np.ndarray, frame: np.ndarray, cuts) -> np.ndarray:
    """reid_lab.score's sub-piece ids: rows of `piece` from the j-th cut frame on get piece + SUB * (j + 1)."""
    out = tid.copy()
    by: dict[int, list[int]] = {}
    for p, f in norm_cuts(cuts):
        by.setdefault(p, []).append(f)
    for p, fs in by.items():
        for j, f in enumerate(sorted(fs)):
            out[(tid == p) & (frame >= f)] = p + SUB * (j + 1)
    return out


# ----------------------------------------------------------------------------- m_cuts
_CUTS: dict = {}


def cuts_for(seq: str, fold: str | None, p: float = P_CUT) -> tuple:
    """Cuts of a clip. fold = a dev clip: m_cuts' weights fitted WITHOUT that clip (so nothing in
    a LOCO fold has seen the held-out clip); fold = None: m_cuts' frozen all-13 weights.
    p: m_cuts' probability level (the method uses P_CUT; other values only in sensitivity.py).
    The lists are kept in cache/cuts.json (written by `python -I prep.py`); delete it to recompute."""
    key = f"{seq}|{fold}" if p == P_CUT else f"{seq}|{fold}|{p}"
    if not _CUTS and (CACHE / "cuts.json").exists():
        _CUTS.update(json.loads((CACHE / "cuts.json").read_text(encoding="utf-8")))
    if key not in _CUTS:
        cf = _import("m_cuts", "features")
        par = p if fold is None else cf.heldout_params(fold, p)
        _CUTS[key] = [list(x) for x in norm_cuts(cf.cuts(seq, par))]
    return norm_cuts(_CUTS[key])


def save_cuts() -> None:
    (CACHE / "cuts.json").write_text(json.dumps(_CUTS), encoding="utf-8")


# ----------------------------------------------------------------------------- m_appear
def app_table(seq: str, cuts) -> pd.DataFrame:
    path = CACHE / f"app_{seq}_{cut_key(cuts)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    af = _import("m_appear", "features")
    df = af.pair_features(seq, cuts=norm_cuts(cuts)).copy()
    df.to_pickle(path)
    return df


# ----------------------------------------------------------------------------- m_space
def _space_modules():
    sf = _import("m_space", "features")
    geom = sys.modules["m_space__geom"]
    return sf, geom


def _clear_space_caches():
    sf, geom = _space_modules()
    for mod in (sf, geom):
        for v in vars(mod).values():
            if hasattr(v, "cache_clear"):
                v.cache_clear()


def space_tables(seq: str, cuts):
    """(pair table, piece ends) of m_space on the cut pieces."""
    p1 = CACHE / f"space_{seq}_{cut_key(cuts)}.pkl"
    p2 = CACHE / f"ends_{seq}_{cut_key(cuts)}.pkl"
    if p1.exists() and p2.exists():
        return pd.read_pickle(p1), pd.read_pickle(p2)
    sf, geom = _space_modules()
    t = data.tracks(seq)
    cuts = norm_cuts(cuts)
    t2 = t.assign(track_id=ids_after_cuts(t.track_id.to_numpy(), t.frame.to_numpy(), cuts)) if cuts else t
    orig = data.tracks
    _clear_space_caches()
    data.tracks = lambda s: t2 if s == seq else orig(s)      # this process only, restored below
    try:
        df = sf.pair_features(seq)
        ends = geom.piece_ends(seq).copy()
    finally:
        data.tracks = orig
        _clear_space_caches()
    df.to_pickle(p1)
    ends.to_pickle(p2)
    return df, ends


def build(job):
    seq, cuts = job
    a = app_table(seq, cuts)
    s, e = space_tables(seq, cuts)
    return seq, cut_key(cuts), len(cuts), len(a), len(s), len(e)


def all_jobs():
    jobs = {}
    for seq in data.DEV_CLIPS:
        jobs[(seq, ())] = None                      # the uncut pieces (ablation without m_cuts)
        for fold in (None, *data.DEV_CLIPS):
            c = cuts_for(seq, fold)
            jobs[(seq, c)] = None
    return list(jobs)


if __name__ == "__main__":
    import time
    from concurrent.futures import ProcessPoolExecutor

    CACHE.mkdir(exist_ok=True)
    t0 = time.time()
    jobs = all_jobs()
    index = {}
    for seq in data.DEV_CLIPS:
        index[seq] = {str(fold): cut_key(cuts_for(seq, fold)) for fold in (None, *data.DEV_CLIPS)}
    (CACHE / "cut_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    (CACHE / "cuts.json").write_text(json.dumps(_CUTS), encoding="utf-8")
    print(len(jobs), "distinct (clip, cut list) tables;", round(time.time() - t0, 1), "s for the cuts", flush=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(build, jobs):
            print(*r, round(time.time() - t0, 1), flush=True)
    print("done", round(time.time() - t0, 1), "s")
