"""v_leak / blind.py -- a "blindfold" for one leave-one-clip-out fold.

install(fold) makes every attempt of THIS process to read the annotation of `fold` fail:
  - builtins.open / io.open (pathlib uses io.open) of a path that names the clip and is an annotation
    file (Labels-GameState.json, anything under .../external/...) or its reentry-eval.json (label-derived);
  - reid_lab.score.clip(fold) and gbf.evaluate.gsr.load_reference(<path naming fold>).
install("SNGS-") blocks the labels of every clip (for stages that must be label-free altogether).
Every path opened through open / io.open / pandas.read_parquet / pandas.read_pickle / numpy.load is
logged in OPENED so that the stage can be audited afterwards (which label-derived files did it read?).
"""
from __future__ import annotations

import builtins
import io
import os
import sys

sys.dont_write_bytecode = True      # never leave .pyc files in other agents' folders
OPENED: list[str] = []
BLOCKED: list[str] = []


class LabelLeak(RuntimeError):
    pass


def _norm(p) -> str:
    try:
        p = os.fspath(p)
    except TypeError:
        return ""
    if isinstance(p, bytes):
        p = p.decode(errors="replace")
    return str(p).replace("\\", "/")


def install(fold: str) -> None:
    def bad(p: str) -> bool:
        return fold in p and ("Labels-GameState" in p or "/external/" in p or "reentry-eval" in p)

    real_open = builtins.open

    def guarded_open(file, *a, **k):
        p = _norm(file) if not isinstance(file, int) else ""
        if p:
            OPENED.append(p)
            if bad(p):
                BLOCKED.append(p)
                raise LabelLeak(f"blindfold: attempt to open {p}")
        return real_open(file, *a, **k)

    builtins.open = guarded_open
    io.open = guarded_open

    import numpy as np
    import pandas as pd

    def wrap(mod, name):
        real = getattr(mod, name)

        def f(path, *a, **k):
            p = _norm(path)
            if p:
                OPENED.append(p)
                if bad(p):
                    BLOCKED.append(p)
                    raise LabelLeak(f"blindfold: attempt to read {p}")
            return real(path, *a, **k)

        setattr(mod, name, f)

    wrap(pd, "read_parquet")
    wrap(pd, "read_pickle")
    wrap(pd, "read_json")
    wrap(pd, "read_csv")
    wrap(np, "load")

    sys.path.insert(0, "C:/Research/plugins/tools")
    from reid_lab import score
    from gbf.evaluate import gsr

    real_clip = score.clip

    def clip(seq):
        if fold in seq:
            BLOCKED.append(f"score.clip({seq})")
            raise LabelLeak(f"blindfold: score.clip({seq})")
        return real_clip(seq)

    score.clip = clip
    real_load = gsr.load_reference

    def load_reference(path, *a, **k):
        if bad(_norm(path)):
            BLOCKED.append(_norm(path))
            raise LabelLeak(f"blindfold: gsr.load_reference({path})")
        return real_load(path, *a, **k)

    gsr.load_reference = load_reference


def selftest(fold: str) -> None:
    """The blindfold must actually bite: every door to the clip's labels raises."""
    from reid_lab import score

    n = 0
    for door in (lambda: score.clip(fold), lambda: score.score_clip(fold, []),
                 lambda: open(score.TRAIN / fold / "Labels-GameState.json"),
                 lambda: (score.TRAIN / fold / "Labels-GameState.json").read_text(encoding="utf-8")):
        try:
            door()
        except LabelLeak:
            n += 1
    if n != 4:
        raise SystemExit(f"blindfold self-test failed: only {n} of 4 doors closed")
    del BLOCKED[:]
    del OPENED[:]


def audit(paths=None) -> dict:
    """Opened files by kind. 'derived' = files under reid_lab/work that are not source code (fitted
    weights, caches, METHOD files of any agent) -- the ones that could carry labels."""
    paths = OPENED if paths is None else paths
    uniq = sorted(set(paths))
    work = [p for p in uniq if "/reid_lab/work/" in p]
    return {
        "n_opened": len(uniq),
        "label_files": [p for p in uniq if "Labels-GameState" in p or "/external/" in p],
        "reentry_eval": [p for p in uniq if "reentry-eval" in p],
        "work_code": [p for p in work if p.endswith((".py", ".pyc"))],
        "work_derived": [p for p in work if not p.endswith((".py", ".pyc"))],
        "blocked": list(BLOCKED),
    }
