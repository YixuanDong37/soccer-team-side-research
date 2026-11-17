"""v_leak round 1 (no labels): two folds were run twice in separate processes; are the results identical?

  SNGS-162  once with its own per-fold table cache (the first test run, out/test_per_fold_cache/SNGS-162.pkl)
            and once from the shared tables of s2_tables.py (out/blind/SNGS-162.pkl)
  SNGS-112  run by a second driver call while the first was busy (out/test_per_fold_cache/SNGS-112_first_run.pkl,
            the one s4_compare.py and n1_nested.py used) and again at the end of the first driver call (out/blind/SNGS-112.pkl)

    python -I -B s5_repeat_check.py   -> out/repeat_check.txt
"""
import pickle
from pathlib import Path

HERE = Path("C:/Research/plugins/tools/reid_lab/work/v_leak_1")
lines = []
for fold, first in (("SNGS-162", "out/test_per_fold_cache/SNGS-162.pkl"), ("SNGS-112", "out/test_per_fold_cache/SNGS-112_first_run.pkl")):
    a = pickle.load(open(HERE / first, "rb"))
    b = pickle.load(open(HERE / f"out/blind/{fold}.pkl", "rb"))
    same = {k: a[k] == b[k] for k in ("models", "cuts", "held", "cells", "chosen")}
    lines.append(f"{fold}: two runs identical in {same}; held-out label files opened: "
                 f"{sum(fold in p for p in a['audit']['label_files'] + a['audit']['reentry_eval'])} / {sum(fold in p for p in b['audit']['label_files'] + b['audit']['reentry_eval'])}")
(HERE / "out/repeat_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
