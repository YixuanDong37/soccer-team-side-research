"""v_leak round 1, stage 0 (no labels): static facts about fuse_r1's apply path and the fitted files it reads.

1. Import graph of the apply path (ast): which modules each file imports; flags any import of
   reid_lab.score, gbf.evaluate, fuse_labels, or a labels module of another agent, and any string in the
   file that names an annotation.
2. The fitted files fuse_r1 reads for a fold (m_cuts/loco_params.json[fold], m_space/params_cache/
   excl_<fold>.json) against the blindfolded re-fits of round 0 (v_leak_0/refit/, fitted in a process
   that could not open the held-out clip's annotation): equal now?  And their modification times
   against the time of that re-fit (unchanged since?).
3. fuse_r1/cache/cuts.json (the cut lists its LOCO run used) against the blind re-fit's cuts.
4. Time line of fuse_r1's files (PREREG.txt against the first results).

    python -I -B s0_static.py   -> out/static.md
"""
import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data  # noqa: E402

WORK = Path("C:/Research/plugins/tools/reid_lab/work")
HERE = WORK / "v_leak_1"
R1 = WORK / "fuse_r1"
V0 = WORK / "v_leak_0"
APPLY = [R1 / "prep.py", R1 / "evidence.py", R1 / "linker.py", R1 / "linker2.py", R1 / "method.py", R1 / "pairmodel.py", R1 / "selfcheck.py",
         WORK / "m_cuts/features.py", WORK / "m_appear/features.py", WORK / "m_appear/core.py", WORK / "m_appear/agglo.py",
         WORK / "m_space/features.py", WORK / "m_space/geom.py", WORK / "m_space/linker.py"]
BAD_IMPORT = ("score", "reid_lab.score", "gbf.evaluate", "gsr", "fuse_labels", "truth", "labels", "fitdata", "fitlabels", "labels_for_fit", "vr_labels")
BAD_TEXT = ("Labels-GameState", "reentry-eval", "sn-gsr", "/external/", "\\external\\")
L = ["# v_leak round 1: static check of fuse_r1's apply path", ""]

# ---------------------------------------------------------------- 1. imports
L += ["## 1. Imports of the apply path (ast)", "", "| file | imports | label import | where | annotation path in a string literal |", "|---|---|---|---|---|"]
for f in APPLY:
    src = f.read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods, bad = set(), []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
        for n in names:
            mods.add(n.split(".")[0] if not n.startswith(("reid_lab", "gbf")) else n)
            if n in BAD_IMPORT or n.split(".")[-1] in BAD_IMPORT:
                # where: module level or inside a function (lazy import of a fit function)
                bad.append((n, node.lineno))
    where = []
    for n, line in bad:
        func = None
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef,)) and fn.lineno <= line <= max(getattr(fn, "end_lineno", fn.lineno), fn.lineno):
                func = fn.name
        where.append(f"{n} line {line} in {'def ' + func if func else 'MODULE LEVEL'}")
    strs = sorted({s.value for s in ast.walk(tree) if isinstance(s, ast.Constant) and isinstance(s.value, str) and len(s.value) < 200
                   and any(b in s.value for b in BAD_TEXT)})
    L.append(f"| {f.relative_to(WORK).as_posix()} | {', '.join(sorted(m for m in mods if m))} | {'yes' if bad else 'no'} | {'; '.join(where) or '-'} | {strs or '-'} |")
L.append("")

# ---------------------------------------------------------------- 2. fitted files against the blind re-fits of round 0
L += ["## 2. Fitted files read per fold against round 0's blindfolded re-fits", "",
      "| fold | m_cuts weights: max abs diff | m_space weights: max abs diff | fold in m_space fit_clips | cuts in fuse_r1/cache/cuts.json equal to the blind re-fit's (13 clips) |", "|---|---|---|---|---|"]
loco = json.load(open(WORK / "m_cuts/loco_params.json", encoding="utf-8"))["folds"]
r1cuts = json.load(open(R1 / "cache/cuts.json", encoding="utf-8"))
norm = lambda c: sorted([int(p), int(f)] for p, f in c)  # noqa: E731
worst = 0.0
allcuts_equal = True
for fold in data.DEV_CLIPS:
    a = json.load(open(V0 / f"refit/cuts_{fold}.json", encoding="utf-8"))
    t = loco[fold]
    d1 = max(float(np.max(np.abs(np.asarray(a["par"][k], float) - np.asarray(t[k], float)))) for k in ("mu", "sd", "w"))
    d1 = max(d1, abs(a["par"]["b"] - t["b"]))
    b = json.load(open(V0 / f"refit/space_{fold}.json", encoding="utf-8"))
    s = json.load(open(WORK / f"m_space/params_cache/excl_{fold.split('-')[1]}.json", encoding="utf-8"))
    d2 = max(float(np.max(np.abs(np.asarray(b["par"][k], float) - np.asarray(s[k], float)))) for k in ("s", "w", "g_in", "g_out"))
    same = sum(norm(a["cuts"][c]) == norm(r1cuts.get(f"{c}|{fold}", [[-1, -1]])) for c in data.DEV_CLIPS)
    allcuts_equal &= same == 13
    worst = max(worst, d1, d2)
    L.append(f"| {fold} | {d1:.1e} | {d2:.1e} | {fold in (s.get('fit_clips') or [])} | {same} of 13 |")
L += ["", f"Largest difference of any weight: {worst:.1e}; all cut lists equal: {allcuts_equal}.", ""]

# ---------------------------------------------------------------- 3. modification times
ts = lambda p: time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(p)))  # noqa: E731
L += ["## 3. Modification times (local time, 2026-11-15)", "", "| file | modified |", "|---|---|"]
for p in [WORK / "m_cuts/features.py", WORK / "m_cuts/loco_params.json", WORK / "m_cuts/final_params.json", WORK / "m_appear/features.py", WORK / "m_appear/core.py",
          WORK / "m_space/features.py", WORK / "m_space/geom.py", WORK / "m_space/linker.py", WORK / "m_space/params_all13.json", WORK / "m_space/params_cache/excl_162.json",
          V0 / "refit/cuts_SNGS-162.json", V0 / "refit/space_SNGS-162.json",
          R1 / "PREREG.txt", R1 / "prep.py", R1 / "evidence.py", R1 / "linker.py", R1 / "linker2.py", R1 / "pairmodel.py", R1 / "rules.py", R1 / "fit_loco.py", R1 / "method.py",
          R1 / "logs/prep.log", R1 / "logs/check_linker2.log", R1 / "logs/fit_loco.log", R1 / "cache/loco_raw.pkl", R1 / "out/run1/tradeoff.md", R1 / "METHOD_strict.json",
          R1 / "out/cli_scores.txt", R1 / "params_loco.json", R1 / "logs/fit_loco_run2.log", R1 / "README.md"]:
    if p.exists():
        L.append(f"| {p.relative_to(WORK).as_posix()} | {ts(p)} |")
oldest_r1_result = min(os.path.getmtime(p) for p in (R1 / "logs").glob("*") if p.name not in ("prep.log", "check_linker2.log"))
L += ["", f"PREREG.txt modified {ts(R1 / 'PREREG.txt')}; rules.py {ts(R1 / 'rules.py')}; fit_loco.py {ts(R1 / 'fit_loco.py')}; "
      f"oldest log of a label-using fuse_r1 script: {time.strftime('%H:%M:%S', time.localtime(oldest_r1_result))}.", ""]
(HERE / "out").mkdir(exist_ok=True)
(HERE / "out/static.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
