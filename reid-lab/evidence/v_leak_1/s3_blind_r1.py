"""v_leak round 1, stage 3 (LABELS of the 12 fit clips only; the held-out clip is blindfolded).

One LOCO fold of fuse_r1, end to end, in a process that cannot read the held-out clip's annotation and
that reads NO fitted or cached file of fuse_r1 or of any other method agent:
  cuts              round 0's blindfolded re-fit of m_cuts' detector without the clip   (v_leak_0/refit/cuts_<fold>.json)
  position weights  round 0's blindfolded re-fit of m_space's model without the clip    (v_leak_0/refit/space_<fold>.json)
  evidence tables   rebuilt from nothing by s2_tables.py in this directory's cache (fuse_r1's code; its cache is never
                    read; every clip's annotation was blindfolded while they were built)
  pair model, links, scoring of the FIT clips, choice of the setting: fuse_r1's own code
  (pairmodel.fit, linker2.link, fit_loco._scored, rules.choose), unchanged.
The held-out clip gets links for every candidate setting (label-free apply) and is never scored here.

Besides the method ("r1") the fold also runs design alternatives that fuse_r1 fixed by hand with the
round-0 held-out tables in view, so that their choice can be repeated INSIDE the fold (n1_nested.py):
  r0    round 0's linker (no tiers)                      n3 / n10   solid from 3 / 10 boxes (PREREG A1 names them)
  c5    r1 with "a rival must share >= 5 frames with the partner" instead of >= 1 (m_appear's README:
        the rule that was fixed after looking at all 13 clips)

    python -I -B s3_blind_r1.py SNGS-162      -> out/blind/SNGS-162.pkl
"""
import json
import pickle
import sys
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True
HERE = "C:/Research/plugins/tools/reid_lab/work/v_leak_1/"
V0 = "C:/Research/plugins/tools/reid_lab/work/v_leak_0/"
R1 = "C:/Research/plugins/tools/reid_lab/work/fuse_r1/"
sys.path.insert(0, HERE)
import blind  # noqa: E402

fold = sys.argv[1]
blind.install(fold)
blind.selftest(fold)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, R1)
from reid_lab import data, score  # noqa: E402
import prep  # noqa: E402        (fuse_r1, label-free)

prep.CACHE = Path(HERE) / "cache" / "tables"      # built from nothing by s2_tables.py with every clip's annotation blindfolded;
prep.CACHE.mkdir(parents=True, exist_ok=True)     # fuse_r1's own cache is never read
prep._CUTS.clear()
import evidence  # noqa: E402    (fuse_r1, label-free)
import linker  # noqa: E402      (fuse_r1, label-free)
import linker2  # noqa: E402     (fuse_r1, label-free)
import pairmodel  # noqa: E402   (fuse_r1)
import rules  # noqa: E402       (fuse_r1, works on scored counts)
import fit_loco  # noqa: E402    (fuse_r1: constants, _scored; its run_fold / report are NOT called)
from method import LEAF_COLS, bonus_of  # noqa: E402
import fuse_labels  # noqa: E402 (fuse_r1, LABELS: fit clips only)

MARGINS, FLOOR, T0, T_ABS, COLS = fit_loco.MARGINS, fit_loco.FLOOR, fit_loco.T0, fit_loco.T_ABS, fit_loco.COLS
VARIANTS = {"r1": {"n_solid": 5, "covis": 1}, "n3": {"n_solid": 3, "covis": 1}, "n10": {"n_solid": 10, "covis": 1}, "c5": {"n_solid": 5, "covis": 5}}

# linker2 with the co-visibility level of a rival as a parameter: the same source text with one
# expression replaced (covis 1 is linker2 itself; checked below on the first clip)
src = open(R1 + "linker2.py", encoding="utf-8").read()
assert src.count("seen = N >= 1") == 1
linker2c = types.ModuleType("linker2c")
linker2c.COVIS = 1
exec(compile(src.replace("seen = N >= 1", "seen = N >= COVIS"), "linker2c", "exec"), linker2c.__dict__)

t0 = time.time()
cuts = {s: prep.norm_cuts(c) for s, c in json.load(open(V0 + f"refit/cuts_{fold}.json", encoding="utf-8"))["cuts"].items()}
pp = json.load(open(V0 + f"refit/space_{fold}.json", encoding="utf-8"))["par"]
fit_clips = [s for s in data.DEV_CLIPS if s != fold]
full = {s: evidence.pair_table(s, cuts[s], pp, 0) for s in data.DEV_CLIPS}
core = {n: {s: evidence.pair_table(s, cuts[s], pp, n) for s in data.DEV_CLIPS} for n in (5, 3, 10)}
t_tables = time.time() - t0

# ---- fit: labels of the FIT clips only
models = {}
for n in (5, 3, 10):
    lab = pd.concat([fuse_labels.label_pairs(core[n][s], s, cuts[s], fit_loco.MIN_ROWS_FIT) for s in fit_clips], ignore_index=True)
    models[n] = pairmodel.fit(lab, COLS)
lab0 = pd.concat([fuse_labels.label_pairs(full[s], s, cuts[s], fit_loco.MIN_ROWS_FIT) for s in fit_clips], ignore_index=True)
models["r0"] = pairmodel.fit(lab0, COLS)

# ---- links of all 13 clips (label-free), scoring of the fit clips only
cells, held = {}, {}
same_as_linker2 = None
for s in data.DEV_CLIPS:
    cl = [list(x) for x in cuts[s]]
    ident = fuse_labels.identities(s, cuts[s]) if s != fold else None
    b0 = bonus_of(full[s], models["r0"])
    for name, v in VARIANTS.items():
        n = v["n_solid"]
        bc = bonus_of(core[n][s], models[n])
        bl = bonus_of(full[s], models[n], only=LEAF_COLS)
        linker2c.COVIS = v["covis"]
        for m in MARGINS:
            c_links, l_links = linker2c.link(s, cuts[s], bc, bl, margin=m, floor=FLOOR, t0=T0, t_abs=T_ABS, n_solid=n, leaves=True)
            if name == "r1" and s == data.DEV_CLIPS[0] and m in (0.2, 0.6):
                ref = linker2.link(s, cuts[s], bc, bl, margin=m, floor=FLOOR, t0=T0, t_abs=T_ABS, n_solid=n, leaves=True)
                same_as_linker2 = (same_as_linker2 is not False) and ref == (c_links, l_links)
            if s == fold:
                held[(name, m)] = {"core": c_links, "leaf": l_links}             # never scored in this process
            else:
                cells[(name, m, s)] = {"off": fit_loco._scored(score, s, c_links, cl, ident, []),
                                       "on": fit_loco._scored(score, s, c_links + l_links, cl, ident, l_links)}
    for m in MARGINS:
        links = linker.agglomerate(s, cuts[s], b0, t_abs=T_ABS, margin=m, floor=FLOOR, t0=T0)
        if s == fold:
            held[("r0", m)] = {"core": links, "leaf": []}
        else:
            cells[("r0", m, s)] = {"off": fit_loco._scored(score, s, links, cl, ident)}
    if s != fold:
        cells[("cuts", None, s)] = {"off": fit_loco._scored(score, s, [], cl)}

# ---- the setting of each operating point, chosen on the fit clips with fuse_r1's rules.py
chosen = {}
for variant, cands in (("r1", fit_loco.CANDS["r1"]), ("r0", fit_loco.CANDS["r0"])):
    for rule in rules.RULES:
        for point in fit_loco.POINTS:
            best, why = rules.choose(cells, variant, cands, fit_clips, point, rule)
            chosen[(variant, rule, point)] = {"leaves": bool(best[0]), "margin": best[1], "why": why}

aud = blind.audit(list(blind.OPENED))
mine, v0 = HERE.lower(), V0.lower()
foreign = [p for p in aud["work_derived"] if not p.lower().startswith(mine) and not p.lower().startswith(v0 + "refit/")]
out = {"fold": fold, "models": models, "cuts": {s: [list(x) for x in cuts[s]] for s in data.DEV_CLIPS}, "held": held, "cells": cells, "chosen": chosen,
       "same_as_linker2": same_as_linker2,
       "audit": {"label_files": aud["label_files"], "reentry_eval": aud["reentry_eval"], "blocked": aud["blocked"], "work_derived_foreign": foreign,
                 "work_derived": sorted(set(p.split("/cache/")[0] + "/cache/..." if "/cache/" in p else p for p in aud["work_derived"])),
                 "work_code": aud["work_code"]},
       "seconds": round(time.time() - t0, 1), "seconds_tables": round(t_tables, 1)}
Path(HERE + "out/blind").mkdir(parents=True, exist_ok=True)
with open(HERE + f"out/blind/{fold}.pkl", "wb") as fh:
    pickle.dump(out, fh)
ch = {p: (chosen[("r1", "B", p)]["margin"], "on" if chosen[("r1", "B", p)]["leaves"] else "off") for p in fit_loco.POINTS}
print(fold, "blind fuse_r1 fold: rule B", ch, "| r1 pair model w", np.round(models[5]["w"], 4).tolist(), "b", round(models[5]["b"], 4),
      "| linker2c(covis 1) == linker2:", same_as_linker2,
      "| label files opened:", len(aud["label_files"]), "(held-out:", sum(fold in p for p in aud["label_files"]) + sum(fold in p for p in aud["reentry_eval"]),
      ") | fitted / cached files of other agents read:", foreign, "| blocked:", aud["blocked"], "|", out["seconds"], "s")
