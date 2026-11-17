"""v_leak stage 3 (LABELS of the 12 fit clips only; the held-out clip is blindfolded).

One LOCO fold of the fused method, end to end, in a process that cannot read the held-out clip's
annotation and that reads NO fitted file of any agent: the cuts come from stage 1 (m_cuts re-fitted
without the clip), the position weights from stage 2 (m_space re-fitted without the clip), the evidence
tables are rebuilt in this directory's own cache, the pair model and the margins are fitted here with
fuse's code. The held-out clip gets links (label-free apply) but is never scored here.

    python -I -B s3_blind_fuse.py SNGS-162      -> out/blind/SNGS-162.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = "C:/Research/plugins/tools/reid_lab/work/v_leak_0/"
sys.path.insert(0, HERE)
import blind  # noqa: E402

fold = sys.argv[1]
blind.install(fold)
blind.selftest(fold)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data, score  # noqa: E402
import prep  # noqa: E402        (fuse, label-free)

prep.CACHE = Path(HERE) / "cache" / fold          # fuse's own cache is never read
prep.CACHE.mkdir(parents=True, exist_ok=True)
import evidence  # noqa: E402    (fuse, label-free)
import linker  # noqa: E402      (fuse, label-free)
import pairmodel  # noqa: E402   (fuse)
from method import bonus_of  # noqa: E402   (fuse, label-free apply path)
import fuse_labels  # noqa: E402 (fuse, LABELS: fit clips only)

MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)   # fuse's grid
FLOOR, T0, T_ABS = 0.3, 0.5, 0.5
COLS = ["cos_w", "sink_lo", "plug"]
LAMBDA = {"strict": 10.0, "balanced": 3.0, "recall": 1.0}

t0 = time.time()
cuts = {s: prep.norm_cuts(c) for s, c in json.load(open(HERE + f"refit/cuts_{fold}.json", encoding="utf-8"))["cuts"].items()}
pp = json.load(open(HERE + f"refit/space_{fold}.json", encoding="utf-8"))["par"]
fit_clips = [s for s in data.DEV_CLIPS if s != fold]
tabs = {s: evidence.pair_table(s, cuts[s], pp) for s in data.DEV_CLIPS}
lab = pd.concat([fuse_labels.label_pairs(tabs[s], s, cuts[s], 1) for s in fit_clips], ignore_index=True)
model = pairmodel.fit(lab, COLS)
fit_res, held_links = {}, {}
for s in data.DEV_CLIPS:
    bonus = bonus_of(tabs[s], model)
    cl = [list(x) for x in cuts[s]]
    for m in MARGINS:
        links = linker.agglomerate(s, cuts[s], bonus, t_abs=T_ABS, margin=m, floor=FLOOR, t0=T0)
        if s == fold:
            held_links[str(m)] = links                      # never scored in this process
        else:
            r = score.score_clip(s, links, cl)
            c = Counter(r["per_return"])
            fit_res[(m, s)] = {"right": c["right"], "wrong": c["wrong"], "lost": c["lost"], "links_wrong": r["links"]["wrong"]}
J, chosen = {}, {}
for point, lam in LAMBDA.items():
    best = None
    for m in MARGINS:
        right = sum(fit_res[(m, s)]["right"] for s in fit_clips)
        pen = sum(fit_res[(m, s)]["wrong"] + fit_res[(m, s)]["links_wrong"] for s in fit_clips)
        j = right - lam * pen
        J.setdefault(point, {})[str(m)] = j
        if best is None or j >= best[0]:                    # ties -> the larger margin (fuse's rule)
            best = (j, m)
    chosen[point] = best[1]
aud = blind.audit(list(blind.OPENED))
mine = HERE.lower()
foreign = [p for p in aud["work_derived"] if not p.lower().startswith(mine)]
out = {"fold": fold, "pair_model": model, "margin": chosen, "J": J, "cuts": [list(x) for x in cuts[fold]], "links": held_links,
       "fit_pooled": {str(m): {k: sum(fit_res[(m, s)][k] for s in fit_clips) for k in ("right", "wrong", "lost", "links_wrong")} for m in MARGINS},
       "audit": {**aud, "work_derived_foreign": foreign, "work_derived": sorted(set(p.split("/cache/")[0] + "/cache/..." if "/cache/" in p else p for p in aud["work_derived"]))},
       "seconds": round(time.time() - t0, 1)}
Path(HERE + "out/blind").mkdir(parents=True, exist_ok=True)
json.dump(out, open(HERE + f"out/blind/{fold}.json", "w"), indent=1)
print(fold, "blind fuse fold: margins", chosen, "| pair model w", np.round(model["w"], 4).tolist(), "b", round(model["b"], 4),
      "| label files opened:", len(aud["label_files"]), "(held-out:", sum(fold in p for p in aud["label_files"]), ") | fitted files of other agents read:", foreign,
      "| blocked:", aud["blocked"], "|", out["seconds"], "s")
