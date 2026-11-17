"""v_leak: NESTED choice of the pair model's inputs (the held-out clip is blindfolded).

fuse picked the inputs of its pair model (cos_w, sink_lo, plug) by leave-one-clip-out log-loss on the pairs
of ALL 13 clips (d2_pairmodel.py, d4_table2d.py: "a design choice made on the dev clips"), so every fold's
held-out clip took part in the feature choice. Here the choice is repeated inside each fold: the input set
with the lowest INNER leave-one-clip-out log-loss over the 12 fit clips (fuse's own criterion and candidate
sets, those the linker can use, i.e. with cos_w as the appearance input). The chosen model is then fitted on
the 12 fit clips, the margin chosen by fuse's J rule on the 12 fit clips, and the held-out clip linked
(never scored here). If the chosen set is fuse's own, the fold's result is the blind re-run's (stage 3).

    python -I -B n2_nested_cols.py SNGS-162   -> out/nested_cols/SNGS-162.json
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
import prep  # noqa: E402

prep.CACHE = Path(HERE) / "cache" / fold
prep.CACHE.mkdir(parents=True, exist_ok=True)     # tables are rebuilt here when stage 3's cache was removed
import evidence  # noqa: E402
import linker  # noqa: E402
import pairmodel  # noqa: E402
from method import bonus_of  # noqa: E402
import fuse_labels  # noqa: E402

SETS = {
    "cos_w": ["cos_w"],
    "cos_w+sink": ["cos_w", "sink"],
    "cos_w+sink_lo": ["cos_w", "sink_lo"],
    "cos_w+sink+plug": ["cos_w", "sink", "plug"],
    "cos_w+sink_lo+plug": ["cos_w", "sink_lo", "plug"],                      # fuse's choice
    "cos_w+unreach+sink": ["cos_w", "unreach", "sink"],
    "cos_w+unreach+sink_lo": ["cos_w", "unreach", "sink_lo"],
    "cos_w+unreach+pos_p": ["cos_w", "unreach", "pos_p"],
    "cos_w+unreach+sink+plug": ["cos_w", "unreach", "sink", "plug"],
    "cos_w+unreach+sink_lo+plug": ["cos_w", "unreach", "sink_lo", "plug"],
    "cos_w+unreach+sink_e/in+plug": ["cos_w", "unreach", "sink_edge", "sink_in", "plug"],
    "cos_w+unreach+sink_lo+lgap": ["cos_w", "unreach", "sink_lo", "lgap"],
    "cos_w+sink_lo+plug+low": ["cos_w", "sink_lo", "plug", "low", "cos_w_low"],
}
FUSE = "cos_w+sink_lo+plug"
MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)
FLOOR, T0, T_ABS = 0.3, 0.5, 0.5
LAMBDA = {"strict": 10.0, "balanced": 3.0, "recall": 1.0}

t0 = time.time()
cuts = {s: prep.norm_cuts(c) for s, c in json.load(open(HERE + f"refit/cuts_{fold}.json", encoding="utf-8"))["cuts"].items()}
pp = json.load(open(HERE + f"refit/space_{fold}.json", encoding="utf-8"))["par"]
fit_clips = [s for s in data.DEV_CLIPS if s != fold]
tabs = {s: evidence.pair_table(s, cuts[s], pp) for s in data.DEV_CLIPS}
lab = pd.concat([fuse_labels.label_pairs(tabs[s], s, cuts[s], 1) for s in fit_clips], ignore_index=True)
inner = {}
for name, cols in SETS.items():
    nll, n = 0.0, 0
    for i in fit_clips:
        m = pairmodel.fit(lab[lab.seq != i], cols)
        h = lab[(lab.seq == i) & lab.known & (lab.imm | ~lab.same)]
        z = pairmodel.logit(h, m)
        y = h.imm.to_numpy(float)
        nll += float((np.logaddexp(0, z) - y * z).sum())
        n += len(y)
    inner[name] = nll / n
best = min(inner, key=inner.get)
out = {"fold": fold, "inner_logloss": inner, "chosen": best, "is_fuse_choice": best == FUSE, "rank_of_fuse_choice": sorted(inner, key=inner.get).index(FUSE) + 1,
       "logloss_gap_fuse_minus_best": inner[FUSE] - inner[best]}
if best != FUSE and len(SETS[best]) > 1:
    model = pairmodel.fit(lab, SETS[best])
    fit_res, held_links = {}, {}
    for s in data.DEV_CLIPS:
        bonus = bonus_of(tabs[s], model)
        cl = [list(x) for x in cuts[s]]
        for m in MARGINS:
            links = linker.agglomerate(s, cuts[s], bonus, t_abs=T_ABS, margin=m, floor=FLOOR, t0=T0)
            if s == fold:
                held_links[str(m)] = links
            else:
                r = score.score_clip(s, links, cl)
                c = Counter(r["per_return"])
                fit_res[(m, s)] = (c["right"], c["wrong"] + r["links"]["wrong"])
    chosen = {}
    for point, lam in LAMBDA.items():
        bestm = None
        for m in MARGINS:
            j = sum(fit_res[(m, s)][0] for s in fit_clips) - lam * sum(fit_res[(m, s)][1] for s in fit_clips)
            if bestm is None or j >= bestm[0]:
                bestm = (j, m)
        chosen[point] = bestm[1]
    out.update({"pair_model": model, "margin": chosen, "links": held_links, "cuts": [list(x) for x in cuts[fold]]})
aud = blind.audit(list(blind.OPENED))
out["audit"] = {"heldout_label_opened": any(fold in p for p in aud["label_files"] + aud["reentry_eval"]), "blocked": aud["blocked"],
                "foreign_fitted_files": [p for p in aud["work_derived"] if not p.lower().startswith(HERE.lower())]}
out["seconds"] = round(time.time() - t0, 1)
Path(HERE + "out/nested_cols").mkdir(parents=True, exist_ok=True)
json.dump(out, open(HERE + f"out/nested_cols/{fold}.json", "w"), indent=1)
top = sorted(inner.items(), key=lambda kv: kv[1])[:3]
print(fold, "inner-LOCO choice:", best, "| fuse's set ranks", out["rank_of_fuse_choice"], "of", len(SETS), "gap", f"{out['logloss_gap_fuse_minus_best']:.5f}",
      "| top 3:", [(k, round(v, 5)) for k, v in top], "| margins:", out.get("margin"), "| blocked:", aud["blocked"], "|", out["seconds"], "s")
