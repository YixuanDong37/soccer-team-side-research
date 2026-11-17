"""v_leak stage 1 (LABELS of the 12 fit clips only; the held-out clip is blindfolded).

Re-fits m_cuts' switch detector for one fold from scratch with m_cuts' own code, never touching the
held-out clip's annotation, compares the weights with m_cuts/loco_params.json[fold] (read only AFTER the
fit, for the comparison) and computes the cuts of all 13 clips with the re-fitted weights (label-free).

    python -I -B s1_refit_cuts.py SNGS-162      -> refit/cuts_SNGS-162.json
"""
import json
import sys
import time

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
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
from reid_lab import data  # noqa: E402
import features as F  # noqa: E402  (m_cuts, label-free)
import truth as T  # noqa: E402     (m_cuts, labels)
import loco  # noqa: E402           (m_cuts: fit_lr, MODELS)

t0 = time.time()
MODEL = "lr(cos,mm,lgap,ld)"
P0 = 0.7
fit_clips = [s for s in data.DEV_CLIPS if s != fold]
frames = []
for s in fit_clips:
    c = F.candidates(s)
    lab, which = T.label(s, c)
    frames.append(c.assign(label=lab, sw_id=which))
tab = pd.concat(frames, ignore_index=True)
par = loco.fit_lr(loco.MODELS[MODEL], tab)
t_fit = time.time() - t0
th = float(np.log(P0 / (1 - P0)))
cuts = {s: sorted([int(p), int(f)] for p, f in F.cuts(s, {**par, "threshold": th})) for s in data.DEV_CLIPS}
opened = list(blind.OPENED)          # everything read up to here (before the comparison files)
aud = blind.audit(opened)

# ---- comparison with what m_cuts published (after the fit)
theirs = json.load(open(F.LOCO_PARAMS_FILE, encoding="utf-8"))["folds"][fold]
diff = max(float(np.max(np.abs(np.asarray(par[k], float) - np.asarray(theirs[k], float)))) for k in ("mu", "sd", "w"))
diff = max(diff, abs(par["b"] - theirs["b"]))
their_cuts = {s: sorted([int(p), int(f)] for p, f in F.cuts(s, {**theirs, "threshold": th})) for s in data.DEV_CLIPS}
same_cuts = {s: cuts[s] == their_cuts[s] for s in data.DEV_CLIPS}
out = {"fold": fold, "model": MODEL, "p0": P0, "par": par, "cuts": cuts, "n_candidates_fit": int(len(tab)),
       "labels_fit": tab.label.value_counts().to_dict(), "max_abs_diff_vs_m_cuts_loco_params": diff,
       "cuts_equal_to_m_cuts_weights": same_cuts, "audit": aud, "seconds": round(time.time() - t0, 1)}
json.dump(out, open(HERE + f"refit/cuts_{fold}.json", "w"), indent=1)
print(fold, "m_cuts refit: max |diff| vs loco_params.json", f"{diff:.2e}", "| cuts identical on", sum(same_cuts.values()), "of 13 clips |",
      "label files opened:", len(aud["label_files"]), "(held-out among them:", sum(fold in p for p in aud["label_files"]), ") | derived work files read before the comparison:",
      aud["work_derived"], "| blocked:", aud["blocked"], "|", round(t_fit, 1), "s fit,", out["seconds"], "s total")
