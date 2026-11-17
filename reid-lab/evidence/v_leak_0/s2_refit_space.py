"""v_leak stage 2 (LABELS of the 12 fit clips only; the held-out clip is blindfolded).

Re-fits m_space's position model for one fold from scratch with m_space's own code (linker.fit with
posscore.CONFIG), never touching the held-out clip's annotation, and compares with
m_space/params_cache/excl_<fold>.json (what fuse reads), read only AFTER the fit.

    python -I -B s2_refit_space.py SNGS-162      -> refit/space_SNGS-162.json
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

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import linker  # noqa: E402    (m_space)
import posscore  # noqa: E402  (m_space; only its CONFIG constant is used, params_excluding is NOT called)

t0 = time.time()
fit_clips = tuple(s for s in data.DEV_CLIPS if s != fold)
par = linker.fit(fit_clips, **posscore.CONFIG)
aud = blind.audit(list(blind.OPENED))

theirs = json.load(open(f"C:/Research/plugins/tools/reid_lab/work/m_space/params_cache/excl_{fold.split('-')[1]}.json", encoding="utf-8"))
diff = max(float(np.max(np.abs(np.asarray(par[k], float) - np.asarray(theirs[k], float)))) for k in ("s", "w", "g_in", "g_out"))
same_cfg = all(par[k] == theirs[k] for k in ("resid", "pair", "ctx", "edge_weight"))
out = {"fold": fold, "par": par, "theirs_fit_clips": theirs.get("fit_clips"), "fold_in_their_fit_clips": fold in (theirs.get("fit_clips") or []),
       "max_abs_diff_vs_m_space_excl": diff, "same_config": same_cfg, "n_true": [par["n_true"], theirs["n_true"]], "n_pairs": [par["n_pairs"], theirs["n_pairs"]],
       "audit": aud, "seconds": round(time.time() - t0, 1)}
json.dump(out, open(HERE + f"refit/space_{fold}.json", "w"), indent=1)
print(fold, "m_space refit: max |diff| vs params_cache/excl", f"{diff:.2e}", "| same config:", same_cfg, "| n_true / n_pairs mine, theirs:", out["n_true"], out["n_pairs"],
      "| label files opened:", len(aud["label_files"]), "(held-out:", sum(fold in p for p in aud["label_files"]), ") | derived work files read before the comparison:",
      aud["work_derived"], "| blocked:", aud["blocked"], "|", out["seconds"], "s")
