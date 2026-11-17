"""Model comparison inside leave-one-clip-out: held-out log-likelihood of the right answer and
the pooled held-out return counts over a grid of thresholds, for several feature sets.
(Labels: fit on the 12 fit clips; scoring of the held-out clip via reid_lab.score.)

python -I explore_models.py [variant ...]
"""
import json
import sys

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data, score  # noqa: E402
import linker  # noqa: E402

VARIANTS = {
    "base": dict(resid="r", pair=("lu", "ls", "iv", "tu"), ctx=("one",)),
    "base_cv": dict(resid="r_cv", pair=("lu", "ls", "iv", "tu"), ctx=("one",)),
    "raw_d": dict(resid="d", pair=("lu", "ls", "iv", "tu"), ctx=("one",)),
    "no_iv": dict(resid="r", pair=("lu", "ls", "tu"), ctx=("one",)),
    "ctx_edge": dict(resid="r", pair=("lu", "ls", "iv", "tu"), ctx=("one", "edge")),
    "ctx_full": dict(resid="r", pair=("lu", "ls", "iv", "tu"), ctx=("one", "edge", "time")),
    "ctx_full_ee": dict(resid="r", pair=("lu", "ls", "iv", "tu", "ee", "ex"), ctx=("one", "edge", "time")),
    "cv_ctx_full": dict(resid="r_cv", pair=("lu", "ls", "iv", "tu"), ctx=("one", "edge", "time")),
    "cv_ctx_full_ee": dict(resid="r_cv", pair=("lu", "ls", "iv", "tu", "ee", "ex"), ctx=("one", "edge", "time")),
    "cv_fb": dict(resid="r_cv", pair=("lu", "ls", "iv", "tu", "fv", "bv"), ctx=("one", "edge", "time")),
    "cv_fb_ee": dict(resid="r_cv", pair=("lu", "ls", "iv", "tu", "ee", "ex", "fv", "bv"), ctx=("one", "edge", "time")),
    "cv_fb_ee_noiv": dict(resid="r_cv", pair=("lu", "ls", "tu", "ee", "ex", "fv", "bv"), ctx=("one", "edge", "time")),
    "adv": dict(resid="r_adv", pair=("lu", "ls", "tu", "ee", "ex", "fv", "bv"), ctx=("one", "edge", "time")),
    "adv_cv": dict(resid="r_adv_cv", pair=("lu", "ls", "tu", "ee", "ex", "fv", "bv"), ctx=("one", "edge", "time")),
    "adv_cv_fa": dict(resid="r_adv_cv", pair=("lu", "ls", "tu", "ee", "ex", "fa", "ba"), ctx=("one", "edge", "time")),
    "cv_fa": dict(resid="r_cv", pair=("lu", "ls", "tu", "ee", "ex", "fa", "ba"), ctx=("one", "edge", "time")),
    "cv_fb_ee2": dict(resid="r_cv", pair=("lu", "ls", "tu", "ee", "ex", "fv", "bv", "lu2", "ls2"), ctx=("one", "edge", "time")),
    "cv_fa_cen": dict(resid="r_cv", pair=("lu", "ls", "tu", "ee", "ex", "fa", "ba"), ctx=("one", "edge", "time", "census")),
    "adv_cen": dict(resid="r_adv_cv", pair=("lu", "ls", "tu", "ee", "ex", "fa", "ba"), ctx=("one", "edge", "time", "census")),
    "adv_cen_notime": dict(resid="r_adv_cv", pair=("lu", "ls", "tu", "ee", "ex", "fa", "ba"), ctx=("one", "edge", "census")),
    "adv_ew3": dict(resid="r_adv_cv", pair=("lu", "ls", "tu", "ee", "ex", "fa", "ba"), ctx=("one", "edge", "time"), edge_weight=3.0),
    "adv_ew10": dict(resid="r_adv_cv", pair=("lu", "ls", "tu", "ee", "ex", "fa", "ba"), ctx=("one", "edge", "time"), edge_weight=10.0),
    "adv_ew3_2": dict(resid="r_adv_cv", pair=("lu", "ls", "tu", "ee", "ex", "fa", "ba", "lu2", "ls2"), ctx=("one", "edge", "time"), edge_weight=3.0),
    "r_fb_ee": dict(resid="r", pair=("lu", "ls", "iv", "tu", "ee", "ex", "fv", "bv"), ctx=("one", "edge", "time")),
}
THETAS = (0.6, 0.5, 0.4, 0.3)
MODES = (("sinkhorn", True, 2),)


def fmt(d):
    return f"R{d['right']:3d} W{d['wrong']:2d} L{d['lost']:3d} | links {d['links']:3d} r{d['links_right']:3d} w{d['links_wrong']:2d} u{d['links_unknown']:3d} | impure {d['impure_row_share']:.4f}"


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    results = {}
    for name in names:
        kw = VARIANTS[name]
        lp = linker.loco_params(**kw)
        tot, n, tot_e, n_e = 0.0, 0, 0.0, 0
        for seq, pr in lp.items():
            a, b = linker.heldout_nll(pr, seq)
            tot += a
            n += b
            a, b = linker.heldout_nll(pr, seq, edge_only=True)
            tot_e += a
            n_e += b
        print(f"== {name}: {kw}  held-out NLL per group {tot / n:.4f} ({n} groups); border groups only {tot_e / n_e:.4f} ({n_e})")
        pr = lp[data.DEV_CLIPS[0]]
        print("   example fold params: s", np.round(pr["s"], 3), "w", np.round(pr["w"], 3), "g_in", np.round(pr["g_in"], 3), "g_out", np.round(pr["g_out"], 3))
        results[name] = {"nll": tot / n, "curves": {}}
        for mode, peel, emit in MODES:
            for theta in THETAS:
                held = score.evaluate(linker.method_links(lp, theta, mode, peel, emit))["pooled"]
                results[name]["curves"][f"{mode}|{peel}|{emit}|{theta}"] = held
                print(f"   {mode:8s} peel={peel!s:5} emit={emit} theta={theta:.2f}  held-out: {fmt(held)}")
    json.dump(results, open("explore_models.json", "w"), indent=1)
