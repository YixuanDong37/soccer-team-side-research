"""v_robust_0 / r10_folds.py -- (LABELS were used by fuse to produce the raw LOCO results read here;
this script refits two pair models with labels as a check) integrity and stability of fuse's LOCO.

    python -I -B r10_folds.py     -> out/r10_folds.md, out/r10_folds.json

Reads fuse/cache/loco_raw.pkl (every fold x margin x clip, written by fuse/fit_loco.py; read-only).
1. The margins in params_loco.json are re-derived from the raw results with fuse's rule.
2. How flat the rule's objective J is per fold (the distance between the best and the second-best
   margin), and what the rule would have chosen with lam moved by +-10% / +-50%.
3. Parameter transfer: every clip scored with the numbers of every fold (13 folds + all 13) at a
   fixed margin; the held-out fold against the 12 folds that had the clip in their fit set.
4. The pair model of two folds refitted here from the fit clips' weak labels (must equal
   params_loco.json).
"""
import json
import pickle
import sys
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_0")
import numpy as np

import vr_common as vc
from vr_common import data, evidence, method, pairmodel, prep

raw = pickle.load(open(vc.FUSE_CACHE / "loco_raw.pkl", "rb"))["folds"]
PAR = vc.params()
MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)
LAM = {"strict": 10.0, "balanced": 3.0, "recall": 1.0}
CLIPS = list(data.DEV_CLIPS)
J = {}
L = ["# r10: integrity and stability of fuse's leave-one-clip-out run", ""]


def pool(items):
    return vc.pool(items)


def jval(fold, m, lam, clips):
    p = pool(raw[fold]["res"][("fuse", m, s)] for s in clips)
    return p["right"] - lam * (p["wrong"] + p["links_wrong"])


def choose(fold, lam):
    fit = [s for s in CLIPS if s != fold]
    best = None
    for m in MARGINS:
        j = jval(fold, m, lam, fit)
        if best is None or j >= best[0]:
            best = (j, m)
    return best[1]


# 1 ------------------------------------------------------------------------------------------
bad = []
for c in CLIPS:
    for point, lam in LAM.items():
        if choose(c, lam) != PAR[c]["margin"][point]:
            bad.append((c, point, choose(c, lam), PAR[c]["margin"][point]))
J["margin_rederived_mismatches"] = bad
L += ["## 1. Margins re-derived from the raw results with fuse's rule", "", f"Mismatches with params_loco.json: {bad or 'none (39 of 39 equal)'}.", ""]
held = {point: pool(raw[c]["res"][("fuse", PAR[c]["margin"][point], c)] for c in CLIPS) for point in LAM}
L += ["Held-out pooled result re-derived from the raw results: " + "; ".join(f"{p} {vc.fmt_pool(v)}" for p, v in held.items()), ""]
J["heldout_rederived"] = held

# 2 ------------------------------------------------------------------------------------------
L += ["## 2. How flat is the objective, and what if lam had been a little different?", "",
      "J of the chosen margin minus J of the best other margin on the 12 fit clips, per fold (small = the choice can flip):", "",
      "| fold (held-out clip) | strict: chosen, J, runner-up (J) | balanced | recall |", "|---|---|---|---|"]
flat = {}
for c in CLIPS:
    fit = [s for s in CLIPS if s != c]
    cells = []
    for point, lam in LAM.items():
        js = {m: jval(c, m, lam, fit) for m in MARGINS}
        m0 = PAR[c]["margin"][point]
        m1 = max((m for m in MARGINS if m != m0), key=lambda m: (js[m], m))
        flat[(c, point)] = js[m0] - js[m1]
        cells.append(f"{m0} ({js[m0]:.0f}); next {m1} ({js[m1]:.0f})")
    L.append(f"| {c} | " + " | ".join(cells) + " |")
J["objective_gap_to_runner_up"] = {f"{c}|{p}": v for (c, p), v in flat.items()}
L += ["", "Held-out pooled result when the rule's penalty lam is changed (same rule, same raw results):", "",
      "| lam | margins chosen per fold | held-out right / wrong / lost; links r / w / u; impure |", "|---|---|---|"]
J["lam_sweep"] = {}
for lam in (20, 15, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3.3, 3, 2.7, 2, 1.5, 1.1, 1, 0.9, 0.5):
    ch = {c: choose(c, lam) for c in CLIPS}
    p = pool(raw[c]["res"][("fuse", ch[c], c)] for c in CLIPS)
    cnt = Counter(ch.values())
    J["lam_sweep"][str(lam)] = {"chosen": ch, "heldout": p}
    tag = {10: " (strict)", 3: " (balanced)", 1: " (recall)"}.get(lam, "")
    L.append(f"| {lam}{tag} | {', '.join(f'{n}x {m}' for m, n in sorted(cnt.items()))} | {vc.fmt_pool(p).replace(' | ', '; ')} |")
L.append("")

# 3 ------------------------------------------------------------------------------------------
L += ["## 3. Parameter transfer: each clip with the numbers of every fold (fixed margin)", "",
      "'held-out' = the clip's own fold (fitted without it); 'in-fit' = the 12 folds whose fit set contains the clip; all13 = the frozen numbers.", "",
      "| margin | held-out pooled right / wrong | all13 pooled | in-fit folds: pooled right (min - max over a random fold per clip is not shown; per-clip sums of min / mean / max) | wrong (sum of per-clip min / mean / max) | clips whose result differs between any two folds |",
      "|---|---|---|---|---|---|"]
J["transfer"] = {}
for m in (0.2, 0.25, 0.5, 0.8, 0.9):
    ho = pool(raw[c]["res"][("fuse", m, c)] for c in CLIPS)
    al = pool(raw["all13"]["res"][("fuse", m, c)] for c in CLIPS)
    rmin = rmax = wmin = wmax = 0
    rmean = wmean = 0.0
    differ = []
    for c in CLIPS:
        rr = [vc.counts(raw[f]["res"][("fuse", m, c)]["per_return"])["right"] for f in CLIPS if f != c]
        ww = [vc.counts(raw[f]["res"][("fuse", m, c)]["per_return"])["wrong"] for f in CLIPS if f != c]
        rmin, rmax, rmean = rmin + min(rr), rmax + max(rr), rmean + float(np.mean(rr))
        wmin, wmax, wmean = wmin + min(ww), wmax + max(ww), wmean + float(np.mean(ww))
        allr = {tuple(raw[f]["res"][("fuse", m, c)]["per_return"]) for f in [*CLIPS, "all13"]}
        if len(allr) > 1:
            differ.append(c)
    J["transfer"][str(m)] = {"heldout": ho, "all13": al, "infit_right_min_mean_max": [rmin, round(rmean, 1), rmax],
                             "infit_wrong_min_mean_max": [wmin, round(wmean, 1), wmax], "clips_that_differ": differ}
    L.append(f"| {m} | {ho['right']} / {ho['wrong']} | {al['right']} / {al['wrong']} | {rmin} / {rmean:.1f} / {rmax} | {wmin} / {wmean:.1f} / {wmax} | {len(differ)}: {differ} |")
L.append("")
# optimism: held-out minus mean in-fit, per clip
L += ["Held-out minus the mean of the 12 in-fit folds, summed over clips (negative right = held-out is worse = the in-sample numbers are optimistic):", ""]
for m in (0.2, 0.25, 0.5, 0.8, 0.9):
    dr = dw = 0.0
    for c in CLIPS:
        h = vc.counts(raw[c]["res"][("fuse", m, c)]["per_return"])
        dr += h["right"] - np.mean([vc.counts(raw[f]["res"][("fuse", m, c)]["per_return"])["right"] for f in CLIPS if f != c])
        dw += h["wrong"] - np.mean([vc.counts(raw[f]["res"][("fuse", m, c)]["per_return"])["wrong"] for f in CLIPS if f != c])
    J["transfer"][str(m)]["heldout_minus_infit"] = [round(dr, 2), round(dw, 2)]
    L.append(f"- margin {m}: right {dr:+.1f}, wrong {dw:+.1f}")
L.append("")

# 4 ------------------------------------------------------------------------------------------
L += ["## 4. Pair model refitted here for two folds (weak labels of the 12 fit clips)", ""]
sys.path.insert(0, str(vc.FUSE))
import pandas as pd  # noqa: E402

import fuse_labels  # noqa: E402  LABELS: the fit of fuse's pair model, repeated as a check

J["refit"] = {}
for fold in ("SNGS-162", "SNGS-107"):
    pp = method.pos_params(fold)
    fit = [s for s in CLIPS if s != fold]
    lab = pd.concat([fuse_labels.label_pairs(evidence.pair_table(s, prep.cuts_for(s, fold), pp), s, prep.cuts_for(s, fold), 1) for s in fit], ignore_index=True)
    model = pairmodel.fit(lab, ["cos_w", "sink_lo", "plug"])
    ref = PAR[fold]["pair_model"]
    diff = float(np.max(np.abs(np.array(model["w"] + [model["b"]]) - np.array(ref["w"] + [ref["b"]]))))
    J["refit"][fold] = {"w": model["w"], "b": model["b"], "n": model["n"], "pos": model["pos"], "max_abs_diff_to_params_loco": diff,
                        "held_out_clip_in_fit_rows": bool((lab.seq == fold).any())}
    L.append(f"- fold {fold}: refit weights {np.round(model['w'], 4).tolist()}, bias {model['b']:.4f}, {model['n']} pairs ({model['pos']} positive); "
             f"largest difference to params_loco.json {diff:.2e}; rows of the held-out clip in the fit table: {bool((lab.seq == fold).any())}")
(vc.OUT / "r10_folds.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(vc.OUT / "r10_folds.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
print("\n".join(L))
