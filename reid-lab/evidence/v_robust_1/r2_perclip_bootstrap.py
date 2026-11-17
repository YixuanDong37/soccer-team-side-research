"""v_robust_1 / r2_perclip_bootstrap.py -- (uses the scored outcomes of r1; no method code here)
per-clip results, concentration of the gain, and the clip bootstrap.

    python -I -B r2_perclip_bootstrap.py      -> out/r2_perclip.md, out/r2_bootstrap.json

Bootstrap: 2000 resamples of the 13 clips with replacement, numpy default_rng(0); the SAME
resampled clips are used for the method and for every baseline (paired). Reported per operating
point: pooled right / wrong share and count-equivalents, and the paired differences against
plugin, split, mutual and "cuts only".
"""
import itertools
import json
import sys
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import numpy as np

import vr_common as vc
from vr_common import data

R1 = json.loads((vc.OUT / "r1_reproduce.json").read_text(encoding="utf-8"))
# strict / balanced / recall: held-out (rule B per fold). fixed_*: the frozen setting with fold weights
# (NOT held-out for the choice of the setting).
METHODS = {k: R1["methods"][k] for k in ("strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall")}
BASES = {"plugin": R1["baselines"]["plugin"], "split": R1["baselines"]["split"], "mutual": R1["baselines"]["mutual"],
         "cuts_only": R1["baselines"]["cuts_only"]}
CLIPS = list(data.DEV_CLIPS)


def vec(res, what):
    out = []
    for s in CLIPS:
        c = Counter(res["per_clip"][s]["per_return"])
        if what == "n":
            out.append(c["right"] + c["wrong"] + c["lost"])
        elif what in ("right", "wrong", "lost"):
            out.append(c[what])
        elif what == "impure":
            out.append(res["per_clip"][s]["impure_rows"])
        elif what == "rows":
            out.append(res["per_clip"][s]["tracked_rows"])
        elif what in ("lk_right", "lk_wrong", "lk_unknown"):
            out.append(res["per_clip"][s]["links"][what[3:]])
    return np.array(out, float)


def q(v, qs=(0.025, 0.05, 0.5, 0.95, 0.975)):
    return [round(float(x), 4) for x in np.quantile(v, qs)]


def g3(v):
    return f"{v[0]:+.3f} / {v[2]:+.3f} / {v[4]:+.3f}"


L = ["# r2: per-clip results, concentration, clip bootstrap", ""]
N = vec(METHODS["recall"], "n")
L += ["## Per clip (held-out): right / wrong / lost", "",
      "| clip | scored | " + " | ".join([*METHODS, *BASES]) + " |", "|---|---|" + "---|" * (len(METHODS) + len(BASES))]
for i, s in enumerate(CLIPS):
    cells = []
    for res in [*METHODS.values(), *BASES.values()]:
        c = Counter(res["per_clip"][s]["per_return"])
        cells.append(f"{c['right']} / {c['wrong']} / {c['lost']}")
    L.append(f"| {s} | {int(N[i])} | " + " | ".join(cells) + " |")
L.append("")

# ---------------------------------------------------------------- concentration
L += ["## Is the gain carried by one or two clips?", ""]
conc = {}
for name, res in METHODS.items():
    R, W = vec(res, "right"), vec(res, "wrong")
    Rp, Wp = vec(BASES["plugin"], "right"), vec(BASES["plugin"], "wrong")
    order = np.argsort(-R)
    top1, top2 = R[order[0]] / R.sum(), R[order[:2]].sum() / R.sum()
    share = R / N
    lo1 = [(R.sum() - R[i]) / (N.sum() - N[i]) for i in range(13)]
    lw1 = [(W.sum() - W[i]) / (N.sum() - N[i]) for i in range(13)]
    lo2 = [(R.sum() - R[i] - R[j]) / (N.sum() - N[i] - N[j]) for i, j in itertools.combinations(range(13), 2)]
    lw2 = [(W.sum() - W[i] - W[j]) / (N.sum() - N[i] - N[j]) for i, j in itertools.combinations(range(13), 2)]
    gain = R - Rp
    gorder = np.argsort(-gain)
    conc[name] = {
        "right_total": int(R.sum()), "top1_clip": CLIPS[order[0]], "top1_share_of_right": round(float(top1), 3),
        "top2_share_of_right": round(float(top2), 3), "largest_clip_share_of_scored": round(float(N.max() / N.sum()), 3),
        "per_clip_right_share_min_med_max": [round(float(x), 3) for x in (share.min(), np.median(share), share.max())],
        "clips_with_right_share_ge_half": int((share >= 0.5).sum()),
        "clips_beating_plugin_right": int((R > Rp).sum()), "clips_tie_plugin_right": int((R == Rp).sum()), "clips_below_plugin_right": int((R < Rp).sum()),
        "clips_wrong_gt_plugin": int((W > Wp).sum()), "clips_wrong_lt_plugin": int((W < Wp).sum()),
        "gain_vs_plugin_total": int(gain.sum()), "gain_top2_clips": [CLIPS[i] for i in gorder[:2]],
        "gain_top2_share": round(float(gain[gorder[:2]].sum() / gain.sum()), 3) if gain.sum() else None,
        "leave_one_out_right_share_min_max": [round(float(min(lo1)), 3), round(float(max(lo1)), 3)],
        "leave_one_out_wrong_share_min_max": [round(float(min(lw1)), 4), round(float(max(lw1)), 4)],
        "leave_two_out_right_share_min_max": [round(float(min(lo2)), 3), round(float(max(lo2)), 3)],
        "leave_two_out_wrong_share_min_max": [round(float(min(lw2)), 4), round(float(max(lw2)), 4)],
        "wrong_total": int(W.sum()), "wrong_top1_clip": CLIPS[int(np.argmax(W))], "wrong_top1_share": round(float(W.max() / W.sum()), 3) if W.sum() else None,
        "sign_test_right_vs_plugin": f"{int((R > Rp).sum())} clips better, {int((R < Rp).sum())} worse, {int((R == Rp).sum())} equal",
    }
    c = conc[name]
    L += [f"**{name}** ({int(R.sum())} right, {int(W.sum())} wrong of {int(N.sum())}):",
          f"- the clip with the most right returns ({c['top1_clip']}) holds {100 * c['top1_share_of_right']:.0f}% of them, the top two {100 * c['top2_share_of_right']:.0f}% "
          f"(the largest clip has {100 * c['largest_clip_share_of_scored']:.0f}% of the scored returns);",
          f"- right share per clip: min {c['per_clip_right_share_min_med_max'][0]}, median {c['per_clip_right_share_min_med_max'][1]}, max {c['per_clip_right_share_min_med_max'][2]}; "
          f"{c['clips_with_right_share_ge_half']} of 13 clips at or above one half;",
          f"- against plugin: {c['sign_test_right_vs_plugin']} in right returns; wrong higher in {c['clips_wrong_gt_plugin']} clips, lower in {c['clips_wrong_lt_plugin']}; "
          f"gain {c['gain_vs_plugin_total']} right returns, {100 * (c['gain_top2_share'] or 0):.0f}% of it in {c['gain_top2_clips']};",
          f"- pooled right share with any one clip removed: {c['leave_one_out_right_share_min_max']}; any two removed: {c['leave_two_out_right_share_min_max']};",
          f"- pooled wrong share with any one clip removed: {c['leave_one_out_wrong_share_min_max']}; any two removed: {c['leave_two_out_wrong_share_min_max']};",
          f"- wrong returns: {c['wrong_total']}, of which {100 * (c['wrong_top1_share'] or 0):.0f}% in {c['wrong_top1_clip']}.", ""]

# ---------------------------------------------------------------- bootstrap
rng = np.random.default_rng(0)
B = 2000
idx = rng.integers(0, 13, size=(B, 13))
boot = {"resamples": B, "seed": 0, "points": {}}
L += ["## Clip bootstrap (2000 resamples of the 13 clips, seed 0, paired)", "",
      "Shares are of the scored returns in the resample; quantiles 2.5% / 5% / 50% / 95% / 97.5%. "
      "x137 = the share times 137 (count-equivalent).", ""]


def share(res, what):
    return vec(res, what)[idx].sum(axis=1) / vec(res, "n")[idx].sum(axis=1)


def imp(res):
    return vec(res, "impure")[idx].sum(axis=1) / vec(res, "rows")[idx].sum(axis=1)


for name, res in METHODS.items():
    r, w = share(res, "right"), share(res, "wrong")
    P = {"right_share_q": q(r), "wrong_share_q": q(w), "right_x137_q": [round(x * 137, 1) for x in q(r)], "wrong_x137_q": [round(x * 137, 1) for x in q(w)],
         "impure_share_q": q(imp(res)),
         "p_wrong_share_le": {f"{k}/137": round(float((w <= k / 137 + 1e-12).mean()), 3) for k in (4, 7, 14)}, "vs": {}}
    lk_r, lk_w = vec(res, "lk_right")[idx].sum(axis=1), vec(res, "lk_wrong")[idx].sum(axis=1)
    P["link_precision_q"] = q(lk_r / np.maximum(lk_r + lk_w, 1))
    L += [f"### {name}", "",
          f"- right share {P['right_share_q']} (x137: {P['right_x137_q']})",
          f"- wrong share {P['wrong_share_q']} (x137: {P['wrong_x137_q']}); P(wrong share <= 4/137, 7/137, 14/137) = {list(P['p_wrong_share_le'].values())}",
          f"- link precision (judged links) {P['link_precision_q']}; impure row share {P['impure_share_q']}", "",
          "| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |",
          "|---|---|---|---|---|---|---|---|"]
    for bname, bres in BASES.items():
        br, bw = share(bres, "right"), share(bres, "wrong")
        dr, dw, di = r - br, w - bw, imp(res) - imp(bres)
        V = {"right_share_diff_q": q(dr), "p_right_higher": round(float((dr > 0).mean()), 4), "wrong_share_diff_q": q(dw),
             "p_wrong_lower": round(float((dw < 0).mean()), 4), "p_wrong_not_higher": round(float((dw <= 0).mean()), 4),
             "impure_diff_q": q(di), "p_impure_lower": round(float((di < 0).mean()), 4)}
        P["vs"][bname] = V
        L.append(f"| {bname} | {g3(V['right_share_diff_q'])} | {V['p_right_higher']} | {g3(V['wrong_share_diff_q'])} | {V['p_wrong_lower']} | {V['p_wrong_not_higher']} | "
                 f"{V['impure_diff_q'][0]:+.4f} / {V['impure_diff_q'][2]:+.4f} / {V['impure_diff_q'][4]:+.4f} | {V['p_impure_lower']} |")
    L.append("")
    boot["points"][name] = P
L += ["### Baselines (same resamples)", ""]
for bname, bres in BASES.items():
    r, w = share(bres, "right"), share(bres, "wrong")
    boot.setdefault("baselines", {})[bname] = {"right_share_q": q(r), "wrong_share_q": q(w), "impure_share_q": q(imp(bres))}
    L.append(f"- {bname}: right share {q(r)}, wrong share {q(w)}, impure {q(imp(bres))}")
i162 = CLIPS.index("SNGS-162")
has = (idx == i162).any(axis=1)
L += ["", f"Resamples without SNGS-162: {int((~has).sum())} of {B} ({100 * (~has).mean():.1f}%). Wrong share in those / in the others:", ""]
for name, res in METHODS.items():
    w = share(res, "wrong")
    L.append(f"- {name}: without SNGS-162 median {np.median(w[~has]):.4f} (max {w[~has].max():.4f}); with it median {np.median(w[has]):.4f} (max {w[has].max():.4f})")
    boot["points"][name]["wrong_share_median_without_162"] = round(float(np.median(w[~has])), 4)
    boot["points"][name]["wrong_share_median_with_162"] = round(float(np.median(w[has])), 4)
boot["concentration"] = conc
(vc.OUT / "r2_bootstrap.json").write_text(json.dumps(boot, indent=1), encoding="utf-8")
(vc.OUT / "r2_perclip.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
