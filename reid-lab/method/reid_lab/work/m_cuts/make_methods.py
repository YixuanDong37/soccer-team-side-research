"""Final outputs of m_cuts.

    python -I make_methods.py

1. Leave-one-clip-out: for every dev clip the logistic regression is fitted on the other 12 clips
   (labels: truth.py) and the cuts of the clip are computed with those weights (label-free code:
   features.cut_table). The fold weights go to loco_params.json (features.cuts_heldout reads them),
   the all-13 weights to final_params.json (features.cuts reads them).
2. METHOD files for reid_lab/score.py, per operating point p0 (cut where P(switch) >= p0):
     methods/heldout_p{p0}_split.json           cuts, no links                       (headline)
     methods/heldout_p{p0}_mutual.json          cuts + the mutual baseline's links moved onto
                                                the sub-pieces (features.remap_links "appearance")
     methods/heldout_p{p0}_mutual_rerun.json    cuts + the mutual rule re-run on the sub-pieces
                                                (relink.mutual_links)
     methods/heldout_p{p0}_plugin.json          cuts + the plugin's re-joins ("adjacent")
     methods/insample_p{p0}_*.json              the same with the all-13 weights (secondary)
3. results.json / results.md: the scores of all of them (score.evaluate) and the cut counts.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
from reid_lab import data, score  # noqa: E402
import features as F  # noqa: E402
import truth as T  # noqa: E402
import loco  # noqa: E402
import relink  # noqa: E402

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
MODEL = "lr(cos,mm,lgap,ld)"
P0S = (0.9, 0.7, 0.5, 0.3)


def variants(cuts: dict) -> dict:
    mutual = score.baseline_links("mutual")
    plugin = score.baseline_links("plugin")
    return {
        "split": {s: {"links": [], "cuts": cuts[s]} for s in data.DEV_CLIPS},
        "mutual": {s: {"links": F.remap_links(s, mutual[s]["links"], cuts[s], "appearance"), "cuts": cuts[s]} for s in data.DEV_CLIPS},
        "mutual_rerun": {s: {"links": relink.mutual_links(s, cuts[s]), "cuts": cuts[s]} for s in data.DEV_CLIPS},
        "plugin": {s: {"links": F.remap_links(s, plugin[s]["links"], cuts[s], "adjacent"), "cuts": cuts[s]} for s in data.DEV_CLIPS},
    }


def counts(sel: pd.DataFrame, n_sw: int) -> dict:
    r = {k: int((sel.label == k).sum()) for k in ("T", "F", "t", "f", "U")}
    r = {"cuts": len(sel), **r, "switches_hit": int(sel[sel.sw_id >= 0].groupby("seq").sw_id.nunique().sum()), "switches": n_sw}
    r["cut_precision"] = round(r["T"] / max(1, r["T"] + r["F"]), 3)
    r["cut_precision_incl_weak"] = round((r["T"] + r["t"]) / max(1, r["T"] + r["F"] + r["t"] + r["f"]), 3)
    return r


def main():
    os.makedirs(HERE + "methods", exist_ok=True)
    tab = loco.table()
    n_sw = sum(len(T.switches(s)) for s in data.DEV_CLIPS)
    cols = loco.MODELS[MODEL]
    res, pars, par, ll = loco.loco(cols, tab, P0S)
    json.dump({"model": MODEL, "note": "per held-out clip: weights fitted on the other 12 dev clips", "folds": pars}, open(HERE + "loco_params.json", "w"), indent=1)
    json.dump({**par, "model": MODEL, "note": "fitted on all 13 dev clips; threshold = logit(probability)"}, open(HERE + "final_params.json", "w"), indent=1)
    out = {"model": MODEL, "heldout_logloss": ll, "weights_all13": par, "baselines": {}, "points": {}}
    for k in ("split", "mutual", "plugin"):
        out["baselines"][k] = score.evaluate(score.baseline_links(k))["pooled"]
    out["baselines"]["mutual(relink.py, no cuts)"] = score.evaluate({s: {"links": relink.mutual_links(s)} for s in data.DEV_CLIPS})["pooled"]
    for p0, (held, ins) in res.items():
        point = {}
        for tag, sel in (("heldout", held), ("insample", ins)):
            cuts = loco.cuts_of(sel)
            point[tag] = {"cut_counts": counts(sel, n_sw), "per_clip_cuts": {s: len(cuts[s]) for s in data.DEV_CLIPS}}
            for name, m in variants(cuts).items():
                path = HERE + f"methods/{tag}_p{p0}_{name}.json"
                json.dump({"name": f"m_cuts {MODEL} p>={p0} {tag} {name}", "clips": m}, open(path, "w"))
                ev = score.evaluate(m)
                point[tag][name] = ev["pooled"]
                point[tag][name + "_per_clip"] = {s: {**v["returns"], "impure_rows": v["impure_rows"]} for s, v in ev["per_clip"].items()}
            sel.drop(columns=[c for c in sel.columns if c.startswith("_")]).to_csv(HERE + f"methods/{tag}_p{p0}_cuts.csv", index=False)
        out["points"][str(p0)] = point
    json.dump(out, open(HERE + "results.json", "w"), indent=1)

    # ------------------------------------------------------------------ table
    def row(label, r, cc=None):
        c = "" if cc is None else f"{cc['cuts']} | {cc['T']} | {cc['F']} | {cc['t']} | {cc['f']} | {cc['U']} | {cc['cut_precision']:.3f} | {cc['switches_hit']}/{cc['switches']}"
        if cc is None:
            c = " | | | | | | | "
        return (f"| {label} | {r['right']} | {r['wrong']} | {r['lost']} | {r['links_right']}/{r['links_wrong']}/{r['links_unknown']} | {100 * r['impure_row_share']:.2f}% | {c} |")

    head = ("| method | right | wrong | lost | links R/W/U | impure rows | cuts | T | F | t | f | U | cut precision | switches hit |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = ["## Baselines (no cuts)", "", head]
    for k, v in out["baselines"].items():
        lines.append(row(k, v))
    for tag, title in (("heldout", "## LOCO held-out (headline)"), ("insample", "## In-sample (weights fitted on all 13 clips; secondary)")):
        lines += ["", title, "", head]
        for p0 in P0S:
            pt = out["points"][str(p0)][tag]
            for name in ("split", "mutual", "mutual_rerun", "plugin"):
                lines.append(row(f"p>={p0} {name}+cuts", pt[name], pt["cut_counts"]))
    open(HERE + "results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("held-out log-loss", round(ll, 4), "weights", np.round(par["w"], 3).tolist(), "bias", round(par["b"], 3))


if __name__ == "__main__":
    main()
