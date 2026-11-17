"""Leave-one-clip-out evaluation of the cut detector (labels are used only to fit the logistic
regression on 12 clips and to score the held-out clip).

    python -I loco.py                       # model comparison: held-out pooled + in-sample
    python -I loco.py "lr(cos,mm,lgap,ld)"  # one model

Per fold: logistic regression on the first-level candidates of the 12 fit clips that have a firm
label (truth.py: T / F; weak and unknown ones left out), then features.cut_table on the held-out
clip with those weights; a candidate is cut where the fitted probability of "different people" is
>= p0 (p0 is not fitted: 0.9 / 0.7 / 0.5 / 0.3 are the operating points).
In-sample: the regression fitted on all 13 clips, applied to all 13.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
from reid_lab import data, score  # noqa: E402
import features as F  # noqa: E402
import truth as T  # noqa: E402

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
L2 = 1e-3
P0S = (0.9, 0.7, 0.5, 0.3)

MODELS = {
    "lr(cos)": ["cos"],
    "lr(mm)": ["mm"],
    "lr(cos,mm)": ["cos", "mm"],
    "lr(cos,lgap)": ["cos", "lgap"],
    "lr(cos,mm,lgap)": ["cos", "mm", "lgap"],
    "lr(cos,mm,lgap,ld)": ["cos", "mm", "lgap", "ld"],
    "lr(cos,lgap,ld)": ["cos", "lgap", "ld"],
    "lr(cos,mm,lgap,ld,side)": ["cos", "mm", "lgap", "ld", "side"],
    "lr(cos,md,mr,lgap,ld)": ["cos", "md", "mr", "lgap", "ld"],
}


def table() -> pd.DataFrame:
    """First-level candidates of all dev clips with their labels (truth.label)."""
    frames = []
    for seq in data.DEV_CLIPS:
        c = F.candidates(seq)
        lab, which = T.label(seq, c)
        frames.append(c.assign(label=lab, sw_id=which))
    return pd.concat(frames, ignore_index=True)


def fit_lr(cols: list[str], tab: pd.DataFrame) -> dict:
    known = tab[tab.label.isin(["T", "F"])]
    X = F.design(known)[cols].to_numpy(float)
    y = (known.label == "T").to_numpy(float)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mu) / sd

    def loss(w):
        z = Z @ w[:-1] + w[-1]
        return np.mean(np.logaddexp(0, z) - y * z) + L2 * (w[:-1] ** 2).sum()

    res = minimize(loss, np.zeros(X.shape[1] + 1), method="L-BFGS-B")
    return {"cols": cols, "mu": mu.tolist(), "sd": sd.tolist(), "w": res.x[:-1].tolist(), "b": float(res.x[-1])}


def logit(p0: float) -> float:
    return float(np.log(p0 / (1 - p0)))


def labelled_cuts(seq: str, par: dict) -> pd.DataFrame:
    ct = F.cut_table(seq, par)
    if not len(ct):
        return pd.DataFrame({"seq": [], "piece": [], "f0": [], "f1": [], "label": [], "sw_id": []})
    lab, which = T.label(seq, ct)
    return ct.assign(label=lab, sw_id=which)


def cuts_of(sel: pd.DataFrame) -> dict[str, list[list[int]]]:
    out = {seq: [] for seq in data.DEV_CLIPS}
    for r in sel.itertuples():
        out[r.seq].append([int(r.piece), int(r.f1)])
    return out


def method(cuts: dict, base: str | None, how: str = "appearance") -> dict:
    """{seq: {"links", "cuts"}} for score.evaluate: no links (base None) or a baseline's links
    re-mapped onto the sub-pieces."""
    if base is None:
        return {s: {"links": [], "cuts": cuts[s]} for s in data.DEV_CLIPS}
    links = score.baseline_links(base)
    return {s: {"links": F.remap_links(s, links[s]["links"], cuts[s], how), "cuts": cuts[s]} for s in data.DEV_CLIPS}


def evaluate_cuts(sel: pd.DataFrame, n_switches: int) -> dict:
    """Cuts (with labels) -> counts and the lab's scores on top of split, mutual and plugin."""
    cuts = cuts_of(sel)
    res = {k: int((sel.label == k).sum()) for k in ("T", "F", "t", "f", "U")}
    res = {"cuts": len(sel), **res, "switches_hit": int(sel[sel.sw_id >= 0].groupby("seq").sw_id.nunique().sum()), "switches": n_switches}
    res["cut_precision"] = round(res["T"] / max(1, res["T"] + res["F"]), 3)
    res["cut_precision_weak"] = round((res["T"] + res["t"]) / max(1, res["T"] + res["F"] + res["t"] + res["f"]), 3)
    res["split+cuts"] = score.evaluate(method(cuts, None))["pooled"]
    for how in ("appearance", "adjacent", "drop"):
        res[f"mutual+cuts({how})"] = score.evaluate(method(cuts, "mutual", how))["pooled"]
    res["plugin+cuts(adjacent)"] = score.evaluate(method(cuts, "plugin", "adjacent"))["pooled"]
    return res


def loco(cols: list[str], tab: pd.DataFrame, p0s=P0S):
    """-> {p0: (held-out cuts, in-sample cuts)}, fold parameters, all-13 parameters, held-out
    log-loss of the regression on the firm candidates."""
    pars, held = {}, {p0: [] for p0 in p0s}
    ll, n = 0.0, 0
    for fit_clips, out_clip in data.loco_folds():
        par = fit_lr(cols, tab[tab.seq.isin(fit_clips)])
        pars[out_clip] = par
        test = tab[(tab.seq == out_clip) & tab.label.isin(["T", "F"])]
        z = F.apply_model(par, test)
        y = (test.label == "T").to_numpy(float)
        ll += float(np.sum(np.logaddexp(0, z) - y * z))
        n += len(test)
        for p0 in p0s:
            held[p0].append(labelled_cuts(out_clip, {**par, "threshold": logit(p0)}))
    par = fit_lr(cols, tab)
    out = {}
    for p0 in p0s:
        ins = pd.concat([labelled_cuts(s, {**par, "threshold": logit(p0)}) for s in data.DEV_CLIPS], ignore_index=True)
        out[p0] = (pd.concat(held[p0], ignore_index=True), ins)
    return out, pars, par, ll / max(1, n)


def short(r: dict) -> str:
    a, b, c, d = r["split+cuts"], r["mutual+cuts(appearance)"], r["mutual+cuts(adjacent)"], r["mutual+cuts(drop)"]
    return (f"cuts {r['cuts']:3d} T {r['T']:3d} F {r['F']:3d} t {r['t']:3d} f {r['f']:3d} U {r['U']:2d} prec {r['cut_precision']:.3f} sw_hit {r['switches_hit']:2d}/{r['switches']} | "
            f"split+cuts R/W/L {a['right']}/{a['wrong']}/{a['lost']} impure {a['impure_row_share']:.4f} | "
            f"mutual+cuts(app) {b['right']}/{b['wrong']}/{b['lost']} links {b['links_right']}/{b['links_wrong']}/{b['links_unknown']} impure {b['impure_row_share']:.4f} | "
            f"(adj) {c['right']}/{c['wrong']}/{c['lost']} links {c['links_right']}/{c['links_wrong']}/{c['links_unknown']} | "
            f"(drop) {d['right']}/{d['wrong']}/{d['lost']} links {d['links_right']}/{d['links_wrong']}/{d['links_unknown']}")


if __name__ == "__main__":
    tab = table()
    n_sw = sum(len(T.switches(s)) for s in data.DEV_CLIPS)
    print("candidates", len(tab), tab.label.value_counts().to_dict(), "true switches", n_sw)
    base = {k: score.evaluate(score.baseline_links(k))["pooled"] for k in ("split", "mutual", "plugin")}
    for k, v in base.items():
        print(f"baseline {k}: R/W/L {v['right']}/{v['wrong']}/{v['lost']} links {v['links_right']}/{v['links_wrong']}/{v['links_unknown']} impure {v['impure_row_share']:.4f}")
    out = {}
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(MODELS)
    for name in names:
        res, pars, par, ll = loco(MODELS[name], tab)
        print(f"{name}: held-out log-loss {ll:.4f}; all-13 weights (standardised) {np.round(par['w'], 2).tolist()} bias {par['b']:.2f}")
        for p0, (held, ins) in res.items():
            rh, ri = evaluate_cuts(held, n_sw), evaluate_cuts(ins, n_sw)
            out[f"{name}@{p0}"] = {"heldout": rh, "insample": ri, "heldout_logloss": ll, "params_all13": par}
            print(f" p>={p0}\n   held-out : {short(rh)}\n   in-sample: {short(ri)}")
    json.dump(out, open(HERE + "loco_models.json", "w"), indent=1)
