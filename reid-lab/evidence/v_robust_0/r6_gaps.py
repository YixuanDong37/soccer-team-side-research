"""v_robust_0 / r6_gaps.py -- (LABELS: diagnosis; scoring rule of reid_lab.score) behaviour by the
length of the absence, in particular long gaps (> 10 s = 250 frames).

    python -I -B r6_gaps.py     -> out/r6_gaps.md, out/r6_gaps.json

Returns: outcome per gap bucket for the fuse points and the baselines (annotated gap of the person).
Links: the links of each fuse point by the frames between the two pieces, with score.py's
judgement and the weak-label judgement (majority person of >= 1 matched row).
"""
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_0")
import numpy as np
import pandas as pd

import vr_common as vc
import vr_labels as vl
from vr_common import data

from reid_lab import score  # LABELS: baselines and the scorer's outcomes

BUCKETS = [("0.4-2 s (10-49 fr)", 10, 49), ("2-5 s (50-124)", 50, 124), ("5-10 s (125-250)", 125, 250), ("> 10 s (251+)", 251, 10**6)]
POINTS = ("strict", "balanced", "recall", "margin_0.50")


def bucket(g):
    for name, lo, hi in BUCKETS:
        if lo <= g <= hi:
            return name
    return "< 10"


def table(rt):
    out = {}
    for name, _, _ in BUCKETS:
        s = rt[rt.bucket == name]
        c = s.outcome.value_counts().to_dict()
        out[name] = {k: int(c.get(k, 0)) for k in ("right", "wrong", "lost", "undetected", "untracked")}
    return out


J = {"returns": {}, "links": {}}
L = ["# r6: behaviour by gap length (diagnosis with labels)", ""]
tabs = {}
for point in POINTS:
    rt = pd.read_csv(vc.OUT / f"returns_{point}.csv")
    rt["bucket"] = rt.gap.map(bucket)
    tabs[point] = rt
for b in ("plugin", "split", "mutual"):
    bl = score.baseline_links(b)
    rows = []
    for s in data.DEV_CLIPS:
        r = vl.returns_table(s, bl[s]["links"], [])
        assert [x["outcome"] for x in r] == score.score_clip(s, bl[s]["links"], [])["per_return"]
        rows += r
    rt = pd.DataFrame(rows)
    rt["bucket"] = rt.gap.map(bucket)
    tabs[b] = rt
n_all = len(tabs["recall"])
long_all = int((tabs["recall"].gap > 250).sum())
L += [f"Annotated returns: {n_all}; with a gap over 10 s: {long_all} ({100 * long_all / n_all:.0f}%). Gap quantiles (frames) 10/50/90%: "
      f"{np.quantile(tabs['recall'].gap, [0.1, 0.5, 0.9]).round(0).tolist()}; longest {int(tabs['recall'].gap.max())}.", "",
      "## Returns: right / wrong / lost (+ undetected) per gap bucket", "",
      "| method | " + " | ".join(n for n, _, _ in BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
for name, rt in tabs.items():
    t = table(rt)
    J["returns"][name] = t
    L.append(f"| {name} | " + " | ".join(f"{v['right']} / {v['wrong']} / {v['lost']} (+{v['undetected'] + v['untracked']})" for v in t.values()) + " |")
L += ["", "Right share and wrong share of the scored returns per bucket:", "", "| method | " + " | ".join(n for n, _, _ in BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
for name, t in J["returns"].items():
    cells = []
    for v in t.values():
        n = v["right"] + v["wrong"] + v["lost"]
        cells.append(f"{v['right'] / n:.2f} right, {v['wrong'] / n:.3f} wrong (n {n})" if n else "-")
    L.append(f"| {name} | " + " | ".join(cells) + " |")
# long-gap returns per clip at the recall point
rt = tabs["recall"]
lg = rt[rt.gap > 250]
L += ["", "Long-gap (> 10 s) returns per clip, recall point: " + ", ".join(
    f"{s} {({k: int(v) for k, v in g.outcome.value_counts().items()})}" for s, g in lg.groupby("seq")), ""]
L += ["## Links by the frames between the two pieces", "",
      "judged = score.py (both pieces >= 5 matched rows); weak = majority person of >= 1 matched row.", "",
      "| point | bucket | links | judged right / wrong / unknown | judged precision | weak agree / disagree / none | weak precision |", "|---|---|---|---|---|---|---|"]
LB = [("< 0.4 s (< 10 fr)", -10**6, 9), *BUCKETS]
for point in POINTS:
    lk = pd.read_csv(vc.OUT / f"links_{point}.csv")
    J["links"][point] = {}
    for name, lo, hi in LB:
        s = lk[(lk.gap >= lo) & (lk.gap <= hi)]
        j = s.judged.value_counts().to_dict()
        w = s.weak.value_counts().to_dict()
        jr, jw, wa, wd = j.get("right", 0), j.get("wrong", 0), w.get("agree", 0), w.get("disagree", 0)
        J["links"][point][name] = {"links": len(s), "judged_right": jr, "judged_wrong": jw, "judged_unknown": j.get("unknown", 0),
                                   "weak_agree": wa, "weak_disagree": wd, "weak_none": w.get("none", 0)}
        L.append(f"| {point} | {name} | {len(s)} | {jr} / {jw} / {j.get('unknown', 0)} | {jr / (jr + jw):.3f} |" if jr + jw else f"| {point} | {name} | {len(s)} | {jr} / {jw} / {j.get('unknown', 0)} | - |")
        L[-1] += f" {wa} / {wd} / {w.get('none', 0)} | " + (f"{wa / (wa + wd):.3f} |" if wa + wd else "- |")
(vc.OUT / "r6_gaps.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(vc.OUT / "r6_gaps.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
print("\n".join(L))
