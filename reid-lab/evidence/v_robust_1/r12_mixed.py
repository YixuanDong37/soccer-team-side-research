"""v_robust_1 / r12_mixed.py -- (LABELS: diagnosis) returns that score.py counts right although the
identity they end up in is not clean, and what the leaf links are.

    python -I -B r12_mixed.py     -> out/r12_mixed.md, out/r12_mixed.json

score.py: a return is right when rows A and B share an identity. It does not ask whether that
identity also holds other people. Here, for every return scored right:
  mixed        the identity holds >= 5 matched boxes of another annotated person;
  minority     the returning person is not even the majority person of the identity.
Reported for the fuse_r1 points and the baselines, so that "right" can be read as
"right and clean". Second part: the leaf links (stage 2) of the recall point by weak labels.
"""
import json
import sys
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import numpy as np
import pandas as pd

import vr_common as vc
import vr_labels as vl
from vr_common import data

from reid_lab import score  # LABELS

METHODS = {b: score.baseline_links(b) for b in ("split", "plugin", "mutual")}
for p in ("cuts_only", "strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall"):
    METHODS[p] = vc.load_method(p)

J = {}
L = ["# r12: returns scored right inside an identity that is not clean (diagnosis with labels)", "",
     "| method | right | of which mixed (>= 5 boxes of another person in the identity) | of which the person is not the identity's majority | wrong | wrong + mixed | "
     "right and clean | clean precision = clean right / (right + wrong) |", "|---|---|---|---|---|---|---|---|"]
detail = {}
for name, M in METHODS.items():
    c = Counter()
    rows = []
    for seq in data.DEV_CLIPS:
        m = M.get(seq) or {}
        clip, ids, g = vl.assign(seq, m.get("links") or [], m.get("cuts") or [])
        outs = score.score_clip(seq, m.get("links") or [], m.get("cuts") or [])["per_return"]
        for r, o in zip(clip.returns, outs):
            if o == "wrong":
                c["wrong"] += 1
            if o != "right":
                continue
            c["right"] += 1
            cnt = Counter(clip.refs[g == g[r["b"]]].tolist())
            others = {k: v for k, v in cnt.items() if k != r["ref_id"] and v >= 5}
            top = cnt.most_common(1)[0][0]
            if others:
                c["mixed"] += 1
                rows.append(f"{seq} person {r['ref_id']} ({cnt[r['ref_id']]} boxes) with {others}" + (" [not the majority]" if top != r["ref_id"] else ""))
            if top != r["ref_id"]:
                c["minority"] += 1
    detail[name] = rows
    clean = c["right"] - c["mixed"]
    J[name] = {"right": c["right"], "mixed": c["mixed"], "minority": c["minority"], "wrong": c["wrong"], "wrong_plus_mixed": c["wrong"] + c["mixed"],
               "clean_right": clean, "clean_precision": round(clean / (c["right"] + c["wrong"]), 4) if c["right"] + c["wrong"] else None, "cases": rows}
    L.append(f"| {name} | {c['right']} | {c['mixed']} | {c['minority']} | {c['wrong']} | {c['wrong'] + c['mixed']} | {clean} | "
             + (f"{clean / (c['right'] + c['wrong']):.3f} |" if c["right"] + c["wrong"] else "- |"))
L.append("")
for name in ("strict", "balanced", "recall", "fixed_strict"):
    L += [f"Mixed right returns, {name}:"] + [f"- {x}" for x in detail[name]] + [""]

# ---------------------------------------------------------------- leaf links
L += ["## Leaf links (stage 2) by weak labels (majority person of >= 1 matched box)", "",
      "| point | leaf links | agree | disagree | no matched box on one side | disagree share of the judgeable | disagree outside SNGS-162 | leaf rows 1 / 2 / 3 / 4 boxes: disagree share |",
      "|---|---|---|---|---|---|---|---|"]
for point in ("recall", "fixed_recall"):
    lk = pd.read_csv(vc.OUT / f"links_{point}.csv")
    lf = lk[lk.stage == 2]
    w = lf.weak.value_counts().to_dict()
    a, d, n = w.get("agree", 0), w.get("disagree", 0), w.get("none", 0)
    short = np.minimum(lf.rows_a, lf.rows_b)
    by = []
    for k in (1, 2, 3, 4):
        s = lf[short == k]
        aa, dd = int((s.weak == "agree").sum()), int((s.weak == "disagree").sum())
        by.append(f"{dd}/{aa + dd}")
    J[f"leaf_{point}"] = {"links": len(lf), "agree": a, "disagree": d, "none": n, "disagree_outside_162": int(((lf.weak == "disagree") & (lf.seq != "SNGS-162")).sum())}
    L.append(f"| {point} | {len(lf)} | {a} | {d} | {n} | {d / (a + d):.3f} | {J[f'leaf_{point}']['disagree_outside_162']} | {', '.join(by)} |")
L.append("")
for point in ("strict", "balanced", "recall"):
    lk = pd.read_csv(vc.OUT / f"links_{point}.csv")
    co = lk[lk.stage == 1]
    w = co.weak.value_counts().to_dict()
    per = co[co.weak == "disagree"].seq.value_counts().to_dict()
    L.append(f"- core links, {point}: {len(co)}; weak agree {w.get('agree', 0)}, disagree {w.get('disagree', 0)} ({per}), none {w.get('none', 0)}; "
             f"weak precision {w.get('agree', 0) / (w.get('agree', 0) + w.get('disagree', 0)):.4f}; without SNGS-162 "
             f"{int(((co.weak == 'agree') & (co.seq != 'SNGS-162')).sum())} / {int(((co.weak == 'disagree') & (co.seq != 'SNGS-162')).sum())}")
(vc.OUT / "r12_mixed.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(vc.OUT / "r12_mixed.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
print("\n".join(L))
