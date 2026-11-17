"""v_robust_1 / r8_impure.py -- (LABELS: scoring and diagnosis) impure_row_share against split, and
what the links add on top of the cuts.

    python -I -B r8_impure.py     -> out/r8_impure.md, out/r8_impure.json

impure rows (score.py): matched tracked rows whose group's majority person is someone else.
Reported per clip for split (gbf now), cuts only (fuse's cuts, no links), the fuse points, plugin
and mutual; plus three diagnosis numbers that score.py does not print:
  mixed groups     identities (groups) that hold >= 5 matched rows of each of two or more annotated people
  absorbed rows    impure rows of the method that were NOT impure under cuts only (added by the links)
  completeness     for every annotated player / goalkeeper, the share of his matched tracked rows that
                   lie in the one identity holding most of them (row-weighted mean): 1.0 = one identity
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

METHODS = {}
for b in ("split", "plugin", "mutual"):
    METHODS[b] = score.baseline_links(b)
rec = vc.load_method("cuts_only")
METHODS["cuts_only"] = {s: {"links": [], "cuts": rec[s]["cuts"]} for s in data.DEV_CLIPS}
for p in ("strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall"):
    METHODS[p] = vc.load_method(p)


def stats(seq, m):
    links, cuts = m.get("links") or [], m.get("cuts") or []
    c, ids, g = vl.assign(seq, links, cuts)
    tracked = ids >= 0
    df = pd.DataFrame({"g": g[tracked], "r": c.refs[tracked], "i": np.nonzero(tracked)[0]})
    major = df.groupby("g")["r"].agg(lambda s: s.value_counts().index[0])
    bad = df["r"].to_numpy() != major.loc[df["g"]].to_numpy()
    chk = score.score_clip(seq, links, cuts)
    assert int(bad.sum()) == chk["impure_rows"]
    cnt = df.groupby(["g", "r"]).size().reset_index(name="n")
    mixed = int((cnt[cnt.n >= 5].groupby("g").size() >= 2).sum())
    roles = vl.ref_attrs(seq)
    pl = cnt[cnt.r.map(lambda r: roles.get(int(r), {}).get("role") in ("player", "goalkeeper"))]
    best = pl.groupby("r").n.max()
    tot = pl.groupby("r").n.sum()
    return {"impure": int(bad.sum()), "rows": int(tracked.sum()), "impure_idx": set(df.i[bad].tolist()), "mixed_groups": mixed,
            "groups": int(df.g.nunique()), "best_rows": int(best.sum()), "person_rows": int(tot.sum()), "persons": int(len(tot)),
            "ids_per_person": float(pl.groupby("r").size().mean())}


R = {name: {s: stats(s, m[s]) for s in data.DEV_CLIPS} for name, m in METHODS.items()}
J = {}
L = ["# r8: impure rows against split (gbf now) and against cuts only", "",
     "## Pooled", "",
     "| method | impure rows | tracked matched rows | impure share | vs split | vs cuts only | rows made impure by the links | mixed groups | identities | completeness | identities per person |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for name, per in R.items():
    imp, rows = sum(v["impure"] for v in per.values()), sum(v["rows"] for v in per.values())
    sp = sum(v["impure"] for v in R["split"].values())
    co = sum(v["impure"] for v in R["cuts_only"].values())
    added = sum(len(per[s]["impure_idx"] - R["cuts_only"][s]["impure_idx"]) for s in per) if name not in ("split", "plugin", "mutual") else None
    comp = sum(v["best_rows"] for v in per.values()) / sum(v["person_rows"] for v in per.values())
    ipp = np.mean([v["ids_per_person"] for v in per.values()])
    J[name] = {"impure": imp, "rows": rows, "share": round(imp / rows, 5), "vs_split": imp - sp, "vs_cuts_only": imp - co, "added_by_links": added,
               "mixed_groups": sum(v["mixed_groups"] for v in per.values()), "groups": sum(v["groups"] for v in per.values()),
               "completeness": round(comp, 4), "ids_per_person_mean_of_clips": round(float(ipp), 2),
               "per_clip": {s: {k: v[k] for k in ("impure", "rows", "mixed_groups", "groups")} for s, v in per.items()}}
    L.append(f"| {name} | {imp} | {rows} | {100 * imp / rows:.2f}% | {imp - sp:+d} | {imp - co:+d} | {added if added is not None else '-'} | "
             f"{J[name]['mixed_groups']} | {J[name]['groups']} | {comp:.3f} | {ipp:.1f} |")
L += ["", "## Per clip: impure rows (share)", "", "| clip | rows | " + " | ".join(R) + " |", "|---|---|" + "---|" * len(R)]
for s in data.DEV_CLIPS:
    L.append(f"| {s} | {R['split'][s]['rows']} | " + " | ".join(f"{R[n][s]['impure']} ({100 * R[n][s]['impure'] / R[n][s]['rows']:.2f}%)" for n in R) + " |")
L += ["", "## Per clip: change of impure rows against split / against cuts only", "", "| clip | " + " | ".join(n for n in R if n not in ("split",)) + " |",
      "|---|" + "---|" * (len(R) - 1)]
for s in data.DEV_CLIPS:
    L.append(f"| {s} | " + " | ".join(f"{R[n][s]['impure'] - R['split'][s]['impure']:+d} / {R[n][s]['impure'] - R['cuts_only'][s]['impure']:+d}" for n in R if n != "split") + " |")
for p in ("strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall"):
    worse = [s for s in data.DEV_CLIPS if R[p][s]["impure"] > R["split"][s]["impure"]]
    better = [s for s in data.DEV_CLIPS if R[p][s]["impure"] < R["split"][s]["impure"]]
    d = {s: R[p][s]["impure"] - R["cuts_only"][s]["impure"] for s in data.DEV_CLIPS}
    tot = sum(d.values())
    top = sorted(d, key=d.get, reverse=True)[:2]
    J[p]["clips_worse_than_split"] = worse
    J[p]["clips_better_than_split"] = better
    J[p]["share_of_link_impurity_in_top2_clips"] = round(sum(d[s] for s in top) / tot, 3) if tot else None
    L += ["", f"- {p}: impure rows lower than split in {len(better)} clips, higher in {len(worse)} ({worse}); of the {tot} rows added over cuts only, "
              f"{100 * sum(d[s] for s in top) / tot:.0f}% are in {top}." if tot else f"- {p}: no rows added"]
# mixed groups per clip
L += ["", "## Per clip: mixed identities (>= 5 matched rows of two or more people)", "", "| clip | " + " | ".join(R) + " |", "|---|" + "---|" * len(R)]
for s in data.DEV_CLIPS:
    L.append(f"| {s} | " + " | ".join(str(R[n][s]["mixed_groups"]) for n in R) + " |")
(vc.OUT / "r8_impure.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(vc.OUT / "r8_impure.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
print("\n".join(L))
