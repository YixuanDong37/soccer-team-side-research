"""(8) ORACLE ceilings (labels decide the links: rule 5c, never a method result).

How many of the 137 scored returns can any method that only LINKS gbf pieces get right, and how
many more need a piece to be CUT first (the plugin's track moved onto another person)?

  O1  link pieces whose majority annotated person (>= 5 matched rows) is the same
  O2  the same with >= 1 matched row
  O3  first cut every piece where its matched person changes for >= 5 rows, then link the
      (sub)pieces by majority person (>= 1 matched row)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import common as C  # noqa: E402

score = C.score
pd.set_option("display.width", 250)
res: dict = {}

# (a) the pieces of the returns' rows A and B
rows = []
for seq in C.CLIPS:
    pi = C.piece_ident(seq).set_index("piece")
    for r in C.returns(seq).itertuples():
        if not r.scored:
            continue

        def status(p):
            rid = pi.at[p, "ref_id"]
            return "none (<5 matched rows)" if pd.isna(rid) else ("the person" if int(rid) == r.ref_id else "another person")

        rows.append({"seq": seq, "A": status(r.piece_a), "B": status(r.piece_b), "a_rows": int(pi.at[r.piece_a, "rows"]), "b_rows": int(pi.at[r.piece_b, "rows"]),
                     "a_purity": pi.at[r.piece_a, "purity"], "b_purity": pi.at[r.piece_b, "purity"], "same_plugin_id": int(pi.at[r.piece_a, "plugin"]) == int(pi.at[r.piece_b, "plugin"])})  # fmt: skip
RB = pd.DataFrame(rows)
ct = pd.crosstab(RB.A, RB.B, margins=True)
print("scored returns by the identity of the piece holding row A (rows) and row B (columns):")
print(ct.to_string())
ct.to_csv(C.OUT / "t8_return_piece_status.csv")
res["return_piece_status"] = {f"A={a} / B={b}": int(v) for (a, b), v in RB.groupby(["A", "B"]).size().items()}
print("rows of pieces that hold a return's row but mostly belong to another person:", C.qtable(pd.concat([RB.a_rows[RB.A == "another person"], RB.b_rows[RB.B == "another person"]])))
print("returns whose rows A and B have the same plugin id:", int(RB.same_plugin_id.sum()))


# (b) oracles
def majority(m, min_rows):
    ident = {}
    for pid, g in m[m.piece >= 0].groupby("piece"):
        top, n = Counter(g.ref_id.tolist()).most_common(1)[0]
        if n >= min_rows:
            ident[int(pid)] = int(top)
    return ident


def chain(ident, first):
    links = []
    by = {}
    for p, rid in ident.items():
        by.setdefault(rid, []).append(p)
    for ps in by.values():
        ps = sorted(ps, key=lambda p: first[p])
        links += [[ps[i], ps[i + 1]] for i in range(len(ps) - 1)]
    return links


methods = {"O1 link by majority person (>= 5 rows)": {}, "O2 link by majority person (>= 1 row)": {}, "O3 cut at identity changes, then link": {}}
n_cuts = 0
for seq in C.CLIPS:
    m = C.matched(seq)
    first = m[m.piece >= 0].groupby("piece").frame.min().to_dict()
    methods["O1 link by majority person (>= 5 rows)"][seq] = {"links": chain(majority(m, 5), first)}
    methods["O2 link by majority person (>= 1 row)"][seq] = {"links": chain(majority(m, 1), first)}
    cuts, ident3, first3 = [], {}, {}
    for pid, g in m[m.piece >= 0].sort_values("frame").groupby("piece"):
        refs, frs = g.ref_id.to_numpy(), g.frame.to_numpy()
        # runs of one person lasting >= 5 matched rows; shorter runs join the run before them
        runs = []
        s = 0
        for i in range(1, len(refs) + 1):
            if i == len(refs) or refs[i] != refs[s]:
                runs.append([s, i, int(refs[s])])
                s = i
        keep = []
        for a, b, rid in runs:
            if b - a >= 5 or not keep:
                if keep and keep[-1][2] == rid:
                    keep[-1][1] = b
                else:
                    keep.append([a, b, rid])
            else:
                keep[-1][1] = b
        # a short first run is absorbed by the next long one
        if len(keep) > 1 and keep[0][1] - keep[0][0] < 5:
            keep[1][0] = keep[0][0]
            keep = keep[1:]
        cut_frames = [int(frs[a]) for a, _, _ in keep[1:]]
        for j, f in enumerate(sorted(cut_frames)):
            cuts.append([int(pid), f])
        ids = [int(pid)] + [int(pid) + score.SUB * (j + 1) for j in range(len(cut_frames))]
        for sid, (a, b, rid) in zip(ids, keep):
            ident3[sid] = Counter(refs[a:b].tolist()).most_common(1)[0][0]
            first3[sid] = int(frs[a])
    n_cuts += len(cuts)
    methods["O3 cut at identity changes, then link"][seq] = {"links": chain(ident3, first3), "cuts": cuts}

tabs = []
for name, meth in methods.items():
    p = score.evaluate(meth)["pooled"]
    tabs.append({"oracle": name, "right": p["right"], "wrong": p["wrong"], "lost": p["lost"], "links": p["links"], "links_wrong": p["links_wrong"], "impure_row_share": p["impure_row_share"]})
for name in ("split", "plugin", "mutual"):
    p = score.evaluate(score.baseline_links(name))["pooled"]
    tabs.append({"oracle": f"(baseline) {name}", "right": p["right"], "wrong": p["wrong"], "lost": p["lost"], "links": p["links"], "links_wrong": p["links_wrong"], "impure_row_share": p["impure_row_share"]})
T = pd.DataFrame(tabs)
print("\noracle ceilings on the 137 scored returns (labels decide; NOT method results):")
print(T.to_string())
print("cuts made by O3:", n_cuts)
T.to_csv(C.OUT / "t8_oracle.csv", index=False)
res["oracles"] = T.to_dict("records")
res["o3_cuts"] = n_cuts
(C.OUT / "t8_oracle.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
