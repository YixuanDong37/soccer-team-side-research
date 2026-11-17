"""v_robust_0 / r12_relaxed.py -- (LABELS: diagnosis; NOT the project's scoring rule) is the price
of the tiny-piece guards real, or does it come from where score.py takes row B?

    python -I -B r12_relaxed.py     -> out/r12_relaxed.md, out/r12_relaxed.json

score.py takes row A / row B as the person's matched box closest to the gap (within 12 frames).
When that box sits in a 1-4 box piece which a guard leaves unlinked, the return counts as lost
although the person may be re-recognised a few frames later in his first real piece.
Relaxed rule (diagnosis only): row A' = the person's last matched box before the gap, row B' = his
first matched box after it, each within 25 frames (1 s) of the gap and in a piece of at least
5 boxes; then score.py's rule unchanged. Returns without such rows are left out.
"""
import json
import sys
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_0")
import numpy as np
import pandas as pd

import vr_common as vc
import vr_labels as vl
from vr_common import data, prep

from reid_lab import score  # LABELS

K_ROWS, SLACK = 5, 25


def piece_counts(seq, cuts):
    ids = vc.piece_ids_after_cuts(seq, cuts)
    clean = data.clean_rows(seq)
    rows = pd.Series(ids[ids >= 0]).value_counts().to_dict()
    cl = pd.Series(ids[(ids >= 0) & clean]).value_counts().to_dict()
    return rows, cl


def relaxed(seq, links, cuts):
    c, ids, g = vl.assign(seq, links, cuts)
    rows, _ = piece_counts(seq, prep.norm_cuts(cuts))
    big = np.array([rows.get(int(x), 0) >= K_ROWS if x >= 0 else False for x in ids])
    out = []
    for r in c.returns:
        mine = np.nonzero((c.refs == r["ref_id"]) & big)[0]
        fa = c.frames[mine]
        a = mine[(fa <= r["last"]) & (fa >= r["last"] - SLACK)]
        b = mine[(fa >= r["first"]) & (fa <= r["first"] + SLACK)]
        if not len(a) or not len(b):
            out.append("no rows")
            continue
        a, b = a[np.argmax(c.frames[a])], b[np.argmin(c.frames[b])]
        if g[a] == g[b]:
            out.append("right")
            continue
        counts = Counter(c.refs[(c.frames < c.frames[b]) & (g == g[b])].tolist())
        if not counts:
            out.append("lost")
        else:
            top, n = counts.most_common(1)[0]
            out.append("right" if top == r["ref_id"] else ("wrong" if n >= score.MIN_MATCHED else "lost"))
    return out


def guard_links(lk, rows, cl, name):
    if name == "none":
        return lk
    ca, cb = lk.a.map(lambda p: cl.get(p, 0)).to_numpy(), lk.b.map(lambda p: cl.get(p, 0)).to_numpy()
    ra, rb = lk.a.map(rows).to_numpy(), lk.b.map(rows).to_numpy()
    if name == "clean >= 2":
        return lk[np.minimum(ca, cb) >= 2]
    if name == "rows >= 5":
        return lk[np.minimum(ra, rb) >= 5]
    both = set(lk.a) & set(lk.b)
    if name == "no bridge < 2 clean":
        small = {p for p in both if cl.get(p, 0) < 2}
    elif name == "no bridge < 5 rows":
        small = {p for p in both if rows[p] < 5}
    return lk[~(lk.a.isin(small) | lk.b.isin(small))]


J = {}
L = ["# r12: the tiny-piece guards under a relaxed choice of rows A and B (diagnosis, not the project's rule)", "",
     f"Relaxed: rows A' / B' = the person's nearest matched box within {SLACK} frames of the gap that lies in a piece of >= {K_ROWS} boxes.", "",
     "| method | guard | project rule: right / wrong / lost | relaxed rule: right / wrong / lost (returns with rows) | relaxed, without SNGS-162 |", "|---|---|---|---|---|"]
rows_out = []
for b in ("split", "plugin", "mutual"):
    bl = score.baseline_links(b)
    pr, rl, rl2 = Counter(), Counter(), Counter()
    for s in data.DEV_CLIPS:
        pr.update(score.score_clip(s, bl[s]["links"], [])["per_return"])
        x = relaxed(s, bl[s]["links"], [])
        rl.update(x)
        if s != "SNGS-162":
            rl2.update(x)
    J[b] = {"project": dict(pr), "relaxed": dict(rl)}
    L.append(f"| {b} | - | {pr['right']} / {pr['wrong']} / {pr['lost']} | {rl['right']} / {rl['wrong']} / {rl['lost']} ({rl['right'] + rl['wrong'] + rl['lost']}) | {rl2['right']} / {rl2['wrong']} / {rl2['lost']} |")
M = vc.load_method("recall")
pr, rl, rl2 = Counter(), Counter(), Counter()
for s in data.DEV_CLIPS:
    pr.update(score.score_clip(s, [], M[s]["cuts"])["per_return"])
    x = relaxed(s, [], M[s]["cuts"])
    rl.update(x)
    if s != "SNGS-162":
        rl2.update(x)
J["cuts_only"] = {"project": dict(pr), "relaxed": dict(rl)}
L.append(f"| cuts only | - | {pr['right']} / {pr['wrong']} / {pr['lost']} | {rl['right']} / {rl['wrong']} / {rl['lost']} ({rl['right'] + rl['wrong'] + rl['lost']}) | {rl2['right']} / {rl2['wrong']} / {rl2['lost']} |")
EXTRA = []
for point in ("strict", "margin_0.50", "balanced", "recall"):
    lk_all = pd.read_csv(vc.OUT / f"links_{point}.csv")
    M = vc.load_method(point)
    for s in data.DEV_CLIPS:      # returns that are wrong under the relaxed rule but not under the project's
        c = score.clip(s)
        a = score.score_clip(s, M[s]["links"], M[s]["cuts"])["per_return"]
        b = relaxed(s, M[s]["links"], M[s]["cuts"])
        for r, x, y in zip(c.returns, a, b):
            if y == "wrong" and x != "wrong":
                EXTRA.append(f"- {point} {s} person {r['ref_id']} (away {r['last'] + 1}-{r['first'] - 1}): project rule '{x}', relaxed 'wrong'")
            if x == "wrong" and y != "wrong":
                EXTRA.append(f"- {point} {s} person {r['ref_id']} (away {r['last'] + 1}-{r['first'] - 1}): project rule 'wrong', relaxed '{y}'")
    for guard in ("none", "no bridge < 2 clean", "no bridge < 5 rows", "clean >= 2", "rows >= 5"):
        pr, rl, rl2 = Counter(), Counter(), Counter()
        for s in data.DEV_CLIPS:
            cuts = prep.norm_cuts(M[s]["cuts"])
            rows, cl = piece_counts(s, cuts)
            lk = guard_links(lk_all[lk_all.seq == s], rows, cl, guard)
            links = lk[["a", "b"]].astype(int).values.tolist()
            pr.update(score.score_clip(s, links, M[s]["cuts"])["per_return"])
            x = relaxed(s, links, M[s]["cuts"])
            rl.update(x)
            if s != "SNGS-162":
                rl2.update(x)
        J[f"{point}|{guard}"] = {"project": dict(pr), "relaxed": dict(rl), "relaxed_without_162": dict(rl2)}
        L.append(f"| {point} | {guard} | {pr['right']} / {pr['wrong']} / {pr['lost']} | {rl['right']} / {rl['wrong']} / {rl['lost']} ({rl['right'] + rl['wrong'] + rl['lost']}) | "
                 f"{rl2['right']} / {rl2['wrong']} / {rl2['lost']} |")
L += ["", "Returns whose outcome changes to or from 'wrong' between the two rules (fuse without a guard):", ""] + EXTRA
(vc.OUT / "r12_relaxed.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(vc.OUT / "r12_relaxed.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
print("\n".join(L))
