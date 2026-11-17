"""Leave-one-clip-out evaluation of the links a cut suggests (cutlinks.py).

    python -I eval_cutlinks.py            (after make_methods.py: needs loco_params.json)

Cuts of a clip: features.cut_table with the weights fitted without that clip. Candidate links:
cutlinks.link_candidates. Two thresholds (margin, floor) are chosen per fold on the 12 fit clips:
the grid point with the most right links among those whose links are >= PREC right (right /
(right + wrong), score.py's link judgement); applied unchanged to the held-out clip.
Writes methods/heldout_p{p0}_cutlinks.json (cuts + these links) and
methods/heldout_p{p0}_mutual_rerun_cutlinks.json (cuts + mutual rule re-run + these links),
their in-sample twins (thresholds chosen on all 13), results_cutlinks.json / .md and
cutlink_params.json (the chosen thresholds; cutlinks.links_default reads them).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
from reid_lab import data, score  # noqa: E402
import features as F  # noqa: E402
import cutlinks  # noqa: E402
import relink  # noqa: E402

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
MARGINS = (0.0, 0.05, 0.1, 0.15, 0.2)
FLOORS = (0.0, 0.85, 0.9, 0.93, 0.95)
PREC = 0.95
P0S = (0.9, 0.7, 0.5)


def judged(seq, cut_df, cl):
    """Candidate links of a clip with score.py's judgement of each (right / wrong / unknown)."""
    c = cutlinks.link_candidates(seq, cut_df).drop_duplicates(["earlier", "later"]).reset_index(drop=True)
    verdict = []
    for r in c.itertuples():
        lk = score.score_clip(seq, [[int(r.earlier), int(r.later)]], cl)["links"]
        verdict.append("right" if lk["right"] else ("wrong" if lk["wrong"] else "unknown"))
    return c.assign(verdict=verdict)


def choose(c: pd.DataFrame):
    best, key = None, None
    for m in MARGINS:
        for f in FLOORS:
            s = c[(c.margin >= m) & (c.cos >= f)]
            r, w = int((s.verdict == "right").sum()), int((s.verdict == "wrong").sum())
            if r + w == 0 or r / (r + w) < PREC:
                continue
            k = (r, m + f)
            if key is None or k > key:
                best, key = (m, f), k
    return best


def main():
    out, params = {}, {}
    lines = ["| method | right | wrong | lost | links R/W/U | impure rows | thresholds (margin, floor) |", "|---|---|---|---|---|---|---|"]
    for p0 in P0S:
        for tag in ("heldout", "insample"):
            tabs, cuts = {}, {}
            for seq in data.DEV_CLIPS:
                par = F.heldout_params(seq, p0) if tag == "heldout" else p0
                ct = F.cut_table(seq, par)
                cuts[seq] = [[int(r.piece), int(r.f1)] for r in ct.itertuples()] if len(ct) else []
                tabs[seq] = judged(seq, ct, cuts[seq])
            allc = pd.concat(tabs.values(), ignore_index=True)
            chosen = {}
            if tag == "heldout":
                for fit, held in data.loco_folds():
                    chosen[held] = choose(allc[allc.seq.isin(fit)])
            else:
                chosen = {s: choose(allc) for s in data.DEV_CLIPS}
            links = {}
            for seq in data.DEV_CLIPS:
                c = tabs[seq]
                mf = chosen[seq]
                sel = c[(c.margin >= mf[0]) & (c.cos >= mf[1])] if mf is not None else c.iloc[:0]
                links[seq] = [[int(a), int(b)] for a, b in zip(sel.earlier, sel.later)]
            m1 = {s: {"links": links[s], "cuts": cuts[s]} for s in data.DEV_CLIPS}
            m2 = {s: {"links": [list(x) for x in sorted({tuple(l) for l in relink.mutual_links(s, cuts[s]) + links[s]})], "cuts": cuts[s]} for s in data.DEV_CLIPS}
            params.setdefault(str(p0), {})[tag] = {s: chosen[s] for s in data.DEV_CLIPS}
            thr = sorted(set(map(str, chosen.values())))
            for name, m in (("cutlinks", m1), ("mutual_rerun_cutlinks", m2)):
                json.dump({"name": f"m_cuts p>={p0} {tag} {name}", "clips": m}, open(HERE + f"methods/{tag}_p{p0}_{name}.json", "w"))
                r = score.evaluate(m)["pooled"]
                out[f"{tag}_p{p0}_{name}"] = {**r, "thresholds": {s: chosen[s] for s in data.DEV_CLIPS}}
                lines.append(f"| {tag} p>={p0} cuts+{name} | {r['right']} | {r['wrong']} | {r['lost']} | {r['links_right']}/{r['links_wrong']}/{r['links_unknown']} | {100 * r['impure_row_share']:.2f}% | {', '.join(thr)} |")
            allc.to_csv(HERE + f"methods/{tag}_p{p0}_cutlink_candidates.csv", index=False)
    json.dump(out, open(HERE + "results_cutlinks.json", "w"), indent=1)
    json.dump({"note": "cut-link thresholds [margin, floor] per cut probability: 'heldout' = chosen without that clip, 'insample' = chosen on all 13",
               "precision_target": PREC, **params}, open(HERE + "cutlink_params.json", "w"), indent=1)
    open(HERE + "results_cutlinks.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
