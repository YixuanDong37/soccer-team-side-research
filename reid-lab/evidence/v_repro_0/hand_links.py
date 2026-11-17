"""v_repro / hand_links.py -- INDEPENDENT recomputation of score.py's link verdicts and impure rows
(LABELS; a check of the scorer). Uses only hand_check.py's own matching (no reid_lab, no gbf).

    python -I -B hand_links.py METHOD.json OUT_of_score_cli.json

Per clip: links right / wrong / unknown (a link is judged when both sub-pieces have a majority
person with >= 5 matched rows), impure rows (matched tracked rows whose group's majority person is
someone else) and tracked matched rows; compared with the per_clip block of score.py's CLI output.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hand_check as hc  # noqa: E402

method = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["clips"]
cli = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
bad = 0
tot = Counter()
ties = 0
for seq in hc.DEV:
    mm = method.get(seq) or {}
    links, cuts = mm.get("links") or [], mm.get("cuts") or []
    people, W, H = hc.read_labels(seq)
    tracks = pd.read_parquet(hc.RUNS / seq / "gbf/refine/tracks.parquet").reset_index(drop=True)
    m = hc.match_clip(people, tracks)
    ident = m.piece.to_numpy().copy()
    by = {}
    for p, f in cuts:
        by.setdefault(int(p), []).append(int(f))
    for p, fs in by.items():
        for j, f in enumerate(sorted(fs)):
            ident[(m.piece.to_numpy() == p) & (m.frame.to_numpy() >= f)] = p + hc.SUB * (j + 1)
    who = {}
    for tid in np.unique(ident[ident >= 0]):
        c = Counter(m.ref.to_numpy()[ident == tid].tolist())
        best = max(c.values())
        tops = [k for k, v in c.items() if v == best]
        ties += len(tops) > 1 and best >= 5
        if best >= 5:
            who[int(tid)] = tops[0]
    lk = Counter()
    for a, b in links:
        a, b = int(a), int(b)
        lk["unknown" if a not in who or b not in who else ("right" if who[a] == who[b] else "wrong")] += 1
    G = hc.Groups()
    for a, b in links:
        G.union(int(a), int(b))
    tracked = ident >= 0
    grp = np.array([G.find(int(x)) for x in ident[tracked]])
    refs = m.ref.to_numpy()[tracked]
    impure = 0
    for g in np.unique(grp):
        c = Counter(refs[grp == g].tolist())
        best = max(c.values())
        ties += sum(v == best for v in c.values()) > 1
        impure += int((grp == g).sum()) - best
    mine = {"right": lk["right"], "wrong": lk["wrong"], "unknown": lk["unknown"], "impure": impure}
    ref = cli["per_clip"][seq]
    theirs = {"right": ref["links"]["right"], "wrong": ref["links"]["wrong"], "unknown": ref["links"]["unknown"], "impure": ref["impure_rows"]}
    same = mine == theirs
    bad += not same
    tot.update(mine)
    tot["rows"] += int(tracked.sum())
    print(seq, mine, "tracked matched rows", int(tracked.sum()), "| same as score.py" if same else f"| DIFFERENT: score.py {theirs}", flush=True)
share = round(tot["impure"] / tot["rows"], 5)
p = cli["pooled"]
print("pooled: links", tot["right"], "/", tot["wrong"], "/", tot["unknown"], "| impure share", share,
      "| score.py:", p["links_right"], "/", p["links_wrong"], "/", p["links_unknown"], "|", p["impure_row_share"])
print("majority ties met:", ties)
print("all clips agree" if not bad and share == p["impure_row_share"] else "DIFFERENCES")
