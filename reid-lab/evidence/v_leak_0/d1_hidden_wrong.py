"""v_leak DIAGNOSIS (labels; not a method result): what the scoring rule does not show.

score.py calls a return "wrong" only when the identity B was joined to had >= 5 matched rows of another
person before B, "right" as soon as A and B share a group (whatever else is in the group), and "lost"
otherwise. For a METHOD file this script re-reads every scored return and splits the three outcomes:

  right   clean            A and B share a group whose majority person is the returning person
          mixed            ... but the group also holds >= 5 matched rows of ANOTHER annotated person
          swallowed        ... and the group's majority person is someone else
          via-earlier      not the same group as A, but B's group before B is mostly the same person
  wrong                    (score.py's: mostly another person before B, >= 5 rows)
  lost    new              B's group has no matched row before B (an unlinked piece: harmless)
          hidden-wrong     B's group before B is mostly ANOTHER person, with 1-4 rows (below score.py's bar of 5):
                           B was joined to another person's rows, but the rule files it under "lost"
          new, mixed later no matched row before B, but B's group holds >= 5 rows of another person after B

    python -I -B d1_hidden_wrong.py   -> out/hidden_wrong.md, out/hidden_wrong.json
"""
import json
import sys
from collections import Counter

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402

HERE = "C:/Research/plugins/tools/reid_lab/work/v_leak_0/"
FUSE = "C:/Research/plugins/tools/reid_lab/work/fuse/"


def detail(seq, links, cuts):
    c = score.clip(seq)
    ids = score._ids_after_cuts(c, cuts)
    groups = score._Groups()
    for a, b in links or []:
        groups.union(int(a), int(b))
    tracked = ids >= 0
    alone = -1 - np.arange(len(ids), dtype=np.int64)
    g = np.where(tracked, np.array([groups.find(int(x)) if x >= 0 else -1 for x in ids], dtype=np.int64), alone)
    out = Counter()
    for r in c.returns:
        a, b = r["a"], r["b"]
        if a is None or b is None or not (tracked[a] and tracked[b]):
            continue
        if g[a] == g[b]:
            cnt = Counter(c.refs[g == g[b]].tolist())
            top = cnt.most_common(1)[0][0]
            other = max([n for k, n in cnt.items() if k != r["ref_id"]], default=0)
            if top != r["ref_id"]:
                out["right: swallowed (group majority is another person)"] += 1
            elif other >= score.MIN_MATCHED:
                out["right: mixed (group also holds >= 5 rows of another person)"] += 1
            else:
                out["right: clean"] += 1
            continue
        cnt = Counter(c.refs[(c.frames < c.frames[b]) & (g == g[b])].tolist())
        if not cnt:
            later = Counter(c.refs[g == g[b]].tolist())      # nothing before B; is B tied to another person LATER in the clip?
            other = max([n for k, n in later.items() if k != r["ref_id"]], default=0)
            out["lost: new identity, but its group holds >= 5 rows of another person later on" if other >= score.MIN_MATCHED else "lost: new identity"] += 1
            continue
        top, n = cnt.most_common(1)[0]
        if top == r["ref_id"]:
            out["right: via an earlier stretch of the same person"] += 1
        elif n >= score.MIN_MATCHED:
            out["wrong"] += 1
        else:
            # was B's own (sub-)piece alone already carrying those rows, or did a LINK bring them?
            own = Counter(c.refs[(c.frames < c.frames[b]) & (ids == ids[b])].tolist())
            if own and own.most_common(1)[0][0] != r["ref_id"]:
                out["lost: hidden wrong, 1-4 rows of another person in B's own piece (no link involved)"] += 1
            else:
                out["lost: hidden wrong, joined by a LINK to 1-4 rows of another person"] += 1
    return out


methods = {}
for name in ("split", "plugin", "mutual"):
    methods[f"baseline {name}"] = score.baseline_links(name)
for p in ("strict", "balanced", "recall", "margin_0.50"):
    methods[f"fuse {p} (held-out)"] = json.load(open(FUSE + f"METHOD_{p}.json", encoding="utf-8"))["clips"]
res = {}
for name, m in methods.items():
    tot, per = Counter(), {}
    for seq in data.DEV_CLIPS:
        d = detail(seq, (m.get(seq) or {}).get("links") or [], (m.get(seq) or {}).get("cuts") or [])
        per[seq] = dict(d)
        tot.update(d)
    res[name] = {"pooled": dict(tot), "per_clip": per}
json.dump(res, open(HERE + "out/hidden_wrong.json", "w"), indent=1)
keys = ["right: clean", "right: via an earlier stretch of the same person", "right: mixed (group also holds >= 5 rows of another person)",
        "right: swallowed (group majority is another person)", "wrong", "lost: hidden wrong, joined by a LINK to 1-4 rows of another person",
        "lost: hidden wrong, 1-4 rows of another person in B's own piece (no link involved)",
        "lost: new identity, but its group holds >= 5 rows of another person later on", "lost: new identity"]
L = ["# What score.py's three outcomes hide (diagnosis, 137 scored returns)", "", "| method | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
for name, r in res.items():
    L.append(f"| {name} | " + " | ".join(str(r["pooled"].get(k, 0)) for k in keys) + " |")
L += ["", "Per clip, fuse recall point: " + "; ".join(f"{s[-3:]}: mixed {d.get(keys[2], 0)}, swallowed {d.get(keys[3], 0)}, wrong {d.get('wrong', 0)}, hidden {d.get(keys[5], 0)}"
                                                    for s, d in res["fuse recall (held-out)"]["per_clip"].items())]
open(HERE + "out/hidden_wrong.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
