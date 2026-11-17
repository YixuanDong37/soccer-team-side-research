"""DIAGNOSIS after the LOCO run (labels; none of these numbers is a method result).
(a) the links score.py cannot judge ("unknown": a piece with < 5 matched rows), judged by the majority
    person of however few matched rows;
(b) the wrong returns of each operating point, by clip and annotated person;
(c) POST HOC sensitivity of the choice rule: the same J, but leaving out the fit clip with the most
    (wrong returns + wrong links) at each margin; and the fixed-margin rows without SNGS-162;
(d) what the lost returns of the recall point look like."""
import json
import pickle
import sys
from collections import Counter

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data, score  # noqa: E402
import fit_loco  # noqa: E402
import fuse_labels as fitlabels  # noqa: E402
import prep  # noqa: E402

HERE = "C:/Research/plugins/tools/reid_lab/work/fuse/"
raw = pickle.load(open(HERE + "cache/loco_raw.pkl", "rb"))["folds"]
rep = json.load(open(HERE + "out/loco_report.json", encoding="utf-8"))
P = rep["variants"]["fuse"]["points"]

print("(a) unknown links, weak labels (majority person of >= 1 matched row)")
for point in ("strict", "balanced", "recall"):
    m = json.load(open(HERE + f"METHOD_{point}.json", encoding="utf-8"))["clips"]
    c = Counter()
    for seq, e in m.items():
        cuts = prep.norm_cuts(e["cuts"])
        ident = fitlabels.identities(seq, cuts)
        for a, b in e["links"]:
            ia, ib = ident.get(a), ident.get(b)
            firm = ia is not None and ib is not None and ia[1] >= 5 and ib[1] >= 5
            if firm:
                c["judged by score.py: " + ("right" if ia[0] == ib[0] else "wrong")] += 1
            elif ia is None or ib is None:
                c["unknown: a piece has no matched row"] += 1
            else:
                c["unknown: weak " + ("agree" if ia[0] == ib[0] else "disagree")] += 1
    print(f"  {point}: {dict(sorted(c.items()))}")

print("\n(b) wrong returns (held-out), clip / annotated person / last frame before the gap / first frame after")
for point in ("strict", "balanced", "recall"):
    m = json.load(open(HERE + f"METHOD_{point}.json", encoding="utf-8"))["clips"]
    out = []
    for seq, e in m.items():
        r = score.score_clip(seq, e["links"], e["cuts"])
        for ret, o in zip(score.clip(seq).returns, r["per_return"]):
            if o == "wrong":
                out.append(f"{seq[-3:]} id {ret['ref_id']} {ret['last']}>{ret['first']}")
    print(f"  {point}: {len(out)}: " + "; ".join(out))

print("\n(c1) POST HOC: J without the fit clip that has the most (wrong + wrong links) at that margin")
for point, lam in fit_loco.LAMBDA.items():
    chosen = {}
    for c in data.DEV_CLIPS:
        fit = [s for s in data.DEV_CLIPS if s != c]
        best = None
        for mg in fit_loco.MARGINS:
            per = []
            for s in fit:
                x = raw[c]["res"][("fuse", mg, s)]
                cnt = Counter(x["per_return"])
                per.append((cnt["wrong"] + x["lk"]["wrong"], cnt["right"]))
            per.sort()
            per = per[:-1]
            j = sum(r for _, r in per) - lam * sum(w for w, _ in per)
            if best is None or j >= best[0]:
                best = (j, mg)
        chosen[c] = best[1]
    held = fit_loco.pool(raw[c]["res"][("fuse", chosen[c], c)] for c in data.DEV_CLIPS)
    print(f"  {point} (lam {lam:g}): {fit_loco.fmt(held)}; margins {dict(Counter(chosen.values()))}")

print("\n(c2) fixed margin, held-out, 12 clips without SNGS-162 (119 scored returns) and SNGS-162 alone (18)")
for mg in fit_loco.MARGINS:
    a = fit_loco.pool(raw[c]["res"][("fuse", mg, c)] for c in data.DEV_CLIPS if c != "SNGS-162")
    b = fit_loco.pool([raw["SNGS-162"]["res"][("fuse", mg, "SNGS-162")]])
    print(f"  margin {mg:.2f}: without 162: {a['right']} / {a['wrong']} / {a['lost']} links {a['links_right']} / {a['links_wrong']} / {a['links_unknown']}"
          f"   | 162: {b['right']} / {b['wrong']} / {b['lost']} links {b['links_right']} / {b['links_wrong']} / {b['links_unknown']}")
for point in ("strict", "balanced", "recall"):
    pc = P[point]["heldout_per_clip"]
    a = Counter()
    for c, x in pc.items():
        if c != "SNGS-162":
            a.update({k: x.get(k, 0) for k in ("right", "wrong", "lost")})
    print(f"  point {point} without SNGS-162: {a['right']} / {a['wrong']} / {a['lost']}")

print("\n(d) lost returns of the recall point: matched rows of the pieces holding row A and row B")
m = json.load(open(HERE + "METHOD_recall.json", encoding="utf-8"))["clips"]
kinds = Counter()
for seq, e in m.items():
    cl = score.clip(seq)
    cuts = prep.norm_cuts(e["cuts"])
    ids = score._ids_after_cuts(cl, e["cuts"])
    ident = fitlabels.identities(seq, cuts)
    r = score.score_clip(seq, e["links"], e["cuts"])
    for ret, o in zip(cl.returns, r["per_return"]):
        if o != "lost":
            continue
        pa, pb = int(ids[ret["a"]]), int(ids[ret["b"]])
        ka = "own" if ident[pa][0] == ret["ref_id"] and ident[pa][1] >= 5 else ("short" if ident[pa][2] < 5 else "other person's")
        kb = "own" if ident[pb][0] == ret["ref_id"] and ident[pb][1] >= 5 else ("short" if ident[pb][2] < 5 else "other person's")
        kinds[f"A in {ka} piece, B in {kb} piece"] += 1
for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
    print(f"  {v:3d}  {k}")
