"""v_leak: how much of the held-out result rests on design choices that were made with all 13 clips in view?
(1) NESTED choice of the variant. fuse's README says the plugin-join bonus "was kept because it adds returns
    at strict margins (0.8: 51 against 43)" -- those are pooled HELD-OUT numbers. Here the variant is chosen
    inside each fold, on the 12 fit clips, by the same J as the margin (ties: fuse first, then larger margin).
(2) How the chosen margin moves across folds, and the fit-clip (in-sample) rate against the held-out rate.
(3) Selection optimism of picking a fixed-margin row afterwards: the best row under a wrong-limit, against
    the same pick made inside each fold (the "cap" rule) and against the pre-fixed J rule.

Reads fuse/cache/loco_raw.pkl (fuse's own LOCO run; its held-out links for the variant fuse were reproduced
blindfolded, see s4_compare.py). Labels enter only through the stored score.py outcomes.

    python -I -B n1_nested_variants.py   -> out/nested_variants.md
"""
import pickle
import sys
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data  # noqa: E402

HERE = "C:/Research/plugins/tools/reid_lab/work/v_leak_0/"
raw = pickle.load(open("C:/Research/plugins/tools/reid_lab/work/fuse/cache/loco_raw.pkl", "rb"))["folds"]
MARG = {"fuse": (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)}
MARG["fuse_pos"] = MARG["fuse_nocut"] = MARG["fuse"]
MARG["app_cut"] = MARG["app_nocut"] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
LAMBDA = {"strict": 10.0, "balanced": 3.0, "recall": 1.0}
CLIPS = list(data.DEV_CLIPS)


def pool(items):
    c, lk, imp, rows = Counter(), Counter(), 0, 0
    for x in items:
        c.update(x["per_return"])
        lk.update({k: v for k, v in x["lk"].items() if k != "n"})
        imp += x["impure"]
        rows += x["rows"]
    return {"right": c["right"], "wrong": c["wrong"], "lost": c["lost"], "links_right": lk["right"], "links_wrong": lk["wrong"], "links_unknown": lk["unknown"], "impure": imp / rows}


def fmt(p):
    return f"{p['right']} / {p['wrong']} / {p['lost']} | {p['links_right']} / {p['links_wrong']} / {p['links_unknown']} | {100 * p['impure']:.2f}%"


def choose(fold, variants, lam, clips):
    best = None
    for v in variants:                      # earlier variant wins ties
        vb = None
        for m in MARG[v]:                   # larger margin wins ties (fuse's rule)
            p = pool(raw[fold]["res"][(v, m, s)] for s in clips)
            j = p["right"] - lam * (p["wrong"] + p["links_wrong"])
            if vb is None or j >= vb[0]:
                vb = (j, v, m)
        if best is None or vb[0] > best[0]:
            best = vb
    return best


L = ["# Nested design choices (LOCO held-out, pooled; right / wrong / lost | links right / wrong / unknown | impure rows)", ""]
L += ["## 1. Variant chosen inside each fold (on the 12 fit clips, same J as the margin)", "",
      "| candidates | point | HELD-OUT | variant chosen per fold | margins |", "|---|---|---|---|---|"]
for cand in (["fuse"], ["fuse", "fuse_pos"], ["fuse", "fuse_pos", "app_cut"], ["fuse", "fuse_pos", "app_cut", "fuse_nocut", "app_nocut"]):
    for point, lam in LAMBDA.items():
        ch = {c: choose(c, cand, lam, [s for s in CLIPS if s != c]) for c in CLIPS}
        held = pool(raw[c]["res"][(ch[c][1], ch[c][2], c)] for c in CLIPS)
        L.append(f"| {' / '.join(cand)} | {point} | {fmt(held)} | {dict(Counter(x[1] for x in ch.values()))} | {dict(sorted(Counter(x[2] for x in ch.values()).items()))} |")
L += ["", "## 2. Fit-clip rate against held-out rate at the margin chosen in the fold (variant fuse)", "",
      "| point | fold | margin | fit clips: right share, wrong share | held-out clip: right / wrong / lost |", "|---|---|---|---|---|"]
for point, lam in LAMBDA.items():
    fr = fw = fn = hr = hw = hn = 0
    for c in CLIPS:
        fit = [s for s in CLIPS if s != c]
        _, v, m = choose(c, ["fuse"], lam, fit)
        p = pool(raw[c]["res"][(v, m, s)] for s in fit)
        h = pool([raw[c]["res"][(v, m, c)]])
        n = p["right"] + p["wrong"] + p["lost"]
        L.append(f"| {point} | {c} | {m} | {p['right'] / n:.3f}, {p['wrong'] / n:.3f} | {h['right']} / {h['wrong']} / {h['lost']} |")
        fr, fw, fn = fr + p["right"], fw + p["wrong"], fn + n
        hr, hw, hn = hr + h["right"], hw + h["wrong"], hn + h["right"] + h["wrong"] + h["lost"]
    L.append(f"| {point} | **all folds** | | **{fr / fn:.3f}, {fw / fn:.3f}** (mean over folds of the fit-clip shares) | **{hr / hn:.3f}, {hw / hn:.3f}** ({hr} / {hw} of {hn}) |")
L += ["", "## 3. Picking a fixed-margin row afterwards (variant fuse)", "",
      "Oracle pick = the fixed margin with the most right returns among those with wrong <= cap on the POOLED HELD-OUT table;",
      "nested pick = the same criterion applied inside each fold to its 12 fit clips (cap scaled by the fit clips' scored returns), then applied to the held-out clip.", "",
      "| cap on wrong | oracle pick on the held-out table | nested pick (per fold) | margins chosen per fold | pre-fixed J rule (fuse's headline) |", "|---|---|---|---|---|"]
head = {4: "strict", 7: "balanced", 14: "recall"}
for cap in (4, 7, 14):
    rows = [(m, pool(raw[c]["res"][("fuse", m, c)] for c in CLIPS)) for m in MARG["fuse"]]
    ok = [r for r in rows if r[1]["wrong"] <= cap]
    om, op = max(ok, key=lambda r: (r[1]["right"], r[0]))
    chosen = {}
    for c in CLIPS:
        fit = [s for s in CLIPS if s != c]
        fr = [(m, pool(raw[c]["res"][("fuse", m, s)] for s in fit)) for m in MARG["fuse"]]
        budget = int(cap * (fr[0][1]["right"] + fr[0][1]["wrong"] + fr[0][1]["lost"]) / 137.0)
        okf = [r for r in fr if r[1]["wrong"] <= budget]
        chosen[c] = max(okf, key=lambda r: (r[1]["right"], r[0]))[0] if okf else max(fr, key=lambda r: (-r[1]["wrong"], r[1]["right"], r[0]))[0]
    nested = pool(raw[c]["res"][("fuse", chosen[c], c)] for c in CLIPS)
    lam = LAMBDA[head[cap]]
    jr = pool(raw[c]["res"][("fuse", choose(c, ["fuse"], lam, [s for s in CLIPS if s != c])[2], c)] for c in CLIPS)
    L.append(f"| {cap} | margin {om}: {fmt(op)} | {fmt(nested)} | {dict(sorted(Counter(chosen.values()).items()))} | {head[cap]}: {fmt(jr)} |")
open(HERE + "out/nested_variants.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
