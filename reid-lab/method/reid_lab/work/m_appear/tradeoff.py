"""The trade-off table of m_appear, from the files loco.py and method.py wrote into out/ (scored again here).

    python -I tradeoff.py            -> prints the table and writes out/tradeoff.md
"""
import json
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reid_lab import score  # noqa: E402

ROWS = [
    ("baselines (reid_lab/score.py)", None),
    ("gbf now: split, no links", ("baseline", "split")),
    ("plugin: keep every plugin join", ("baseline", "plugin")),
    ("mutual: appearance_match.py pairs, margin 0.05", ("baseline", "mutual")),
    ("HELD-OUT (leave one clip out): agglomeration; kind, t_abs, margin picked on the other 12 clips", None),
    ("agglo lam=1  (recall; not recommended)", ("file", "agglo-lam1")),
    ("agglo lam=2  (balanced)", ("file", "agglo-lam2")),
    ("agglo lam=3  (cautious)", ("file", "agglo-lam3")),
    ("agglo lam=5", ("file", "agglo-lam5")),
    ("agglo lam=10 (strict)", ("file", "agglo-lam10")),
    ("HELD-OUT: agglomeration, only the margin picked on the other 12 clips (kind w, t_abs 0.5 fixed after seeing all 13)", None),
    ("agglo-w-0.5 lam=1", ("file", "agglo-kind_w-t_abs_0.5-lam1")),
    ("agglo-w-0.5 lam=2", ("file", "agglo-kind_w-t_abs_0.5-lam2")),
    ("agglo-w-0.5 lam=3", ("file", "agglo-kind_w-t_abs_0.5-lam3")),
    ("agglo-w-0.5 lam=5 and 10", ("file", "agglo-kind_w-t_abs_0.5-lam5")),
    ("HELD-OUT: pairwise mutual best + rival margin on the same features (no chains)", None),
    ("pairwise lam=1", ("file", "mutual-lam1")),
    ("pairwise lam=2", ("file", "mutual-lam2")),
    ("pairwise lam=3, 5, 10", ("file", "mutual-lam3")),
    ("HELD-OUT: as the first block, with the rival rule (share >= 1 or >= 5 frames) also picked inside the loop", None),
    ("agglo nested-covis lam=2", ("file", "agglo+agglo_covis5-lam2")),
    ("agglo nested-covis lam=3", ("file", "agglo+agglo_covis5-lam3")),
    ("IN-SAMPLE (parameters fitted on all 13 clips; secondary): frozen operating points of method.py", None),
    ("frozen recall   (w, t_abs 0.40, margin 0.15)", ("file", "frozen-recall")),
    ("frozen balanced (w, t_abs 0.50, margin 0.25)", ("file", "frozen-balanced")),
    ("frozen cautious (w, t_abs 0.50, margin 0.30)", ("file", "frozen-cautious")),
    ("frozen strict   (w, t_abs 0.50, margin 0.35)", ("file", "frozen-strict")),
]

if __name__ == "__main__":
    lines = ["| method | right | wrong | lost | links right / wrong / unknown | link precision | impure rows |", "|---|---|---|---|---|---|---|"]
    for name, src in ROWS:
        if src is None:
            lines.append(f"| **{name}** | | | | | | |")
            continue
        if src[0] == "baseline":
            m = score.baseline_links(src[1])
        else:
            with open(os.path.join(HERE, "out", f"{src[1]}.method.json"), encoding="utf-8") as f:
                m = json.load(f)["clips"]
        p = score.evaluate(m)["pooled"]
        prec = "-" if p["link_precision"] is None else f"{p['link_precision']:.3f}"
        lines.append(f"| {name} | {p['right']} | {p['wrong']} | {p['lost']} | {p['links_right']} / {p['links_wrong']} / {p['links_unknown']} | {prec} | {100 * p['impure_row_share']:.2f} % |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "out", "tradeoff.md"), "w", encoding="utf-8") as f:
        f.write("137 scored returns in the 13 dev clips (30 more are undetected). lam = price of one wrong identity\n"
                "(extra wrong return or wrong link) in recovered returns, used to pick the parameters.\n\n" + text + "\n")
