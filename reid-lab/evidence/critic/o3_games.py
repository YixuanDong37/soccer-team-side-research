"""critic / o3_games.py -- DIAGNOSIS (labels): the 13 dev clips come from 3 matches. How do the results
look per match, and what happens when the SETTING of an operating point is chosen without any clip of
the held-out clip's match (leave-one-match-out for the choice of the setting)?

Reads fuse_r1/cache/loco_raw.pkl (every fold x candidate x clip, already scored by fuse_r1) and
fuse_r1/rules.py (pure functions). APPROXIMATION, stated: the weights (pair model, m_cuts, m_space) of
a fold are still fitted on the 12 other clips, which include clips of the same match; only the choice
of (leaves, margin) is made without the match. A full leave-one-match-out refit was not run.
"""
import pickle
import sys
from collections import Counter

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/critic")
import common as C
from common import data

sys.path.insert(0, str(C.FUSE))
import rules  # noqa: E402  fuse_r1/rules.py

P = C.Tee(C.OUT / "o3_games.txt")
raw = pickle.load(open(C.FUSE / "cache" / "loco_raw.pkl", "rb"))
folds = raw["folds"]
MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)
CANDS = [(lv, m) for lv in (False, True) for m in MARGINS]
game = {s: C.game(s) for s in data.DEV_CLIPS}
games = sorted(set(game.values()))
P("match of each dev clip:", game)
P("clips per match:", dict(Counter(game.values())))


def show(cells):
    p = rules.pool(cells)
    return f"{p['right']:3d} / {p['wrong']:2d} / {p['lost']:3d} | links {p['links_right']:3d} / {p['links_wrong']:2d} / {p['links_unknown']:3d}"


P("\n== fuse_r1 rule B, held-out per clip, pooled per match ==")
for point in ("strict", "balanced", "recall"):
    chosen = {c: rules.choose(folds[c]["res"], "r1", CANDS, [s for s in data.DEV_CLIPS if s != c], point, "B")[0] for c in data.DEV_CLIPS}
    cells = {c: rules.cell(folds[c]["res"], "r1", chosen[c], c) for c in data.DEV_CLIPS}
    P(f"{point:9s} all: {show(cells.values())}")
    for g in games:
        P(f"          match {g}: {show(v for c, v in cells.items() if game[c] == g)}   ({sum(game[c] == g for c in cells)} clips)")

P("\n== fixed setting (leaves off), held-out weights, pooled per match: right / wrong / lost ==")
P("margin   " + "   ".join(f"match {g:>2s}      " for g in games) + "   match 9 without SNGS-162")
for m in MARGINS:
    row = []
    for g in games:
        p = rules.pool(rules.cell(folds[c]["res"], "r1", (False, m), c) for c in data.DEV_CLIPS if game[c] == g)
        row.append(f"{p['right']:3d} / {p['wrong']:2d} / {p['lost']:3d}")
    p = rules.pool(rules.cell(folds[c]["res"], "r1", (False, m), c) for c in data.DEV_CLIPS if game[c] == "9" and c != "SNGS-162")
    P(f"{m:5.2f}    " + "    ".join(row) + f"    {p['right']:3d} / {p['wrong']:2d} / {p['lost']:3d}")

P("\n== setting chosen WITHOUT the held-out clip's match (weights still leave-one-clip-out) ==")
for rule in ("B", "R0", "Rcap", "Rnotrim"):
    for point in ("strict", "balanced", "recall"):
        chosen, cells = {}, {}
        for c in data.DEV_CLIPS:
            fit = [s for s in data.DEV_CLIPS if game[s] != game[c]]
            chosen[c] = rules.choose(folds[c]["res"], "r1", CANDS, fit, point, rule)[0]
            cells[c] = rules.cell(folds[c]["res"], "r1", chosen[c], c)
        per_game = {g: sorted({f"{chosen[c][1]:.2f}{'+leaves' if chosen[c][0] else ''}" for c in data.DEV_CLIPS if game[c] == g}) for g in games}
        P(f"rule {rule:7s} {point:9s}: {show(cells.values())} | settings per held-out match {per_game}")
        for g in games:
            P(f"      match {g}: {show(v for c, v in cells.items() if game[c] == g)}")
