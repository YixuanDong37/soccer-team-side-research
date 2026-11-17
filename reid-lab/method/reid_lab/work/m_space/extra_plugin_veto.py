"""EXTRA, NOT position-only: position evidence as a veto / confirmation of the plugin's own
re-joins (the plugin's track id across a re-entry gap rests on its appearance model). Shown as
a pointer for the fusion step; LOCO weights (posscore out-of-fold scores), thresholds fixed.

    python -I extra_plugin_veto.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data, score  # noqa: E402
import posscore  # noqa: E402

HERE = Path(__file__).resolve().parent


def fmt(d):
    return (f"right {d['right']:3d}  wrong {d['wrong']:2d}  lost {d['lost']:3d} | links {d['links']:3d}: right {d['links_right']:3d} wrong {d['links_wrong']:2d} "
            f"unknown {d['links_unknown']:3d} | impure rows {100 * d['impure_row_share']:.2f}%")


plugin = score.baseline_links("plugin")
scores = {s: posscore.position_scores(s).set_index(["a", "b"]) for s in data.DEV_CLIPS}
strict = json.loads((HERE / "METHOD_pos_strict.json").read_text(encoding="utf-8"))["clips"]
print("plugin re-joins (baseline)          :", fmt(score.evaluate(plugin)["pooled"]))
print("position strict (LOCO)              :", fmt(score.evaluate(strict)["pooled"]))
rules = {
    "plugin & admissible": lambda r: bool(r.pos_adm),
    "plugin & rank_in <= 3": lambda r: bool(r.pos_adm) and r.pos_rank_in <= 3,
    "plugin & rank_in <= 2 & rank_out <= 2": lambda r: bool(r.pos_adm) and r.pos_rank_in <= 2 and r.pos_rank_out <= 2,
    "plugin & rank_in == 1": lambda r: bool(r.pos_adm) and r.pos_rank_in == 1,
    "plugin & rank_in == 1 & rank_out == 1": lambda r: bool(r.pos_adm) and r.pos_rank_in == 1 and r.pos_rank_out == 1,
    "plugin & pos_sink >= 0.10": lambda r: bool(r.pos_adm) and r.pos_sink >= 0.10,
    "plugin & pos_sink >= 0.20": lambda r: bool(r.pos_adm) and r.pos_sink >= 0.20,
}
out = {}
for name, keep in rules.items():
    method, union = {}, {}
    for s in data.DEV_CLIPS:
        kept = []
        for a, b in plugin[s]["links"]:
            a, b = (a, b) if (a, b) in scores[s].index else (b, a)
            if (a, b) in scores[s].index and keep(scores[s].loc[(a, b)]):
                kept.append([int(a), int(b)])
        method[s] = {"links": kept}
        # union with the strict position links: drop plugin links that clash with a position link's slots
        pos = [list(map(int, l)) for l in strict[s]["links"]]
        used_a, used_b = {l[0] for l in pos}, {l[1] for l in pos}
        union[s] = {"links": pos + [l for l in kept if l not in pos and l[0] not in used_a and l[1] not in used_b]}
    a, b = score.evaluate(method)["pooled"], score.evaluate(union)["pooled"]
    out[name] = {"plugin_filtered": a, "union_with_pos_strict": b}
    print(f"{name:38s}: {fmt(a)}")
    print(f"{'   ... union with position strict':38s}: {fmt(b)}")
(HERE / "extra_plugin_veto.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

# the one rule kept as a METHOD file (fixed rule, LOCO position weights; picked among the rules
# above after seeing their pooled numbers, so treat it as a pointer, not as a validated result)
name = "plugin & rank_in <= 2 & rank_out <= 2"
keep = rules[name]
method = {}
for s in data.DEV_CLIPS:
    kept = []
    for a, b in plugin[s]["links"]:
        a, b = (a, b) if (a, b) in scores[s].index else (b, a)
        if (a, b) in scores[s].index and keep(scores[s].loc[(a, b)]):
            kept.append([int(a), int(b)])
    pos = [list(map(int, l)) for l in strict[s]["links"]]
    used_a, used_b = {l[0] for l in pos}, {l[1] for l in pos}
    method[s] = {"links": pos + [l for l in kept if l not in pos and l[0] not in used_a and l[1] not in used_b]}
(HERE / "METHOD_extra_pos_strict_plus_plugin_rank2.json").write_text(json.dumps({"name": "m_space/extra_pos_strict_plus_plugin_rank2", "meta": {"note": "NOT position-only: position strict links plus the plugin's own re-joins that the position model ranks among the two best candidates on both sides"}, "clips": method}, indent=1) + "\n", encoding="utf-8")
res = score.evaluate(method)
(HERE / "OUT_extra_pos_strict_plus_plugin_rank2.json").write_text(json.dumps({"name": "m_space/extra_pos_strict_plus_plugin_rank2", **res}, indent=1) + "\n", encoding="utf-8")
print("written METHOD_extra_pos_strict_plus_plugin_rank2.json:", fmt(res["pooled"]))
