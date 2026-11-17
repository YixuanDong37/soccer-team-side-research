"""Pool the per-clip re-entry checks (gsr_reentry_eval.py) of the SN-GSR-2025 test clips.

Run in gbf's .venv: python -I gsr_summary.py RUNS_DIR SELECTION_JSON OUT.json

Counts are summed over clips. Intervals resample whole clips (returns and links of one clip are
not independent): 2000 draws of clip indices made once (seed 0) and shared by every quantity, so
differences between assignments are paired; each difference gets its own interval. Returns are
also split by gap length. A selected clip without a result is listed and makes the exit code 1.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BINS = [(10, 25, "0.4-1 s"), (25, 75, "1-3 s"), (75, 250, "3-10 s"), (250, 10**9, ">10 s")]
OUTCOMES = ("right", "wrong", "lost")
DRAWS = 2000


def share(k: np.ndarray, n: np.ndarray, draws: np.ndarray) -> dict:
    if n.sum() == 0:
        return {"k": 0, "n": 0, "share": None, "ci95": None}
    kk, nn = k[draws].sum(axis=1), n[draws].sum(axis=1)
    boot = kk[nn > 0] / nn[nn > 0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"k": int(k.sum()), "n": int(n.sum()), "share": round(float(k.sum() / n.sum()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}  # fmt: skip


def diff(k1: np.ndarray, k2: np.ndarray, n: np.ndarray, draws: np.ndarray) -> dict:
    """share1 - share2 over the same returns (same n), paired by clip."""
    if n.sum() == 0:
        return {"diff": None, "ci95": None}
    nn = n[draws].sum(axis=1)
    ok = nn > 0
    boot = (k1[draws].sum(axis=1)[ok] - k2[draws].sum(axis=1)[ok]) / nn[ok]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": round(float((k1.sum() - k2.sum()) / n.sum()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def main() -> int:
    runs, selection, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    clips = json.loads(selection.read_text(encoding="utf-8"))["selected"]
    done = [c for c in clips if (runs / c / "reentry-eval.json").exists()]
    missing = [c for c in clips if c not in done]
    if missing:
        print(f"WARNING: {len(missing)} selected clip(s) have no result and are left out: {missing}")
    evals = {c: json.loads((runs / c / "reentry-eval.json").read_text(encoding="utf-8")) for c in done}
    if not evals:
        raise SystemExit("no clip has a result")
    names = list(next(iter(evals.values()))["outcomes"])
    draws = np.random.default_rng(0).integers(0, len(evals), (DRAWS, len(evals)))
    E = list(evals.values())
    count = lambda cfg, o: np.array([e["outcomes"][cfg].get(o, 0) for e in E], float)  # noqa: E731
    n = np.array([sum(e["outcomes"][names[0]].values()) for e in E], float)  # scored returns, same for every assignment

    summary = {"clips_selected": len(clips), "clips_evaluated": len(evals), "clips_missing": missing}
    summary["annotation"] = {k: int(sum(e["annotation"][k] for e in E))
                             for k in ("frames", "player_gk_ids", "returns_at_edge", "occlusion_gaps")}  # fmt: skip
    summary["annotation"]["clips_with_same_team_jersey_on_two_ids"] = sum(1 for e in E if e["annotation"]["same_team_and_jersey_on_two_ids"])
    summary["returns_scored"] = int(n.sum())
    summary["returns_undetected"] = int(sum(e["undetected_returns"] for e in E))
    summary["returns_untracked"] = int(sum(e.get("untracked_returns", 0) for e in E))
    summary["returns_spanning_a_gbf_cut"] = int(sum(1 for e in E for r in e["returns"] if r.get("spans_detected_cut")))
    summary["clips_where_appearance_failed"] = [c for c, e in evals.items() if e.get("appearance_failed")]
    summary["matching"] = {k: int(sum(e["matching"][k] for e in E)) for k in ("predicted_rows", "untracked_rows", "matched_rows")}
    summary["returns_outcomes"] = {cfg: {o: share(count(cfg, o), n, draws) for o in OUTCOMES} for cfg in names}
    summary["contaminated_groups"] = {cfg: int(sum(e["contaminated"][cfg] for e in E)) for cfg in names}
    summary["paired_differences_vs_plugin"] = {
        cfg: {o: diff(count(cfg, o), count("plugin", o), n, draws) for o in OUTCOMES} for cfg in names if cfg != "plugin"}
    summary["paired_differences_vs_split"] = {
        cfg: {o: diff(count(cfg, o), count("split", o), n, draws) for o in OUTCOMES}
        for cfg in names if cfg not in ("plugin", "split")}  # fmt: skip
    by_gap = {}
    for lo, hi, label in BINS:
        by_gap[label] = {cfg: dict(Counter(r["outcome"][cfg] for e in E for r in e["returns"]
                                           if isinstance(r["outcome"], dict) and lo <= r["gap"] < hi)) for cfg in names}  # fmt: skip
    summary["returns_by_gap"] = by_gap
    links = {}
    for key in E[0]["links"]:
        right = np.array([e["links"][key]["right"] for e in E], float)
        judged = np.array([e["links"][key]["right"] + e["links"][key]["wrong"] for e in E], float)
        links[key] = {"links": int(sum(e["links"][key]["links"] for e in E)),
                      "unknown": int(sum(e["links"][key]["unknown"] for e in E)), "precision": share(right, judged, draws)}  # fmt: skip
    summary["links"] = links
    joins = Counter()
    for e in E:
        joins.update(e["plugin_reentry_joins"])
    wrong_j = np.array([e["plugin_reentry_joins"].get("plugin wrong (split removed a false identity)", 0) for e in E], float)
    right_j = np.array([e["plugin_reentry_joins"].get("plugin right (split broke a true identity)", 0) for e in E], float)
    summary["plugin_reentry_joins"] = dict(joins)
    summary["plugin_reentry_joins_wrong_share"] = share(wrong_j, wrong_j + right_j, draws)
    summary["per_clip"] = {c: {"returns": e["annotation"]["returns_at_edge"], "outcomes": e["outcomes"],
                               "links": {k: [v["right"], v["wrong"]] for k, v in e["links"].items()},
                               "chosen": e["chosen_thresholds_on_clip"] and [e["chosen_thresholds_on_clip"]["t_abs"],
                                                                             e["chosen_thresholds_on_clip"]["margin"]]}
                           for c, e in evals.items()}  # fmt: skip
    summary["restrictions"] = ["derived from SN-GSR-2025 annotations: publish numbers only, never frames"]
    out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("clips_evaluated", "annotation", "returns_scored", "returns_undetected",
                                              "returns_untracked", "returns_outcomes", "links", "plugin_reentry_joins",
                                              "plugin_reentry_joins_wrong_share")}, indent=1))  # fmt: skip
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
