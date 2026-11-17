"""The position (+ team / role) only linker, end to end, with leave-one-clip-out (LOCO) fitting.

    python -I run_position_linker.py            (about 4 minutes)

Writes into this directory:
  params_loco.json        {held-out clip: weights fitted on the other 12 clips}
  params_all13.json       weights fitted on all 13 clips (for clips outside the dev set; in-sample here)
  tradeoff.json / .md     pooled LOCO held-out counts (and in-sample counts) over a threshold grid, for
                          emit = 2 (links only between a piece that ends at an image border and one that
                          starts at one) and emit = 1 (at least one of the two at a border)
  METHOD_pos_e<E>_t<NN>.json   links at fixed (emit, threshold) (LOCO weights; nothing else tuned)
  METHOD_pos_strict.json  (emit, threshold) chosen per fold on the 12 fit clips: most returns right
                          there among the settings that make no return wrong there and keep the judged
                          link precision >= 0.95
  METHOD_pos_balanced.json  (emit, threshold) chosen per fold on the 12 fit clips: most
                          (right - 3 * new wrong)
  OUT_*.json              score.py results of each METHOD file

Labels are used only (b) for fitting inside the LOCO loop (linker.fit on the fit clips; the rules
that choose emit and threshold are evaluated on the fit clips) and (a) for scoring through
reid_lab.score.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data, score  # noqa: E402
import features  # noqa: E402
import linker  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIG = dict(resid="r_adv_cv", pair=("lu", "ls", "tu", "ee", "ex", "fa", "ba"), ctx=("one", "edge", "time"), edge_weight=3.0)
MODE, FLOOR = "sinkhorn", 0.1
EMITS = (2, 1)
GRID = (0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20)
FIXED = (0.60, 0.50, 0.40)
N_PARAMS = 3 + len(CONFIG["pair"]) + 2 * len(CONFIG["ctx"])
EMIT = 2  # default of the helpers below (sensitivity.py, tradeoff_emit.py)


def sequences(params_by_clip: dict) -> dict:
    return {seq: linker.peel_sequence(features.pair_features(seq), pr, MODE, FLOOR) for seq, pr in params_by_clip.items()}


def method_at(seqs: dict, setting) -> dict:
    """setting: (emit, theta), or {clip: (emit, theta)}."""
    st = (lambda s: setting[s]) if isinstance(setting, dict) else (lambda s: setting)
    return {s: {"links": linker.cut_sequence(q, st(s)[1], st(s)[0])} for s, q in seqs.items()}


def pooled(method: dict, clips=data.DEV_CLIPS) -> dict:
    return score.evaluate(method, clips=clips)["pooled"]


def precision(d: dict) -> float:
    judged = d["links_right"] + d["links_wrong"]
    return d["links_right"] / judged if judged else 1.0


def choose(curve: dict, base_wrong: int, rule: str):
    """curve: {(emit, theta): pooled result on the fit clips} -> the chosen (emit, theta).
    Ties go to the higher threshold, then to the stricter emit."""
    keys = sorted(curve, key=lambda k: (k[1], k[0]))
    if rule == "strict":
        ok = [k for k in keys if curve[k]["wrong"] <= base_wrong and precision(curve[k]) >= 0.95]
        return max(ok, key=lambda k: (curve[k]["right"], k[1], k[0])) if ok else (max(EMITS), max(GRID))
    if rule == "balanced":
        return max(keys, key=lambda k: (curve[k]["right"] - 3 * (curve[k]["wrong"] - base_wrong), k[1], k[0]))
    raise ValueError(rule)


def write_method(name: str, method: dict, meta: dict) -> dict:
    path = HERE / f"METHOD_{name}.json"
    path.write_text(json.dumps({"name": f"m_space/{name}", "meta": meta, "clips": method}, indent=1) + "\n", encoding="utf-8")
    res = score.evaluate(method)
    (HERE / f"OUT_{name}.json").write_text(json.dumps({"name": f"m_space/{name}", **res}, indent=1) + "\n", encoding="utf-8")
    return res["pooled"]


def fmt(d: dict) -> str:
    return (f"right {d['right']:3d}  wrong {d['wrong']:2d}  lost {d['lost']:3d} | links {d['links']:3d}: right {d['links_right']:3d} wrong {d['links_wrong']:2d} "
            f"unknown {d['links_unknown']:3d} | impure rows {100 * d['impure_row_share']:.2f}%")


def main() -> None:
    meta = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items()}, "mode": MODE, "peel": True,
            "params_fitted_per_fold": N_PARAMS}
    # ---- fit: LOCO and all 13
    loco = linker.loco_params(**CONFIG)
    all13 = linker.fit(data.DEV_CLIPS, **CONFIG)
    (HERE / "params_loco.json").write_text(json.dumps(loco, indent=1) + "\n", encoding="utf-8")
    (HERE / "params_all13.json").write_text(json.dumps(all13, indent=1) + "\n", encoding="utf-8")
    held = sequences(loco)                                   # each clip under the weights that never saw it
    insample = sequences({s: all13 for s in data.DEV_CLIPS})
    base = score.evaluate(score.baseline_links("split"))
    print("split baseline:", fmt(base["pooled"]))

    # ---- trade-off over fixed settings
    rows = []
    lines = ["| emit | theta | right | wrong | lost | links right | links wrong | links unknown | impure rows | in-sample right / wrong / lost |", "|---|---|---|---|---|---|---|---|---|---|"]
    for emit in EMITS:
        for t in GRID:
            h, i = pooled(method_at(held, (emit, t))), pooled(method_at(insample, (emit, t)))
            rows.append({"emit": emit, "theta": t, "heldout": h, "insample": i})
            print(f"emit>={emit} theta {t:.2f}  LOCO held-out: {fmt(h)}   || in-sample: R{i['right']} W{i['wrong']} L{i['lost']} links r{i['links_right']} w{i['links_wrong']} u{i['links_unknown']}")
            lines.append(f"| {emit} | {t:.2f} | {h['right']} | {h['wrong']} | {h['lost']} | {h['links_right']} | {h['links_wrong']} | {h['links_unknown']} | {100 * h['impure_row_share']:.2f}% | {i['right']} / {i['wrong']} / {i['lost']} |")
    (HERE / "tradeoff.json").write_text(json.dumps({"meta": meta, "baseline_split": base["pooled"], "rows": rows}, indent=1) + "\n", encoding="utf-8")
    (HERE / "tradeoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for emit in EMITS:
        for t in FIXED:
            name = f"pos_e{emit}_t{int(round(100 * t)):02d}"
            p = write_method(name, method_at(held, (emit, t)), {**meta, "emit_kind": emit, "theta": t, "choice": "fixed, not tuned"})
            print(f"METHOD_{name}: {fmt(p)}")

    # ---- (emit, threshold) chosen inside LOCO: per fold, on the 12 fit clips, with the fold's weights
    chosen = {"strict": {}, "balanced": {}}
    for fit_clips, held_clip in data.loco_folds():
        pr = loco[held_clip]
        fit_seqs = {s: linker.peel_sequence(features.pair_features(s), pr, MODE, FLOOR) for s in fit_clips}
        curve = {(e, t): pooled(method_at(fit_seqs, (e, t)), clips=fit_clips) for e in EMITS for t in GRID}
        base_wrong = sum(base["per_clip"][s]["returns"]["wrong"] for s in fit_clips)
        for rule in chosen:
            chosen[rule][held_clip] = choose(curve, base_wrong, rule)
    for rule, by_clip in chosen.items():
        p = write_method(f"pos_{rule}", method_at(held, by_clip),
                         {**meta, "choice": f"rule '{rule}' on the 12 fit clips of each fold", "emit_theta_by_clip": {s: list(v) for s, v in by_clip.items()}})
        print(f"METHOD_pos_{rule}: (emit, theta) chosen per fold {sorted(set(by_clip.values()))} -> {fmt(p)}")
    # in-sample counterpart of the rules (chosen and applied on all 13: optimistic)
    curve = {(e, t): pooled(method_at(insample, (e, t))) for e in EMITS for t in GRID}
    for rule in chosen:
        k = choose(curve, base["pooled"]["wrong"], rule)
        print(f"in-sample {rule}: (emit, theta) {k} -> {fmt(curve[k])}")


if __name__ == "__main__":
    main()
