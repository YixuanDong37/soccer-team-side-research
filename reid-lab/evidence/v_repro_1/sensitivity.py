"""fuse_r1 / sensitivity.py -- how much the three operating points move when a threshold or a hand-set
constant is changed (LABELS: scoring only; nothing is refitted or re-chosen).

Every clip keeps its fold's numbers and its fold's chosen setting (params_loco.json, mode "heldout")
or the frozen setting with fold weights (mode "fixed"); one thing is changed at a time:
  margin x 0.9 / x 1.1            the one threshold of the method
  floor 0.27 / 0.33               appearance floor (0.3)
  nobody 0.45 / 0.55              the "nobody" rival t0 (0.5)
  bonus x 0.9 / x 1.1             size of the position / plugin bonus
  solid 4 / 6 / 3 / 10            boxes a piece needs to be solid (5)
  cuts P 0.63 / 0.77              m_cuts' probability level (0.7)
  all looser / all stricter       margin, floor, nobody x 0.9 / x 1.1 together (bonus x 1.1 / x 0.9)
  40 random draws                 margin, floor, nobody, bonus each x uniform(0.9, 1.1), same draw for all clips

    python -I sensitivity.py [workers] [heldout|fixed]     -> out/sensitivity_<mode>.md, .json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_1")
from reid_lab import data  # noqa: E402
import method  # noqa: E402

HERE = Path(__file__).resolve().parent
N_RANDOM = 40
SINGLE = [("base", {}),
          ("margin x 0.9", {"margin": 0.9}), ("margin x 1.1", {"margin": 1.1}),
          ("floor 0.27", {"floor": 0.9}), ("floor 0.33", {"floor": 1.1}),
          ("nobody 0.45", {"t0": 0.9}), ("nobody 0.55", {"t0": 1.1}),
          ("bonus x 0.9", {"scale": 0.9}), ("bonus x 1.1", {"scale": 1.1}),
          ("solid 4 boxes", {"n_solid": 4}), ("solid 6 boxes", {"n_solid": 6}),
          ("solid 3 boxes", {"n_solid": 3}), ("solid 10 boxes", {"n_solid": 10}),
          ("cuts P 0.63", {"p_cut": 0.63}), ("cuts P 0.77", {"p_cut": 0.77}),
          ("all looser", {"margin": 0.9, "floor": 0.9, "t0": 0.9, "scale": 1.1}),
          ("all stricter", {"margin": 1.1, "floor": 1.1, "t0": 1.1, "scale": 0.9})]


def settings():
    rng = np.random.default_rng(20261115)
    rnd = [(f"random {i + 1}", dict(zip(("margin", "floor", "t0", "scale"), rng.uniform(0.9, 1.1, 4).round(4).tolist()))) for i in range(N_RANDOM)]
    return SINGLE + rnd


def run_clip(job):
    seq, mode = job
    from reid_lab import score          # LABELS: scoring only
    params = method.load_params()
    par = params[seq]
    out = {}
    for name, ch in settings():
        for point in method.POINTS:
            c = params[method.ALL if mode == "fixed" else seq]["choice"][point]
            cuts, core, leaf = method.clip_links(
                seq, c["margin"] * ch.get("margin", 1.0), c["leaves"], par, seq, n_solid=ch.get("n_solid", method.N_SOLID),
                scale=ch.get("scale", 1.0), floor=par["floor"] * ch.get("floor", 1.0), t0=par["t0"] * ch.get("t0", 1.0), p_cut=ch.get("p_cut"))
            r = score.score_clip(seq, core + leaf, [list(x) for x in cuts])
            out[(name, point)] = {**dict(Counter(r["per_return"])), "lw": r["links"]["wrong"], "lr": r["links"]["right"], "lu": r["links"]["unknown"],
                                  "impure": r["impure_rows"], "rows": r["tracked_rows"]}
    return seq, out


def main():
    from concurrent.futures import ProcessPoolExecutor

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    mode = sys.argv[2] if len(sys.argv) > 2 else "heldout"
    res = {}
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex:
        for seq, out in ex.map(run_clip, [(s, mode) for s in data.DEV_CLIPS]):
            res[seq] = out
            print(seq, "done", flush=True)
    if len(res) != len(data.DEV_CLIPS):
        raise SystemExit("a clip is missing")
    pooled = {}
    for name, _ in settings():
        for point in method.POINTS:
            g = lambda k: int(sum(res[s][(name, point)].get(k, 0) for s in res))  # noqa: E731
            pooled[(name, point)] = {"right": g("right"), "wrong": g("wrong"), "lost": g("lost"), "links_right": g("lr"), "links_wrong": g("lw"),
                                     "links_unknown": g("lu"), "impure_row_share": round(g("impure") / g("rows"), 5)}
    L = [f"# fuse_r1: sensitivity of the operating points ({mode}; 137 scored returns)", "",
         "Cells: right / wrong / lost | links right / wrong / unknown | impure rows. One change at a time; nothing refitted or re-chosen.", "",
         "| change | strict | balanced | recall |", "|---|---|---|---|"]
    cell = lambda p: f"{p['right']} / {p['wrong']} / {p['lost']} | {p['links_right']} / {p['links_wrong']} / {p['links_unknown']} | {100 * p['impure_row_share']:.2f}%"  # noqa: E731
    for name, _ in SINGLE:
        L.append(f"| {name} | " + " | ".join(cell(pooled[(name, p)]) for p in method.POINTS) + " |")
    L += ["", f"{N_RANDOM} random draws (margin, floor, nobody, bonus each x uniform(0.9, 1.1)):", "",
          "| point | right min / median / max | wrong min / median / max | draws by wrong count | draws over the limit |", "|---|---|---|---|---|"]
    caps = {"strict": 4, "balanced": 7, "recall": 14}
    summary = {}
    for p in method.POINTS:
        R = np.array([pooled[(f"random {i + 1}", p)]["right"] for i in range(N_RANDOM)])
        W = np.array([pooled[(f"random {i + 1}", p)]["wrong"] for i in range(N_RANDOM)])
        cnt = Counter(W.tolist())
        summary[p] = {"right_min_med_max": [int(R.min()), float(np.median(R)), int(R.max())], "wrong_min_med_max": [int(W.min()), float(np.median(W)), int(W.max())],
                      "wrong_counts": {str(k): v for k, v in sorted(cnt.items())}, "over_limit": int((W > caps[p]).sum())}
        L.append(f"| {p} | {R.min()} / {np.median(R):g} / {R.max()} | {W.min()} / {np.median(W):g} / {W.max()} | "
                 f"{', '.join(f'{v}x {k} wrong' for k, v in sorted(cnt.items()))} | {int((W > caps[p]).sum())} of {N_RANDOM} |")
    text = "\n".join(L) + "\n"
    (HERE / "out" / f"sensitivity_{mode}.md").write_text(text, encoding="utf-8")
    (HERE / "out" / f"sensitivity_{mode}.json").write_text(json.dumps({"pooled": {f"{k[0]}|{k[1]}": v for k, v in pooled.items()}, "random": summary,
                                                                         "draws": {n: c for n, c in settings() if n.startswith("random")}}, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
