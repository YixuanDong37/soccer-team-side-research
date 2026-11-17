"""v_robust_0 / r3_sensitivity.py -- (LABELS: scoring only) sensitivity of fuse's held-out result to
+-10% changes of its thresholds. Nothing is refitted: every clip keeps the numbers of ITS fold
(fuse/params_loco.json: pair model, margins; m_cuts' and m_space's held-out weights) and ONE
threshold is moved by -10% / +10% at a time; then all of them together in the loose and in the
strict direction; then 40 random joint draws (each threshold x U[0.9, 1.1], numpy default_rng(0)).

    python -I -B r3_sensitivity.py [workers]   -> out/r3_sensitivity.md, out/r3_sensitivity.json

Thresholds moved:
  margin        the operating point's margin (per fold) and the fixed margins 0.25 / 0.5
  floor         appearance floor 0.3
  t0            the "nobody" rival 0.5              (t_abs 0.5 moved separately: it is inactive)
  bonus         both bonus weights (position and plugin) relative to the appearance weight
  w_sink/w_plug the position weight / the plugin weight alone
  p_cut         m_cuts' probability level 0.7 -> 0.63 / 0.77 (new cut lists, new tables)
  MAX_SPEED     m_space's speed bound 8 m/s -> 7.2 / 8.8;  START_RADIUS 3 m -> 2.7 / 3.3
"""
import json
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_0")
import numpy as np

import vr_common as vc
from vr_common import data, evidence, linker, method, prep

POINTS = ("strict", "balanced", "recall", "fixed_0.50", "fixed_0.25")
BASE = dict(margin=1.0, floor=1.0, t0=1.0, t_abs=1.0, bonus=1.0, w_sink=1.0, w_plug=1.0, p_cut=0.7, MAX_SPEED=8.0, START_RADIUS=3.0)
ONE = {
    "base": {},
    "margin -10%": {"margin": 0.9}, "margin +10%": {"margin": 1.1},
    "floor -10%": {"floor": 0.9}, "floor +10%": {"floor": 1.1},
    "t0 (nobody rival) -10%": {"t0": 0.9}, "t0 (nobody rival) +10%": {"t0": 1.1},
    "t_abs -10%": {"t_abs": 0.9}, "t_abs +10%": {"t_abs": 1.1},
    "bonus -10%": {"bonus": 0.9}, "bonus +10%": {"bonus": 1.1},
    "position weight -10%": {"w_sink": 0.9}, "position weight +10%": {"w_sink": 1.1},
    "plugin weight -10%": {"w_plug": 0.9}, "plugin weight +10%": {"w_plug": 1.1},
    "p_cut 0.63 (-10%: more cuts)": {"p_cut": 0.63}, "p_cut 0.77 (+10%: fewer cuts)": {"p_cut": 0.77},
    "MAX_SPEED 7.2 (-10%)": {"MAX_SPEED": 7.2}, "MAX_SPEED 8.8 (+10%)": {"MAX_SPEED": 8.8},
    "START_RADIUS 2.7 (-10%)": {"START_RADIUS": 2.7}, "START_RADIUS 3.3 (+10%)": {"START_RADIUS": 3.3},
    "ALL looser (margin, floor, t0 -10%; bonus +10%; p_cut 0.77; speed, radius +10%)":
        {"margin": 0.9, "floor": 0.9, "t0": 0.9, "bonus": 1.1, "p_cut": 0.77, "MAX_SPEED": 8.8, "START_RADIUS": 3.3},
    "ALL stricter (margin, floor, t0 +10%; bonus -10%; p_cut 0.63; speed, radius -10%)":
        {"margin": 1.1, "floor": 1.1, "t0": 1.1, "bonus": 0.9, "p_cut": 0.63, "MAX_SPEED": 7.2, "START_RADIUS": 2.7},
    "margin -20%": {"margin": 0.8}, "margin +20%": {"margin": 1.2},
}
N_RANDOM = 40


def random_configs():
    rng = np.random.default_rng(0)
    return {f"random {i:02d}": dict(zip(("margin", "floor", "t0", "bonus"), [float(x) for x in rng.uniform(0.9, 1.1, 4)])) for i in range(N_RANDOM)}


def run_clip(seq: str):
    from reid_lab import score  # LABELS: scoring

    t0 = time.time()
    par = vc.params()[seq]
    pp = method.pos_params(seq)
    cols, w = par["pair_model"]["cols"], np.asarray(par["pair_model"]["w"])
    out = {}
    tabs = {}
    configs = {**ONE, **random_configs()}
    for name, change in configs.items():
        c = {**BASE, **change}
        key = (c["p_cut"], c["MAX_SPEED"], c["START_RADIUS"])
        if key not in tabs:
            cuts = prep.cuts_for(seq, seq, c["p_cut"])
            vc.SPACE_PATCH.clear()
            if c["MAX_SPEED"] != 8.0:
                vc.SPACE_PATCH["MAX_SPEED"] = c["MAX_SPEED"]
            if c["START_RADIUS"] != 3.0:
                vc.SPACE_PATCH["START_RADIUS"] = c["START_RADIUS"]
            try:
                tabs[key] = (cuts, evidence.pair_table(seq, cuts, pp))
            finally:
                vc.SPACE_PATCH.clear()
        cuts, table = tabs[key]
        w2 = w.copy()
        w2[cols.index("sink_lo")] *= c["bonus"] * c["w_sink"]
        w2[cols.index("plug")] *= c["bonus"] * c["w_plug"]
        bonus = method.bonus_of(table, {"cols": cols, "w": w2.tolist()})
        cl = [list(x) for x in cuts]
        for point in POINTS:
            m = float(point.split("_")[1]) if point.startswith("fixed_") else par["margin"][point]
            links = linker.agglomerate(seq, cuts, bonus, t_abs=par["t_abs"] * c["t_abs"], margin=m * c["margin"],
                                       floor=par["floor"] * c["floor"], t0=par["t0"] * c["t0"])
            r = score.score_clip(seq, links, cl)
            out[(name, point)] = {"per_return": r["per_return"], "links": r["links"], "impure_rows": r["impure_rows"], "tracked_rows": r["tracked_rows"],
                                  "n_cuts": len(cl)}
    return seq, out, round(time.time() - t0, 1)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    import pickle

    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "tables" else 6
    t0 = time.time()
    RAW = vc.MYCACHE / "r3_raw.pkl"
    if len(sys.argv) > 1 and sys.argv[1] == "tables":       # re-render the tables from the saved raw results
        res = pickle.load(open(RAW, "rb"))
    else:
        res = {}
        with ProcessPoolExecutor(workers) as ex:
            for seq, out, sec in ex.map(run_clip, data.DEV_CLIPS):
                res[seq] = out
                print(seq, sec, "s", flush=True)
        with open(RAW, "wb") as fh:
            pickle.dump(res, fh)
    rnd = random_configs()
    J = {"one_at_a_time": {}, "random": {}}
    L = ["# r3: sensitivity of fuse's held-out result to +-10% threshold changes", "",
         "Nothing refitted; every clip with its fold's numbers. Cells: right / wrong / lost; links right / wrong / unknown; impure rows.", "",
         "| change | " + " | ".join(POINTS) + " | cuts |", "|---|" + "---|" * (len(POINTS) + 1)]
    for name in ONE:
        cells = []
        for point in POINTS:
            p = vc.pool(res[s][(name, point)] for s in data.DEV_CLIPS)
            J["one_at_a_time"].setdefault(name, {})[point] = p
            cells.append(vc.fmt_pool(p).replace(" | ", "; "))
        ncut = sum(res[s][(name, POINTS[0])]["n_cuts"] for s in data.DEV_CLIPS)
        L.append(f"| {name} | " + " | ".join(cells) + f" | {ncut} |")
    L += ["", "## Largest move of one threshold by +-10% (one at a time; rows 'margin +-20%' and 'ALL' excluded)", "",
          "| point | base right / wrong | right range | wrong range | largest right change | largest wrong change |", "|---|---|---|---|---|---|"]
    single = [n for n in ONE if n != "base" and not n.startswith("ALL") and "20%" not in n]
    for point in POINTS:
        b = J["one_at_a_time"]["base"][point]
        rr = {n: J["one_at_a_time"][n][point]["right"] for n in single}
        ww = {n: J["one_at_a_time"][n][point]["wrong"] for n in single}
        nr = max(rr, key=lambda n: abs(rr[n] - b["right"]))
        nw = max(ww, key=lambda n: abs(ww[n] - b["wrong"]))
        L.append(f"| {point} | {b['right']} / {b['wrong']} | {min(rr.values())} - {max(rr.values())} | {min(ww.values())} - {max(ww.values())} | "
                 f"{rr[nr] - b['right']:+d} ({nr}) | {ww[nw] - b['wrong']:+d} ({nw}) |")
    L += ["", f"## {N_RANDOM} random joint draws (margin, floor, t0, bonus each x U[0.9, 1.1], seed 0)", "",
          "| point | base right / wrong | right min / median / max | wrong min / median / max | draws with wrong > base | draws with wrong > 4 | > 7 | impure min / max |",
          "|---|---|---|---|---|---|---|---|"]
    for point in POINTS:
        b = J["one_at_a_time"]["base"][point]
        ps = [vc.pool(res[s][(n, point)] for s in data.DEV_CLIPS) for n in rnd]
        R, W, I = np.array([p["right"] for p in ps]), np.array([p["wrong"] for p in ps]), np.array([p["impure_row_share"] for p in ps])
        J["random"][point] = {"right": R.tolist(), "wrong": W.tolist(), "impure": I.tolist(), "factors": list(rnd.values())}
        L.append(f"| {point} | {b['right']} / {b['wrong']} | {R.min()} / {int(np.median(R))} / {R.max()} | {W.min()} / {int(np.median(W))} / {W.max()} | "
                 f"{int((W > b['wrong']).sum())} | {int((W > 4).sum())} | {int((W > 7).sum())} | {100 * I.min():.2f}% / {100 * I.max():.2f}% |")
    # per clip: which clips move under margin +-10%
    L += ["", "## Per clip, margin -10% / base / +10% (right / wrong / lost)", "", "| clip | " + " | ".join(POINTS[:3]) + " |", "|---|---|---|---|"]
    for s in data.DEV_CLIPS:
        cells = []
        for point in POINTS[:3]:
            v = [vc.counts(res[s][(n, point)]["per_return"]) for n in ("margin -10%", "base", "margin +10%")]
            cells.append("; ".join(f"{x['right']}/{x['wrong']}/{x['lost']}" for x in v))
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    text = "\n".join(L) + "\n"
    (vc.OUT / "r3_sensitivity.md").write_text(text, encoding="utf-8")
    (vc.OUT / "r3_sensitivity.json").write_text(json.dumps(J), encoding="utf-8")
    print(text)
    print("done", round(time.time() - t0, 1), "s")
