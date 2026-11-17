"""v_robust_1 / r3_sensitivity.py -- (LABELS: scoring only) sensitivity of fuse_r1's result to +-10%
changes of its thresholds and hand-set constants. Nothing is refitted and nothing is re-chosen:
every clip keeps the numbers of ITS fold (fuse_r1/params_loco.json: pair model; m_cuts' and m_space's
held-out weights) and its setting, and ONE thing is moved at a time; then all of them together in the
loose and in the strict direction; then random joint draws (numpy default_rng(0), not fuse_r1's seed).

    python -I -B r3_sensitivity.py [workers]   -> cache/r3_raw.pkl, out/r3_sensitivity.md, out/r3_sensitivity.json
    python -I -B r3_sensitivity.py tables      re-render the tables from cache/r3_raw.pkl

Six settings per clip: strict / balanced / recall = the setting rule B chose WITHOUT the clip (the
METHOD files); fixed_strict / fixed_balanced / fixed_recall = the frozen all-13 setting with the
clip's fold weights (METHOD_fixed_*; not held-out for the choice of the setting).

Moved (fuse_r1's own sensitivity.py moved margin, floor, t0, bonus, n_solid, p_cut):
  margin        the setting's margin                         x 0.9 / x 1.1  (and x 0.8 / x 1.2)
  floor         appearance floor 0.3                         0.27 / 0.33
  t0            the "nobody" rival 0.5                       0.45 / 0.55
  t_abs         m_appear's absolute bar 0.5                  0.45 / 0.55
  bonus         both bonus weights relative to appearance    x 0.9 / x 1.1
  w_sink/w_plug the position weight / the plugin weight alone
  n_solid       boxes a piece needs to be solid (5)          4 / 6 (the nearest integers), and 3 / 10
  p_cut         m_cuts' probability level 0.7                0.63 / 0.77 (new cut lists, new tables)
  MAX_SPEED     m_space's speed bound 8 m/s                  7.2 / 8.8
  START_RADIUS  m_space's end-point slack 3 m                2.7 / 3.3
  WCCN_LAMBDA   m_appear's whitening shrinkage 0.3           0.27 / 0.33
  MIN_CLEAN     m_appear's clean rows for a clean prototype (5)   4 / 6
Random draws: 40 x (margin, floor, t0, bonus each x U[0.9, 1.1]); 20 more that also draw t_abs,
n_solid from {4, 5, 6} and p_cut from {0.63, 0.70, 0.77}.
"""
import json
import pickle
import sys
import time
from functools import lru_cache

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import numpy as np

import vr_common as vc
from vr_common import data, evidence, linker2, method, prep

POINTS = ("strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall")
BASE = dict(margin=1.0, floor=1.0, t0=1.0, t_abs=1.0, bonus=1.0, w_sink=1.0, w_plug=1.0, p_cut=0.7, MAX_SPEED=8.0, START_RADIUS=3.0,
            n_solid=5, lam=0.3, min_clean=5)
ONE = {
    "base": {},
    "margin -10%": {"margin": 0.9}, "margin +10%": {"margin": 1.1},
    "floor 0.27": {"floor": 0.9}, "floor 0.33": {"floor": 1.1},
    "nobody (t0) 0.45": {"t0": 0.9}, "nobody (t0) 0.55": {"t0": 1.1},
    "t_abs 0.45": {"t_abs": 0.9}, "t_abs 0.55": {"t_abs": 1.1},
    "bonus -10%": {"bonus": 0.9}, "bonus +10%": {"bonus": 1.1},
    "position weight -10%": {"w_sink": 0.9}, "position weight +10%": {"w_sink": 1.1},
    "plugin weight -10%": {"w_plug": 0.9}, "plugin weight +10%": {"w_plug": 1.1},
    "solid from 4 boxes": {"n_solid": 4}, "solid from 6 boxes": {"n_solid": 6},
    "p_cut 0.63 (more cuts)": {"p_cut": 0.63}, "p_cut 0.77 (fewer cuts)": {"p_cut": 0.77},
    "MAX_SPEED 7.2": {"MAX_SPEED": 7.2}, "MAX_SPEED 8.8": {"MAX_SPEED": 8.8},
    "START_RADIUS 2.7": {"START_RADIUS": 2.7}, "START_RADIUS 3.3": {"START_RADIUS": 3.3},
    "WCCN_LAMBDA 0.27": {"lam": 0.27}, "WCCN_LAMBDA 0.33": {"lam": 0.33},
    "MIN_CLEAN 4": {"min_clean": 4}, "MIN_CLEAN 6": {"min_clean": 6},
    "ALL looser (margin, floor, t0, t_abs -10%; bonus +10%; p_cut 0.77; speed, radius +10%; solid from 4)":
        {"margin": 0.9, "floor": 0.9, "t0": 0.9, "t_abs": 0.9, "bonus": 1.1, "p_cut": 0.77, "MAX_SPEED": 8.8, "START_RADIUS": 3.3, "n_solid": 4},
    "ALL stricter (margin, floor, t0, t_abs +10%; bonus -10%; p_cut 0.63; speed, radius -10%; solid from 6)":
        {"margin": 1.1, "floor": 1.1, "t0": 1.1, "t_abs": 1.1, "bonus": 0.9, "p_cut": 0.63, "MAX_SPEED": 7.2, "START_RADIUS": 2.7, "n_solid": 6},
    # beyond +-10% (not counted in the +-10% ranges)
    "margin -20%": {"margin": 0.8}, "margin +20%": {"margin": 1.2},
    "solid from 3 boxes": {"n_solid": 3}, "solid from 10 boxes": {"n_solid": 10},
    "solid from 2 boxes": {"n_solid": 2}, "solid from 20 boxes": {"n_solid": 20},
}
BEYOND = ("margin -20%", "margin +20%", "solid from 3 boxes", "solid from 10 boxes", "solid from 2 boxes", "solid from 20 boxes")
N_RANDOM, N_RANDOM2 = 40, 20


def random_configs():
    rng = np.random.default_rng(0)
    out = {f"random {i:02d}": dict(zip(("margin", "floor", "t0", "bonus"), [float(x) for x in rng.uniform(0.9, 1.1, 4)])) for i in range(N_RANDOM)}
    for i in range(N_RANDOM2):
        d = dict(zip(("margin", "floor", "t0", "t_abs", "bonus"), [float(x) for x in rng.uniform(0.9, 1.1, 5)]))
        d["n_solid"] = int(rng.choice([4, 5, 6]))
        d["p_cut"] = float(rng.choice([0.63, 0.7, 0.77]))
        out[f"random+ {i:02d}"] = d
    return out


def run_clip(seq: str):
    from reid_lab import score  # LABELS: scoring

    t0 = time.time()
    P = vc.params()
    par = P[seq]
    pp = method.pos_params(seq)
    cols, w = par["pair_model"]["cols"], np.asarray(par["pair_model"]["w"])
    af = prep._import("m_appear", "features")
    orig_protos = af._protos
    out, tabs, cache_links = {}, {}, {}
    configs = {**ONE, **random_configs()}
    for name, change in configs.items():
        c = {**BASE, **change}
        cuts = prep.cuts_for(seq, seq, c["p_cut"])
        cl = [list(x) for x in cuts]
        vc.SPACE_PATCH.clear()
        if c["MAX_SPEED"] != 8.0:
            vc.SPACE_PATCH["MAX_SPEED"] = c["MAX_SPEED"]
        if c["START_RADIUS"] != 3.0:
            vc.SPACE_PATCH["START_RADIUS"] = c["START_RADIUS"]
        try:
            for ns in (c["n_solid"], 0):
                key = (c["p_cut"], c["MAX_SPEED"], c["START_RADIUS"], ns)
                if key not in tabs:
                    tabs[key] = evidence.pair_table(seq, cuts, pp, ns)
        finally:
            vc.SPACE_PATCH.clear()
        core_tab = tabs[(c["p_cut"], c["MAX_SPEED"], c["START_RADIUS"], c["n_solid"])]
        full_tab = tabs[(c["p_cut"], c["MAX_SPEED"], c["START_RADIUS"], 0)]
        w2 = w.copy()
        w2[cols.index("sink_lo")] *= c["bonus"] * c["w_sink"]
        w2[cols.index("plug")] *= c["bonus"] * c["w_plug"]
        model = {"cols": cols, "w": w2.tolist()}
        bonus_core = method.bonus_of(core_tab, model)
        bonus_leaf = method.bonus_of(full_tab, model, only=method.LEAF_COLS)
        patched = c["lam"] != 0.3 or c["min_clean"] != 5
        if patched:
            af._protos = lru_cache(maxsize=None)(lambda s, cu, _l=c["lam"], _m=c["min_clean"]: af.Protos(s, cuts=cu, lam=_l, min_clean=_m))
        try:
            for point in POINTS:
                margin, leaves, _, _ = vc.setting(point, seq)
                ck = (name, margin, leaves)
                if ck not in cache_links:
                    core, leaf = linker2.link(seq, cuts, bonus_core, bonus_leaf if leaves else None, margin=margin * c["margin"],
                                              floor=par["floor"] * c["floor"], t0=par["t0"] * c["t0"], t_abs=par["t_abs"] * c["t_abs"],
                                              n_solid=c["n_solid"], leaves=leaves)
                    r = score.score_clip(seq, core + leaf, cl)
                    cache_links[ck] = {"per_return": r["per_return"], "links": r["links"], "impure_rows": r["impure_rows"], "tracked_rows": r["tracked_rows"],
                                       "n_cuts": len(cl), "link_list": core + leaf}
                out[(name, point)] = cache_links[ck]
        finally:
            af._protos = orig_protos
        if name == "base":       # the unchanged configuration must give the METHOD files' links
            for point in POINTS:
                M = vc.load_method(point)[seq]
                assert sorted(map(tuple, out[(name, point)]["link_list"])) == sorted(map(tuple, M["links"])), (seq, point)
                assert sorted(map(tuple, cl)) == sorted(map(tuple, M["cuts"])), (seq, point)
    for v in out.values():
        v.pop("link_list", None)
    return seq, out, round(time.time() - t0, 1)


def render(res):
    rnd = random_configs()
    J = {"one_at_a_time": {}, "random": {}, "random_plus": {}}
    L = ["# r3: sensitivity of fuse_r1's result to +-10% changes (nothing refitted, nothing re-chosen)", "",
         "Every clip with its fold's numbers. strict / balanced / recall: the setting rule B chose without the clip (METHOD files). "
         "fixed_*: the frozen all-13 setting with fold weights (not held-out for the choice of the setting).",
         "Cells: right / wrong / lost; links right / wrong / unknown; impure rows.", "",
         "| change | " + " | ".join(POINTS) + " | cuts |", "|---|" + "---|" * (len(POINTS) + 1)]
    for name in ONE:
        cells = []
        for point in POINTS:
            p = vc.pool(res[s][(name, point)] for s in data.DEV_CLIPS)
            J["one_at_a_time"].setdefault(name, {})[point] = p
            cells.append(vc.fmt_pool(p).replace(" | ", "; "))
        ncut = sum(res[s][(name, POINTS[0])]["n_cuts"] for s in data.DEV_CLIPS)
        L.append(f"| {name} | " + " | ".join(cells) + f" | {ncut} |")
    L += ["", "## Largest move of one threshold by +-10% (one at a time; the 'ALL' rows and the rows beyond +-10% excluded)", "",
          "| point | base right / wrong | right range | wrong range | largest right change | largest wrong change | changes that raise wrong |", "|---|---|---|---|---|---|---|"]
    single = [n for n in ONE if n != "base" and not n.startswith("ALL") and n not in BEYOND]
    caps = {"strict": 4, "balanced": 7, "recall": 14, "fixed_strict": 4, "fixed_balanced": 7, "fixed_recall": 14}
    for point in POINTS:
        b = J["one_at_a_time"]["base"][point]
        rr = {n: J["one_at_a_time"][n][point]["right"] for n in single}
        ww = {n: J["one_at_a_time"][n][point]["wrong"] for n in single}
        nr = max(rr, key=lambda n: abs(rr[n] - b["right"]))
        nw = max(ww, key=lambda n: abs(ww[n] - b["wrong"]))
        up = [f"{n} ({ww[n]})" for n in single if ww[n] > b["wrong"]]
        J["one_at_a_time"]["_summary_" + point] = {"right_range": [min(rr.values()), max(rr.values())], "wrong_range": [min(ww.values()), max(ww.values())],
                                                   "raise_wrong": up}
        L.append(f"| {point} | {b['right']} / {b['wrong']} | {min(rr.values())} - {max(rr.values())} | {min(ww.values())} - {max(ww.values())} | "
                 f"{rr[nr] - b['right']:+d} ({nr}) | {ww[nw] - b['wrong']:+d} ({nw}) | {'; '.join(up) or 'none'} |")
    for title, key, names in ((f"{N_RANDOM} random joint draws (margin, floor, t0, bonus each x U[0.9, 1.1], seed 0)", "random", [n for n in rnd if n.startswith("random ")]),
                              (f"{N_RANDOM2} more draws that also move t_abs, n_solid in {{4, 5, 6}} and p_cut in {{0.63, 0.70, 0.77}}", "random_plus",
                               [n for n in rnd if n.startswith("random+")])):
        L += ["", f"## {title}", "",
              "| point | base right / wrong | right min / median / max | wrong min / median / max | draws by wrong count | draws over the limit | impure min / max |",
              "|---|---|---|---|---|---|---|"]
        for point in POINTS:
            b = J["one_at_a_time"]["base"][point]
            ps = [vc.pool(res[s][(n, point)] for s in data.DEV_CLIPS) for n in names]
            R, W, I = np.array([p["right"] for p in ps]), np.array([p["wrong"] for p in ps]), np.array([p["impure_row_share"] for p in ps])
            cnt = {int(k): int((W == k).sum()) for k in sorted(set(W.tolist()))}
            J[key][point] = {"right": R.tolist(), "wrong": W.tolist(), "impure": I.tolist(), "factors": [rnd[n] for n in names], "wrong_counts": cnt,
                             "over_limit": int((W > caps[point]).sum())}
            L.append(f"| {point} | {b['right']} / {b['wrong']} | {R.min()} / {int(np.median(R))} / {R.max()} | {W.min()} / {int(np.median(W))} / {W.max()} | "
                     f"{', '.join(f'{v}x {k}' for k, v in cnt.items())} | {int((W > caps[point]).sum())} of {len(names)} | {100 * I.min():.2f}% / {100 * I.max():.2f}% |")
    L += ["", "## Per clip, margin -10% / base / +10% (right / wrong / lost)", "", "| clip | " + " | ".join(POINTS[:4]) + " |", "|---|---|---|---|---|"]
    for s in data.DEV_CLIPS:
        cells = []
        for point in POINTS[:4]:
            v = [vc.counts(res[s][(n, point)]["per_return"]) for n in ("margin -10%", "base", "margin +10%")]
            cells.append("; ".join(f"{x['right']}/{x['wrong']}/{x['lost']}" for x in v))
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    # which clips carry extra wrong returns under any single +-10% change
    L += ["", "## Wrong returns per clip under each single +-10% change (only changes where a clip's wrong count differs from base)", ""]
    for point in POINTS:
        for n in single:
            diff = []
            for s in data.DEV_CLIPS:
                a, b = vc.counts(res[s][(n, point)]["per_return"])["wrong"], vc.counts(res[s][("base", point)]["per_return"])["wrong"]
                if a != b:
                    diff.append(f"{s} {b} -> {a}")
            if diff:
                L.append(f"- {point}, {n}: " + ", ".join(diff))
    text = "\n".join(L) + "\n"
    (vc.OUT / "r3_sensitivity.md").write_text(text, encoding="utf-8")
    (vc.OUT / "r3_sensitivity.json").write_text(json.dumps(J), encoding="utf-8")
    return text


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    RAW = vc.MYCACHE / "r3_raw.pkl"
    if len(sys.argv) > 1 and sys.argv[1] == "tables":
        res = pickle.load(open(RAW, "rb"))
    else:
        workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
        res = {}
        with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex:
            for seq, out, sec in ex.map(run_clip, data.DEV_CLIPS):
                res[seq] = out
                print(seq, sec, "s", flush=True)
        if len(res) != len(data.DEV_CLIPS):
            raise SystemExit("a clip is missing")
        with open(RAW, "wb") as fh:
            pickle.dump(res, fh)
    print(render(res))
    print("done", round(time.time() - t0, 1), "s")
