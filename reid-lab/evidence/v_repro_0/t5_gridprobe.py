"""Probe (labels for scoring; fit clips of ONE fold only, the held-out clip is not touched):
a wider grid of linker settings, to choose the grid that the LOCO run will search.
usage: python -I t5_gridprobe.py FOLD VARIANT [scale]"""
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
from reid_lab import data  # noqa: E402

FLOORS = (None, 0.3, 0.4, 0.5, 0.6)
T_ABS = (0.5, 0.8, 1.0, 1.2)
MARGINS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def work(job):
    fold, variant, scale, s = job
    import evidence
    import linker
    import explore_engine as loco
    import prep
    from reid_lab import score
    r = pickle.load(open(f"C:/Research/plugins/tools/reid_lab/work/v_repro_0/cache/onefold_{fold}.pkl", "rb"))
    cuts = prep.norm_cuts(r["cuts"][s])
    tab = evidence.pair_table(s, cuts, loco.pos_params(fold))
    model = r["models"][variant]
    bonus = {k: v * scale for k, v in loco.bonus_of(tab, model).items()} if model else None
    out = {}
    for fl in FLOORS:
        for t in T_ABS:
            for m in MARGINS:
                links = linker.agglomerate(s, cuts, bonus, t_abs=t, margin=m, floor=fl)
                x = score.score_clip(s, links, r["cuts"][s])
                out[(fl, t, m)] = (Counter(x["per_return"]), x["links"], x["impure_rows"], x["tracked_rows"])
    return s, out


if __name__ == "__main__":
    fold, variant = sys.argv[1], sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    fit = [s for s in data.DEV_CLIPS if s != fold]
    with ProcessPoolExecutor(6) as ex:
        res = dict(ex.map(work, [(fold, variant, scale, s) for s in fit]))
    pickle.dump(res, open(f"C:/Research/plugins/tools/reid_lab/work/v_repro_0/cache/gridprobe_{fold}_{variant}_{scale}.pkl", "wb"))
    print(f"fold {fold} variant {variant} bonus x{scale}: 12 fit clips; in brackets: without the fit clip with the most wrong returns + wrong links")
    for fl in FLOORS:
        for t in T_ABS:
            for m in MARGINS:
                c, lk, imp, rows = Counter(), Counter(), 0, 0
                per = []
                for s in fit:
                    pr, l, i, n = res[s][(fl, t, m)]
                    c.update(pr)
                    lk.update({k: v for k, v in l.items() if k != "n"})
                    imp += i
                    rows += n
                    per.append((pr["wrong"] + l["wrong"], pr["right"], pr["wrong"], l["wrong"]))
                worst = max(per)
                print(f"floor {str(fl):4s} t_abs {t:.1f} margin {m:.1f}: {c['right']:3d}/{c['wrong']:2d}/{c['lost']:3d} links {lk['right']:3d}/{lk['wrong']:2d}/{lk['unknown']:3d} impure {imp / rows:.4f}"
                      f"   [{c['right'] - worst[1]:3d}/{c['wrong'] - worst[2]:2d} links wrong {lk['wrong'] - worst[3]:2d}]")
