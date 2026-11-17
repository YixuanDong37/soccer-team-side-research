"""Probe (labels for scoring; fit clips of ONE fold only, the held-out clip is not touched):
the one-parameter family that the LOCO run searches (margin), for a variant / floor / t0.
usage: python -I t6_family.py FOLD VARIANT FLOOR T0      (FLOOR, T0: number or 'none')"""
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data  # noqa: E402

MARGINS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)


def work(job):
    fold, variant, floor, t0, s = job
    import evidence
    import linker
    import explore_engine as loco
    import prep
    from reid_lab import score
    r = pickle.load(open(f"C:/Research/plugins/tools/reid_lab/work/fuse/cache/onefold_{fold}.pkl", "rb"))
    cuts = prep.norm_cuts(r["cuts"][s])
    tab = evidence.pair_table(s, cuts, loco.pos_params(fold))
    model = r["models"][variant]
    bonus = loco.bonus_of(tab, model) if model else None
    out = {}
    for m in MARGINS:
        links = linker.agglomerate(s, cuts, bonus, t_abs=0.5, margin=m, floor=floor, t0=t0)
        x = score.score_clip(s, links, r["cuts"][s])
        out[m] = (Counter(x["per_return"]), x["links"], x["impure_rows"], x["tracked_rows"])
    return s, out


if __name__ == "__main__":
    fold, variant = sys.argv[1], sys.argv[2]
    floor = None if sys.argv[3] == "none" else float(sys.argv[3])
    t0 = None if sys.argv[4] == "none" else float(sys.argv[4])
    fit = [s for s in data.DEV_CLIPS if s != fold]
    with ProcessPoolExecutor(6) as ex:
        res = dict(ex.map(work, [(fold, variant, floor, t0, s) for s in fit]))
    print(f"fold {fold} variant {variant} floor {floor} t0 {t0}: 12 fit clips; J = right - lam * (wrong + wrong links)")
    for m in MARGINS:
        c, lk, imp, rows = Counter(), Counter(), 0, 0
        for s in fit:
            pr, l, i, n = res[s][m]
            c.update(pr)
            lk.update({k: v for k, v in l.items() if k != "n"})
            imp += i
            rows += n
        pen = c["wrong"] + lk["wrong"]
        print(f"margin {m:.2f}: {c['right']:3d}/{c['wrong']:2d}/{c['lost']:3d} links {lk['right']:3d}/{lk['wrong']:2d}/{lk['unknown']:3d} impure {imp / rows:.4f}"
              f"   J10 {c['right'] - 10 * pen:5d}  J3 {c['right'] - 3 * pen:4d}  J1 {c['right'] - pen:4d}")
