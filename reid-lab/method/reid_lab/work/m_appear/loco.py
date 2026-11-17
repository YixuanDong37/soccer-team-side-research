"""Leave-one-clip-out selection of a method's parameters from a grid cache (grid.py), and the pooled
held-out result.

For every dev clip: on the other 12 clips pick the setting with the highest

    utility = right - lam * (extra wrong returns over gbf-now + wrong links)

(ties: the larger margin, then the larger t_abs, then the other parameters in grid order), apply it
unchanged to the held-out clip, pool the 13 held-out clips. lam is the price of one wrong identity in
recovered returns: the operating points of the trade-off table differ only in lam.

    python -I loco.py FAMILY [--fix name=value ...]   -> out/FAMILY[...]-lamX.method.json + .score.json, table on stdout
"""
import json
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reid_lab import data, score  # noqa: E402

LAMS = (1, 2, 3, 5, 10)
SPLIT_WRONG = None


def load(family):
    """A grid cache; "a+b" merges several caches into one family with the extra parameter "family"."""
    if "+" in family:
        grid, results = None, {}
        names = family.split("+")
        for name in names:
            g, r = load(name)
            grid = grid or dict(g, family=names)
            for k, v in r.items():
                results[json.dumps(dict(json.loads(k), family=name), sort_keys=True)] = v
        return grid, results
    with open(os.path.join(HERE, "cache", f"{family}.json"), encoding="utf-8") as f:
        c = json.load(f)
    return c["grid"], {k: v for k, v in c["results"].items()}


def split_wrong():
    """Wrong returns of gbf-now (no links) per clip: the part of 'wrong' no linking method causes."""
    global SPLIT_WRONG
    if SPLIT_WRONG is None:
        SPLIT_WRONG = {s: score.score_clip(s, [])["returns"]["wrong"] for s in data.DEV_CLIPS}
    return SPLIT_WRONG


def pool(per, clips):
    out = {"right": 0, "wrong": 0, "lost": 0, "links": 0, "links_right": 0, "links_wrong": 0, "links_unknown": 0, "impure": 0, "rows": 0}
    for s in clips:
        r = per[s]
        for k in ("right", "wrong", "lost"):
            out[k] += r["returns"][k]
        out["links"] += r["link_score"]["n"]
        for k in ("right", "wrong", "unknown"):
            out[f"links_{k}"] += r["link_score"][k]
        out["impure"] += r["impure_rows"]
        out["rows"] += r["tracked_rows"]
    out["impure_row_share"] = round(out["impure"] / out["rows"], 5)
    return out


def utility(p, lam, base_wrong):
    return p["right"] - lam * ((p["wrong"] - base_wrong) + p["links_wrong"])


def order_key(params, grid):
    """Tie-break: more cautious first (larger margin, larger t_abs), then grid order of the rest."""
    k = []
    for name in ("margin", "t_abs"):
        if name in params:
            k.append(-params[name])
    for name in grid:
        if name not in ("margin", "t_abs"):
            k.append(grid[name].index(params[name]))
    return tuple(k)


def choose(results, grid, clips, lam, fix=None):
    sw = split_wrong()
    base = sum(sw[s] for s in clips)
    best = None
    for k, per in results.items():
        params = json.loads(k)
        if fix and any(params.get(n) != v for n, v in fix.items()):
            continue
        u = utility(pool(per, clips), lam, base)
        cand = (-u, order_key(params, grid), k)
        if best is None or cand < best:
            best = cand
    return best[2], -best[0]


def loco(family, lam, fix=None):
    grid, results = load(family)
    method, chosen = {}, {}
    for fit, held in data.loco_folds():
        k, u = choose(results, grid, fit, lam, fix)
        method[held] = {"links": results[k][held]["links"]}
        chosen[held] = json.loads(k)
    k_all, u_all = choose(results, grid, data.DEV_CLIPS, lam, fix)
    insample = pool(results[k_all], data.DEV_CLIPS)
    return method, chosen, json.loads(k_all), insample


def fmt(p):
    return (f"right {p['right']:3d} wrong {p['wrong']:2d} lost {p['lost']:3d} | links r/w/u {p['links_right']:3d}/{p['links_wrong']:2d}/{p['links_unknown']:3d} "
            f"| impure {p['impure_row_share']:.4f}")


def main():
    family = sys.argv[1]
    fix = {}
    if "--fix" in sys.argv:
        for kv in sys.argv[sys.argv.index("--fix") + 1:]:
            n, v = kv.split("=")
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            fix[n] = v
    tag = family + "".join(f"-{n}_{v}" for n, v in fix.items())
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    print(f"family {tag}: LOCO held-out (headline) and in-sample (secondary), by lam")
    for lam in LAMS:
        method, chosen, k_all, insample = loco(family, lam, fix)
        res = score.evaluate(method)
        p = res["pooled"]
        p["impure_row_share"] = p["impure_row_share"]
        name = f"{tag}-lam{lam}"
        with open(os.path.join(HERE, "out", f"{name}.method.json"), "w", encoding="utf-8") as f:
            json.dump({"name": f"m_appear {name} (LOCO held-out links)", "clips": method, "chosen_per_held_out_clip": chosen,
                       "fit_on_all_13": k_all}, f)
        with open(os.path.join(HERE, "out", f"{name}.score.json"), "w", encoding="utf-8") as f:
            json.dump({"name": name, "held_out": res, "in_sample_fit_on_all_13": {"params": k_all, "pooled": insample}}, f, indent=1)
        vals = {}
        for c in chosen.values():
            vals.setdefault(json.dumps(c, sort_keys=True), 0)
            vals[json.dumps(c, sort_keys=True)] += 1
        print(f"lam={lam:2d} HELD-OUT {fmt(p)}")
        print(f"        in-sample {fmt(insample)}  params {k_all}")
        print(f"        chosen per fold: {vals}")


if __name__ == "__main__":
    main()
