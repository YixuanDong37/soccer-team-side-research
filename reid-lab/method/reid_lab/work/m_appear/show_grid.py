"""IN-SAMPLE table of a grid cache: right / wrong returns / wrong links pooled over the 13 clips, t_abs (rows) x margin (columns)."""
import sys, os, json; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
import loco
family = sys.argv[1]; grid, results = loco.load(family)
for kind in grid["kind"]:
    print(f"\n{family} kind={kind}: right/wrong/wrong-links  (rows t_abs, columns margin)")
    print("t_abs  " + "".join(f"{m:>11.2f}" for m in grid["margin"]))
    for t in grid["t_abs"]:
        cells = []
        for m in grid["margin"]:
            p = loco.pool(results[json.dumps({"kind": kind, "margin": m, "t_abs": t}, sort_keys=True)], data.DEV_CLIPS)
            cells.append(f"{p['right']:3d}/{p['wrong']:2d}/{p['links_wrong']:2d}")
        print(f"{t:5.2f}  " + "".join(f"{c:>11s}" for c in cells))
