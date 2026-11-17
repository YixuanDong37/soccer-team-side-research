"""Probe: per-clip right/wrong of some settings in the one-fold cache (fit-clip view)."""
import pickle, sys
from collections import Counter
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data
import explore_engine as loco
fold = sys.argv[1]
r = pickle.load(open(f"C:/Research/plugins/tools/reid_lab/work/fuse/cache/onefold_{fold}.pkl", "rb"))
want = [("app", 0.5, 0.30), ("app", 0.5, 0.25), ("app", 0.6, 0.40), ("app", 0.8, 0.40), ("pos", 0.5, 0.40), ("pos", 0.5, 0.25), ("posplug", 0.5, 0.25)]
for v, t, m in want:
    gi = loco.GRID.index((t, m))
    line = f"{v:8s} {t} {m}: "
    for s in data.DEV_CLIPS:
        c = Counter(r["res"][(v, gi, s)]["per_return"])
        line += f"{s[-3:]}{'*' if s == fold else ''} {c['right']}/{c['wrong']}/{c['lost']}  "
    print(line)
