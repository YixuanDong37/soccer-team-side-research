"""Probe: timings and cut-set stability across LOCO parameter sets (label-free)."""
import sys, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import features as cutf

t0 = time.time()
folds = json.load(open(cutf.LOCO_PARAMS_FILE, encoding="utf-8"))["folds"]
print({k: [round(x, 3) for x in v["w"]] + [round(v["b"], 3)] for k, v in folds.items()})
for seq in data.DEV_CLIPS[:3]:
    t1 = time.time()
    own = sorted(map(tuple, cutf.cuts_heldout(seq, 0.7)))
    print(seq, "own heldout cuts", len(own), round(time.time() - t1, 1), "s")
    diff = {}
    for c in data.DEV_CLIPS:
        if c == seq:
            continue
        other = sorted(map(tuple, cutf.cuts(seq, cutf.heldout_params(c, 0.7))))
        diff[c] = len(set(own) ^ set(other))
    print("   symmetric difference vs params of other folds:", diff, round(time.time() - t1, 1), "s")
print(round(time.time() - t0, 1), "s")
