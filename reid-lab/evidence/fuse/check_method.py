"""Check (no labels read here; it only compares link lists): the label-free method.py reproduces, from
params_loco.json alone, exactly the links that fit_loco.py scored for the chosen margins."""
import pickle
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data  # noqa: E402
import method  # noqa: E402

raw = pickle.load(open("C:/Research/plugins/tools/reid_lab/work/fuse/cache/loco_raw.pkl", "rb"))["folds"]
params = method.load_params()
bad = 0
for mode in ("heldout", "frozen"):
    for point in method.POINTS:
        m = method.build(mode, point)
        for seq in data.DEV_CLIPS:
            key = seq if mode == "heldout" else method.ALL
            stored = raw[key]["res"][("fuse", params[key]["margin"][point], seq)]["links"]
            same = sorted(map(tuple, stored)) == sorted(map(tuple, m["clips"][seq]["links"])) and raw[key]["cuts"][seq] == m["clips"][seq]["cuts"]
            bad += not same
            if not same:
                print("DIFFERENT", mode, point, seq)
        print(mode, point, "ok" if not bad else "", flush=True)
print("method.py reproduces the scored links" if not bad else f"{bad} differences")
