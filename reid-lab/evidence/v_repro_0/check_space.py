"""Label-free check of the machinery that hands m_space a cut tracks table: on the UNCUT pieces,
fuse's tables must equal what m_space's own modules give when they are run natively.

    python -I check_space.py native     # m_space's own posscore.position_scores -> cache/check_space_native.pkl
    python -I check_space.py compare    # fuse's evidence.position_scores on cuts=() against it

(two processes, because m_space's modules and fuse's modules share names)"""
import json
import pickle
import sys

sys.dont_write_bytecode = True   # leave no .pyc in m_space's folder
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data  # noqa: E402

OUT = "C:/Research/plugins/tools/reid_lab/work/v_repro_0/cache/check_space_native.pkl"
CLIPS = data.DEV_CLIPS

if sys.argv[1] == "native":
    sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
    import posscore  # m_space's own; reads its cached weights params_cache/excl_<clip>.json (nothing is fitted or written)

    out = {s: posscore.position_scores(s)[["a", "b", "pos_adm", "pos_z", "pos_sink", "pos_p_in", "pos_p_out"]] for s in CLIPS}
    pickle.dump(out, open(OUT, "wb"))
    print("native tables written:", {s: len(v) for s, v in out.items()})
else:
    import numpy as np

    sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
    import evidence
    import prep

    native = pickle.load(open(OUT, "rb"))
    worst = 0.0
    for s in CLIPS:
        pp = json.load(open(prep.WORK + f"m_space/params_cache/excl_{s.split('-')[1]}.json", encoding="utf-8"))
        mine = evidence.position_scores(s, (), pp)
        n = native[s][native[s].pos_adm]
        m = n.merge(mine, on=["a", "b"], how="outer", suffixes=("_n", "_m"), indicator=True)
        only = int((m["_merge"] != "both").sum())
        d = max(float(np.abs(m[f"{c}_n"] - m[f"{c}_m"]).max()) for c in ("pos_z", "pos_sink", "pos_p_in", "pos_p_out"))
        worst = max(worst, d)
        print(s, "admissible pairs", len(n), "| not in both:", only, "| largest difference", f"{d:.2e}")
    print("identical" if worst < 1e-9 else f"largest difference {worst:.3e}")
