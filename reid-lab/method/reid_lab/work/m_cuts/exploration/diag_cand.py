"""DIAGNOSIS ONLY: candidate boundaries (label-free, features.candidates) joined with truth
(diag_switches.json + local identity from score.clip).  Writes diag_cand.csv.

    python -I diag_cand.py
"""
import json
import sys
import time

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np
import pandas as pd
from reid_lab import data, score
import features as F

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
TOL = 5      # rows of the piece a cut may leave on the wrong side of a true switch
pd.set_option("display.width", 280); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 800)


def truth(seq, cand, SW):
    """sw: the boundary lies inside a true switch interval; loc: the majority annotated person of
    the 10 matched rows before differs from the 10 after (None: < 3 matched rows on a side);
    minor: rows of the smaller side of the switch."""
    c = score.clip(seq)
    m = pd.DataFrame({"frame": c.frames, "ref": c.refs, "piece": c.piece})
    mp = {pid: (g.frame.to_numpy(), g.ref.to_numpy()) for pid, g in m[m.piece >= 0].sort_values("frame").groupby("piece")}
    sw, loc, minor = [], [], []
    for r in cand.itertuples():
        sws = [s for s in SW[seq] if s["piece"] == r.piece]
        pf = F.piece_table(seq)[r.piece].frame
        # a cut at f1 is a hit when at most TOL rows of the piece end up on the wrong side
        hit = [s for s in sws if ((pf >= r.f1) & (pf <= s["last_a"])).sum() <= TOL and ((pf >= s["first_b"]) & (pf <= r.f0)).sum() <= TOL]
        sw.append(bool(hit))
        minor.append(min(hit[0]["rows_a"], hit[0]["rows_b"]) if hit else 0)
        mf, mr = mp.get(r.piece, (np.zeros(0, int), np.zeros(0, int)))
        b = mr[mf <= r.f0][-10:]
        a = mr[mf >= r.f1][:10]
        loc.append(None if len(a) < 3 or len(b) < 3 else bool(np.bincount(a).argmax() != np.bincount(b).argmax()))
    return sw, loc, minor


if __name__ == "__main__":
    SW = json.load(open(HERE + "diag_switches.json"))
    out = []
    for seq in data.DEV_CLIPS:
        t0 = time.time()
        cand = F.candidates(seq).copy()
        cand["sw"], cand["loc"], cand["minor"] = truth(seq, cand, SW)
        got = set()
        for r in cand[cand.sw].itertuples():
            pf = F.piece_table(seq)[r.piece].frame
            for k, s in enumerate(SW[seq]):
                if s["piece"] == r.piece and ((pf >= r.f1) & (pf <= s["last_a"])).sum() <= TOL and ((pf >= s["first_b"]) & (pf <= r.f0)).sum() <= TOL:
                    got.add(k)
        missing = [s for k, s in enumerate(SW[seq]) if k not in got]
        print(seq, len(cand), "cands; kinds", cand.kind.value_counts().to_dict(), "sw", int(cand.sw.sum()),
              "switches without candidate:", len(missing), f"{time.time()-t0:.1f}s")
        for s in missing:
            print("     no candidate:", s)
        out.append(cand)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(HERE + "diag_cand.csv", index=False)
    print(len(df), int(df.sw.sum()))
    print(df.groupby("kind").sw.agg(["size", "sum"]))
