"""DIAGNOSIS ONLY: the returns scored "wrong" under split (no links): which piece, what is in it."""
import sys
from collections import Counter

sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np
import pandas as pd
from reid_lab import data, score
import features as F
import truth as T

for seq in data.DEV_CLIPS:
    c = score.clip(seq)
    res = score.score_clip(seq)
    k = 0
    for r, o in zip(c.returns, res["per_return"]):
        if o != "wrong":
            continue
        a, b = r["a"], r["b"]
        pb = int(c.piece[b])
        sel = (c.piece == pb)
        fr, rf = c.frames[sel], c.refs[sel]
        order = np.argsort(fr)
        fr, rf = fr[order], rf[order]
        runs = []
        for f, x in zip(fr, rf):
            if runs and runs[-1][0] == x:
                runs[-1][2] = f; runs[-1][3] += 1
            else:
                runs.append([int(x), int(f), int(f), 1])
        p = F.piece_table(seq)[pb]
        print(f"{seq} return ref {r['ref_id']} last {r['last']} first {r['first']} gap {r['gap']}: A piece {int(c.piece[a])} @ {int(c.frames[a])}, B piece {pb} @ {int(c.frames[b])}")
        print(f"    piece {pb}: frames {int(p.frame[0])}..{int(p.frame[-1])} rows {len(p)}; matched runs (ref, first, last, n): {runs}")
        gaps = np.diff(p.frame) - 1
        big = np.where(gaps >= 2)[0]
        print("    gaps>=2 (f0,f1):", [(int(p.frame[i]), int(p.frame[i + 1])) for i in big])
        print("    true switches:", [s for s in T.switches(seq) if s["piece"] == pb])
        cand = F.candidates(seq)
        cc = cand[cand.piece == pb]
        print(cc[["f0", "f1", "gap", "kind", "cos", "alt_d", "alt_r", "d_m", "dt", "nb", "na", "rows_before", "rows_after"]].round(3).to_string())
