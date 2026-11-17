"""ORACLE / DIAGNOSIS ONLY. For every scored return: where does the position model (weights
fitted without the clip: posscore out-of-fold scores) put a right earlier piece among the
candidates of the returning piece? Also the same for plain distance and for the residual
alone. Numbers here describe the evidence, they are not method results.

    python -I oracle_returns_rank.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data, score  # noqa: E402
import features  # noqa: E402
import labels  # noqa: E402
import linker  # noqa: E402
import posscore  # noqa: E402

pd.set_option("display.width", 280)
pd.set_option("display.max_rows", 500)
rows = []
for seq in data.DEV_CLIPS:
    c = score.clip(seq)
    weak = labels.identity_map(seq, 1)
    p = features.pair_features(seq)
    s = posscore.position_scores(seq)
    q = p.merge(s, on=["a", "b"])
    q = q[q.pos_adm]
    for r in c.returns:
        if r["a"] is None or r["b"] is None:
            continue
        pb = int(c.piece[r["b"]])
        if pb < 0 or int(c.piece[r["a"]]) < 0:
            continue
        cand = q[q.b == pb]
        ok = cand[cand.a.map(weak) == r["ref_id"]]
        rec = dict(seq=seq, ref=r["ref_id"], gap=r["gap"], pb=pb, n_cand=len(cand), has_right=len(ok) > 0)
        for name, col, asc in (("model", "pos_z", False), ("d", "d", True), ("r", "r_adv_cv", True)):
            if len(ok):
                best = ok[col].max() if not asc else ok[col].min()
                rec[f"rank_{name}"] = int(((cand[col] > best) if not asc else (cand[col] < best)).sum()) + 1
            else:
                rec[f"rank_{name}"] = -1
        rec["p_in"] = float(ok.pos_p_in.max()) if len(ok) else np.nan
        rec["sink"] = float(ok.pos_sink.max()) if len(ok) else np.nan
        rows.append(rec)
df = pd.DataFrame(rows)
n = len(df)
out = {"returns": n, "with_admissible_right_candidate": int(df.has_right.sum()), "median_candidates": float(df.n_cand.median())}
print(f"scored returns {n}; with an admissible right earlier piece {int(df.has_right.sum())}; candidates per returning piece: median {df.n_cand.median():.0f}, mean {df.n_cand.mean():.1f}")
for name in ("model", "r", "d"):
    col = df[f"rank_{name}"]
    line = {f"top{k}": int(((col > 0) & (col <= k)).sum()) for k in (1, 2, 3, 5)}
    out[name] = line
    print(f"  ranked by {name:5s}: right piece is top-1 for {line['top1']}, top-2 {line['top2']}, top-3 {line['top3']}, top-5 {line['top5']} of {n}")
for lo, hi in ((0, 50), (50, 125), (125, 250), (250, 800)):
    g = df[(df.gap >= lo) & (df.gap < hi)]
    print(f"  gap [{lo},{hi}) frames: n={len(g)}  model top-1 {int((g.rank_model == 1).sum())}  top-3 {int(((g.rank_model > 0) & (g.rank_model <= 3)).sum())}  | distance top-1 {int((g.rank_d == 1).sum())}  | candidates median {g.n_cand.median():.0f}")
    out[f"gap_{lo}_{hi}"] = {"n": len(g), "model_top1": int((g.rank_model == 1).sum()), "model_top3": int(((g.rank_model > 0) & (g.rank_model <= 3)).sum()), "d_top1": int((g.rank_d == 1).sum())}
Path("oracle_returns_rank.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
df.to_csv("oracle_returns_rank.csv", index=False)
