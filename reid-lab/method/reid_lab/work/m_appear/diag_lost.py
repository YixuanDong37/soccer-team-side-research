"""DIAGNOSIS (labels): the returns a method loses, and what the pair table says about A -> B."""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
import numpy as np, pandas as pd
from agglo import agglomerate
from features import pair_features
from core import ctx
import fitlabels
kind, t_abs, margin = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = []
for s in data.DEV_CLIPS:
    links = agglomerate(s, kind=kind, t_abs=t_abs, margin=margin, mutual=True)
    res = score.score_clip(s, links); c = score.clip(s); F = pair_features(s).set_index(["a", "b"]); P = ctx(s).P.set_index("piece"); ident = fitlabels.piece_identity(s)
    Fb = pair_features(s)
    for r, o in zip(c.returns, res["per_return"]):
        if o != "lost": continue
        pa, pb = int(c.piece[r["a"]]), int(c.piece[r["b"]])
        row = F.loc[(pa, pb)] if (pa, pb) in F.index else None
        # earlier same-person pieces for pb
        same = [p for p, v in ident.items() if v[0] == r["ref_id"] and P.loc[p, "last"] < P.loc[pb, "first"]]
        best_same = max([F.loc[(p, pb)][f"cos_{kind}"] for p in same if (p, pb) in F.index], default=np.nan)
        cb = Fb[Fb.b == pb]; best_any = cb[f"cos_{kind}"].max() if len(cb) else np.nan
        rows.append(dict(seq=s, ref=r["ref_id"], gap=r["gap"], pa=pa, pb=pb, rows_a=P.loc[pa, "rows"], rows_b=P.loc[pb, "rows"], cl_a=P.loc[pa, "n_clean"], cl_b=P.loc[pb, "n_clean"],
                         id_a=ident.get(pa, (None,))[0], id_b=ident.get(pb, (None,))[0], cand=row is not None,
                         cos=None if row is None else round(row[f"cos_{kind}"], 3), rk_f=None if row is None else int(row[f"rank_fwd_{kind}"]), rk_b=None if row is None else int(row[f"rank_bwd_{kind}"]),
                         riv_f=None if row is None else round(row[f"rival_fwd_{kind}"], 2), riv_b=None if row is None else round(row[f"rival_bwd_{kind}"], 2),
                         best_same=round(best_same, 3), best_any=round(best_any, 3), n_same=len(same)))
d = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(d.to_string())
print("lost", len(d), "| B piece < 5 rows:", (d.rows_b < 5).sum(), "| A piece < 5 rows:", (d.rows_a < 5).sum(), "| either <5:", ((d.rows_a < 5) | (d.rows_b < 5)).sum(),
      "| no earlier same-person piece with identity:", (d.n_same == 0).sum(), "| B identity is another person:", (d.id_b.notna() & (d.id_b != d.ref)).sum())
