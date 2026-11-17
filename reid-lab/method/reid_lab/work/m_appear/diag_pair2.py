"""DIAGNOSIS: piece-level view of all allowed partners (either time order) of a piece, with rival margins."""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from core import ctx
from features import protos, candidate_ids
import fitlabels
seq, kind, p = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = ctx(seq); pr = protos(seq); ids = candidate_ids(seq); ident = fitlabels.piece_identity(seq); P = c.P.set_index("piece")
X = np.stack([pr.get(kind)[q] for q in ids]); S = X @ X.T
n = len(ids); pos = {q: i for i, q in enumerate(ids)}
O = np.array([[bool(c.frames_of[a] & c.frames_of[b]) if a != b else False for b in ids] for a in ids])
team = np.array([pr.ateam[q] for q in ids]); can = (team[:, None] == team[None, :]) & ~O; np.fill_diagonal(can, False)
def nm(q): return f"{q}[{ident.get(q, ('?',))[0]}|{P.loc[q,'first']}-{P.loc[q,'last']}|{P.loc[q,'n_clean']}c]"
g = pos[p]
order = [j for j in np.argsort(-S[g]) if can[g, j]][:6]
for h in order:
    rg = [(S[g, k], ids[k]) for k in range(n) if can[g, k] and O[h, k]]
    rh = [(S[h, k], ids[k]) for k in range(n) if can[h, k] and O[g, k]]
    bh = max((S[h, k], ids[k]) for k in range(n) if can[h, k])
    print(f"{nm(p)} ~ {nm(ids[h])}: s={S[g,h]:.3f}; partner's best = {nm(bh[1])} {bh[0]:.3f}; rival for {p}: {max(rg, default=(np.nan, None))[0]:.3f} {nm(max(rg)[1]) if rg else ''}; rival for partner: {max(rh, default=(np.nan, None))[0]:.3f} {nm(max(rh)[1]) if rh else ''}")
