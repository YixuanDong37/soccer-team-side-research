"""DIAGNOSIS (labels via diag_switches.json): before/after appearance cosine at row boundaries."""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
SW = json.load(open("diag_switches.json"))

def boundary_cos(E, ok, K, m=3):
    """E (n,256) rows of a piece in frame order, ok = usable rows. For boundary i (between i and i+1):
    cos(mean of last K ok rows <= i, mean of first K ok rows >= i+1); nan when a side has < m."""
    n = len(E)
    idx = np.where(ok)[0]
    out = np.full(n - 1, np.nan); nb = np.zeros(n - 1, int); na = np.zeros(n - 1, int)
    if len(idx) < 2 * m:
        return out, nb, na
    C = np.vstack([np.zeros((1, E.shape[1])), np.cumsum(E[idx], axis=0)])
    # number of ok rows with index <= i
    cnt = np.cumsum(ok)
    for i in range(n - 1):
        c = cnt[i]
        b0 = max(0, c - K); a1 = min(len(idx), c + K)
        if c - b0 < m or a1 - c < m:
            continue
        mb = C[c] - C[b0]; ma = C[a1] - C[c]
        out[i] = mb @ ma / (np.linalg.norm(mb) * np.linalg.norm(ma))
        nb[i] = c - b0; na[i] = a1 - c
    return out, nb, na

rows = []
for K in (10, 20, 40):
  for use in ("clean", "any"):
    pos, neg_gap, neg_all = [], [], []
    miss = 0
    for seq in data.DEV_CLIPS:
        t = data.tracks(seq); E = data.embeddings(seq); cl = data.clean_rows(seq)
        has = ~np.isnan(E[:, 0])
        for pid, idx in t[t.track_id >= 0].sort_values("frame").groupby("track_id").indices.items():
            pass
        tt = t[t.track_id >= 0].sort_values(["track_id", "frame"])
        for pid, g in tt.groupby("track_id"):
            ii = g.index.to_numpy(); f = g.frame.to_numpy()
            if len(ii) < 6: continue
            ok = cl[ii] if use == "clean" else has[ii]
            s, nb, na = boundary_cos(np.nan_to_num(E[ii]), ok, K)
            gap = np.diff(f) - 1
            sws = [x for x in SW[seq] if x["piece"] == pid]
            inpos = np.zeros(len(s), bool)
            for x in sws:
                sel = (f[:-1] >= x["last_a"]) & (f[1:] <= x["first_b"])
                inpos |= (f[:-1] >= x["last_a"] - 15) & (f[1:] <= x["first_b"] + 15)
                v = s[sel]
                if np.all(np.isnan(v)): miss += 1
                else: pos.append(np.nanmin(v))
            neg_all += s[~inpos & ~np.isnan(s)].tolist()
            neg_gap += s[~inpos & ~np.isnan(s) & (gap >= 5)].tolist()
    pos, neg_gap, neg_all = map(np.array, (pos, neg_gap, neg_all))
    q = lambda a: np.round(np.quantile(a, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95]), 3)
    print(f"K={K} {use}: pos n={len(pos)} (no stat {miss}) q={q(pos)} | neg gap>=5 n={len(neg_gap)} q={q(neg_gap)} | neg all n={len(neg_all)} q={q(neg_all)}")
    for th in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        print(f"    th {th}: pos below {np.mean(pos < th):.2f} ({(pos<th).sum()}) neg_gap below {np.mean(neg_gap < th):.4f} ({(neg_gap<th).sum()}) neg_all below {np.mean(neg_all < th):.4f} ({(neg_all<th).sum()})")
