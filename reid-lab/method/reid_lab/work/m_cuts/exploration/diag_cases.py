"""DIAGNOSIS ONLY (labels through reid_lab.score.clip): per true switch, what cues exist?"""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np, pandas as pd
from reid_lab import data, score
import features as F
pd.set_option("display.width", 280); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)
SW = json.load(open("diag_switches.json"))
rows = []
for seq in data.DEV_CLIPS:
    c = score.clip(seq)
    m = pd.DataFrame({"frame": c.frames, "ref": c.refs, "piece": c.piece})
    P = F.piece_table(seq)
    for s in SW[seq]:
        p = P[s["piece"]]
        f = p.frame
        sel = np.where((f[:-1] >= s["last_a"]) & (f[1:] <= s["first_b"]))[0]
        gaps = np.diff(f) - 1
        i = sel[np.argmax(gaps[sel])]
        cs = F.boundary_cos(p)
        # pitch jump
        pb = np.where(p.placed[: i + 1])[0][-3:]; pa = np.where(p.placed[i + 1:])[0][:3] + i + 1
        if len(pb) and len(pa):
            d = float(np.hypot(*(np.median(p.xy[pa], axis=0) - np.median(p.xy[pb], axis=0))))
            dt = (f[pa[0]] - f[pb[-1]]) / 25
        else:
            d, dt = np.nan, np.nan
        # donor: piece of to_ref just before first_b (other than this piece)
        mb = m[(m.ref == s["to_ref"]) & (m.frame < s["first_b"]) & (m.piece != s["piece"]) & (m.piece >= 0)]
        donor = mb[mb.frame >= s["first_b"] - 250].piece.value_counts()
        dn = None
        if len(donor):
            dp = int(mb.sort_values("frame").piece.iloc[-1]); dn = (dp, int(mb[mb.piece == dp].frame.max()), int((mb.piece == dp).sum()), int(P[dp].frame[-1]))
        ma = m[(m.ref == s["from_ref"]) & (m.frame > s["last_a"]) & (m.piece != s["piece"]) & (m.piece >= 0)]
        rc = None
        if len(ma):
            rp = int(ma.sort_values("frame").piece.iloc[0]); rc = (rp, int(ma[ma.piece == rp].frame.min()), int((ma.piece == rp).sum()), int(P[rp].frame[0]))
        rows.append({"seq": seq[-3:], "piece": s["piece"], "a>b": f"{s['from_ref']}>{s['to_ref']}", "last_a": s["last_a"], "first_b": s["first_b"], "na": s["rows_a"], "nb": s["rows_b"],
                     "f0": int(f[i]), "f1": int(f[i + 1]), "gap": int(gaps[i]), "cos": round(float(cs[i]), 3), "mincos": round(float(np.nanmin(cs[sel])) if not np.all(np.isnan(cs[sel])) else np.nan, 3),
                     "d_m": round(d, 1), "dt": dt, "excess": round(d - 12 * dt, 1) if d == d else np.nan,
                     "team": p.team, "role": p.role, "donor(piece,lastmatch,n,pieceend)": dn, "receiver(piece,firstmatch,n,piecestart)": rc})
df = pd.DataFrame(rows)
print(df.to_string())
