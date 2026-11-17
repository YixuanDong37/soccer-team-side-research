"""DIAGNOSIS ONLY: how often does a piece restart exactly where another piece just ended (takeover),
and is that a true switch?  Labels via diag_switches.json + score.clip (local identity)."""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np, pandas as pd
from reid_lab import data, score
import features as F
pd.set_option("display.width", 280); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 800)
SW = json.load(open("diag_switches.json"))
rows = []
for seq in data.DEV_CLIPS:
    c = score.clip(seq)
    m = pd.DataFrame({"frame": c.frames, "ref": c.refs, "piece": c.piece})
    mp = {pid: (g.frame.to_numpy(), g.ref.to_numpy()) for pid, g in m[m.piece >= 0].sort_values("frame").groupby("piece")}
    P = F.piece_table(seq)
    ends = {}   # last frame -> pieces
    starts = {}
    for q in P.values():
        ends.setdefault(int(q.frame[-1]), []).append(q)
        starts.setdefault(int(q.frame[0]), []).append(q)
    for p in P.values():
        f = p.frame
        sws = [s for s in SW[seq] if s["piece"] == p.pid]
        mf, mr = mp.get(p.pid, (np.zeros(0, int), np.zeros(0, int)))
        for i in range(len(f) - 1):
            f0, f1 = int(f[i]), int(f[i + 1])
            best_d = (0.0, None, None)
            for T in (1, 2, 3):
                for q in ends.get(f1 - T, []):
                    if q.pid == p.pid: continue
                    v = F.iou(q.box[-1], p.box[i + 1])
                    if v > best_d[0]: best_d = (v, q.pid, T)
            best_r = (0.0, None, None)
            for T in (1, 2, 3):
                for r in starts.get(f0 + T, []):
                    if r.pid == p.pid: continue
                    v = F.iou(r.box[0], p.box[i])
                    if v > best_r[0]: best_r = (v, r.pid, T)
            if best_d[0] < 0.3 and best_r[0] < 0.3: continue
            sw = any(s["last_a"] <= f0 and f1 <= s["first_b"] for s in sws)
            b = mr[mf <= f0][-10:]; a = mr[mf >= f1][:10]
            loc = None
            if len(a) >= 3 and len(b) >= 3:
                loc = bool(np.bincount(a).argmax() != np.bincount(b).argmax())
            rows.append({"seq": seq[-3:], "piece": p.pid, "f0": f0, "f1": f1, "gap": f1 - f0 - 1, "iou_self": round(F.iou(p.box[i], p.box[i + 1]), 2),
                         "d_iou": round(best_d[0], 2), "d_piece": best_d[1], "d_T": best_d[2], "r_iou": round(best_r[0], 2), "r_piece": best_r[1], "r_T": best_r[2],
                         "sw": sw, "loc": loc})
df = pd.DataFrame(rows)
df.to_csv("diag_takeover.csv", index=False)
print(len(df))
for name, sel in (("donor iou>=0.5", df.d_iou >= 0.5), ("donor>=0.5 & gap>=1", (df.d_iou >= 0.5) & (df.gap >= 1)), ("donor>=0.5 & gap>=2", (df.d_iou >= 0.5) & (df.gap >= 2)),
                  ("donor>=0.5 & gap>=5", (df.d_iou >= 0.5) & (df.gap >= 5)), ("donor>=0.5 & iou_self<0.2", (df.d_iou >= 0.5) & (df.iou_self < 0.2)),
                  ("recv>=0.5", df.r_iou >= 0.5), ("recv>=0.5 & gap>=2", (df.r_iou >= 0.5) & (df.gap >= 2)), ("recv>=0.5 & iou_self<0.2", (df.r_iou >= 0.5) & (df.iou_self < 0.2)),
                  ("donor>=.3&iou_self<.2", (df.d_iou >= 0.3) & (df.iou_self < 0.2)), ("recv>=.3&iou_self<.2", (df.r_iou >= 0.3) & (df.iou_self < 0.2))):
    d = df[sel]
    print(f"{name}: n={len(d)} sw={int(d.sw.sum())} loc_true={int((d['loc'] == True).sum())} loc_false={int((d['loc'] == False).sum())} loc_unknown={int(d['loc'].isna().sum())}")
print(df[(df.d_iou >= 0.5) & (df.iou_self < 0.2)].to_string())
