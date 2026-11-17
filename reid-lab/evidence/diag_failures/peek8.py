import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
c = ctx("SNGS-162")
print("prototypes", data.refine("SNGS-162")["teams"]["prototypes"])
for pid in (8, 23, 116, 11, 20, 84, 95, 4, 96, 18, 104, 106, 27, 93, 42, 35, 114, 100031, 200031, 81):
    e = c.team_entry.get(pid, {})
    p = c.P[pid]
    print(pid, "person", p["ident_any"], c.att.get(p["ident_any"]), "rows", p["rows"], "lab", e.get("lab"), e.get("verdict"), e.get("nearer_team"), "samples", e.get("samples"), "role", p["role"], "team", p["team"])
for rid in (5, 8, 17, 2):
    g = c.people_by[rid]
    f = g.frame.to_numpy(); brk = np.where(np.diff(f) > 1)[0]
    print(rid, c.att[rid], list(zip(np.concatenate([[f[0]], f[brk+1]]).tolist(), np.concatenate([f[brk], [f[-1]]]).tolist())))
    for fr in (np.concatenate([[f[0]], f[brk+1]]).tolist() + np.concatenate([f[brk], [f[-1]]]).tolist()):
        r = g[g.frame == fr].iloc[0]; print("   ", fr, "box", r.x1, r.y1, r.x2, r.y2, "pitch", round(r.pitch_x, 1), round(r.pitch_y, 1))
