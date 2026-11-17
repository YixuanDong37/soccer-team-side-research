import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import pandas as pd
pd.set_option("display.width", 250)
def own(c, rid):
    return sorted([p for p in c.P.values() if p["ident"] == rid], key=lambda p: p["first"])
def table(seq, pa, pb):
    c = ctx(seq)
    A, B = own(c, pa), own(c, pb)
    ps = A + B
    names = [f"{p['piece']}(p{p['ident']},{p['first']}-{p['last']},c{p['n_clean']})" for p in ps]
    M = np.array([[np.nan if (p["desc_any"] is None or q["desc_any"] is None) else float(p["desc_any"] @ q["desc_any"]) for q in ps] for p in ps])
    print(seq, "persons", pa, c.att[pa], pb, c.att[pb])
    print(pd.DataFrame(M.round(3), index=names, columns=[n.split("(")[0] for n in names]).to_string())
    for p in ps:
        print("   ", p["piece"], "team", p["team"], p["vteam"], "role", p["role"], "end_xy", p["end_xy"], "start_xy", p["start_xy"], "sides", p["start_side"], p["end_side"], "jersey", p["jersey"], p["jersey_reads"])
table("SNGS-162", 14, 12)
table("SNGS-162", 9, 18)
table("SNGS-162", 21, 13)
table("SNGS-162", 5, 8)
