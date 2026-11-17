"""DIAGNOSIS ONLY: what kind of people do pieces switch between (annotation's role and team), how
long are the gaps at the switch, and which kinds the held-out cuts catch.

    python -I diag_switch_types.py [p0]     (after make_methods.py)
"""
import json
import sys

sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np
import pandas as pd
from gbf.evaluate import gsr
from reid_lab import data, score
import features as F
import truth as T

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
p0 = sys.argv[1] if len(sys.argv) > 1 else "0.7"
cuts = pd.read_csv(HERE + f"methods/heldout_p{p0}_cuts.csv")
rows = []
for seq in data.DEV_CLIPS:
    ref = gsr.load_reference(score.TRAIN / seq / "Labels-GameState.json")
    raw = json.load(open(score.TRAIN / seq / "Labels-GameState.json", encoding="utf-8"))
    att = {}
    for a in raw["annotations"]:
        if "track_id" in a and isinstance(a.get("attributes"), dict):
            att.setdefault(int(a["track_id"]), (a["attributes"].get("role"), a["attributes"].get("team")))
    who = ref.people.groupby("ref_id").agg(role=("role", "first"))
    who["team"] = [att.get(int(i), (None, None))[1] for i in who.index]
    assert all(att.get(int(i), (None,))[0] == r for i, r in zip(who.index, who.role)), "ref_id is not the annotation's track_id"
    hit = set(cuts[(cuts.seq == seq) & (cuts.sw_id >= 0)].sw_id.tolist())
    P = F.piece_table(seq)
    for k, s in enumerate(T.switches(seq)):
        a, b = who.loc[s["from_ref"]], who.loc[s["to_ref"]]
        pf = P[s["piece"]].frame
        seg = pf[(pf >= s["last_a"]) & (pf <= s["first_b"])]
        gap = int((np.diff(seg) - 1).max()) if len(seg) > 1 else 0
        kind = "same team" if (a.role == b.role == "player" and a.team == b.team) else ("other team" if a.role == b.role == "player" else f"{a.role}/{b.role}")
        rows.append({"seq": seq, "kind": kind, "gap": gap, "minor": min(s["rows_a"], s["rows_b"]), "hit": k in hit})
df = pd.DataFrame(rows)
print(df.groupby("kind").agg(n=("hit", "size"), hit=("hit", "sum"), minor_rows=("minor", "sum")))
df["gapbin"] = pd.cut(df.gap, [0, 1, 2, 5, 10, 25, 100, 1000], right=False)
print(df.groupby("gapbin", observed=False).agg(n=("hit", "size"), hit=("hit", "sum"), minor_rows=("minor", "sum")))
df["size"] = pd.cut(df.minor, [0, 10, 20, 50, 1000], right=False)
print(df.groupby("size", observed=False).agg(n=("hit", "size"), hit=("hit", "sum"), minor_rows=("minor", "sum")))
print("rows on the smaller side: all", int(df.minor.sum()), "hit", int(df[df.hit].minor.sum()))
