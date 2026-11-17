"""DIAGNOSIS ONLY: the true switches between players of DIFFERENT teams (annotation): does the
appearance agree that the kit changes? Team score of a window = cos(window, mean embedding of gbf's
"left" pieces) - cos(window, mean embedding of gbf's "right" pieces).

    python -I diag_crossteam.py [p0]     (after make_methods.py)
"""
import json
import sys

sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np
import pandas as pd
from reid_lab import data, score
import features as F
import truth as T

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 800)
p0 = sys.argv[1] if len(sys.argv) > 1 else "0.7"
cuts = pd.read_csv(HERE + f"methods/heldout_p{p0}_cuts.csv")
rows = []
for seq in data.DEV_CLIPS:
    raw = json.load(open(score.TRAIN / seq / "Labels-GameState.json", encoding="utf-8"))
    att = {}
    for a in raw["annotations"]:
        if "track_id" in a and isinstance(a.get("attributes"), dict):
            att.setdefault(int(a["track_id"]), (a["attributes"].get("role"), a["attributes"].get("team")))
    c = F.ctx(seq)
    proto = {}
    for team in ("left", "right"):
        v = [p.E[p.clean].sum(axis=0) for p in c.P.values() if p.team == team and p.role == "player" and p.clean.sum() >= 5]
        proto[team] = np.sum(v, axis=0) / np.linalg.norm(np.sum(v, axis=0))
    hit = set(cuts[(cuts.seq == seq) & (cuts.sw_id >= 0)].sw_id.tolist())
    for k, s in enumerate(T.switches(seq)):
        (ra, ta), (rb, tb) = att[s["from_ref"]], att[s["to_ref"]]
        if not (ra == rb == "player"):
            continue
        p = c.P[s["piece"]]
        ia = int(np.searchsorted(p.frame, s["last_a"], side="right"))    # rows up to last_a
        ib = int(np.searchsorted(p.frame, s["first_b"]))                 # rows from first_b
        # the run of person a ends at ia, the run of b starts at ib; windows inside those runs
        B = c.desc(p.pid, 0, ia, "last", small=True)[0]
        A = c.desc(p.pid, ib, len(p), "first", small=True)[0]
        ts = lambda v: float(v @ proto["left"] - v @ proto["right"]) if v is not None else np.nan  # noqa: E731
        rows.append({"seq": seq[-3:], "piece": s["piece"], "a>b": f"{s['from_ref']}>{s['to_ref']}", "teams": f"{ta}>{tb}", "cross": ta != tb, "gbf_team": p.team,
                     "rows_a": s["rows_a"], "rows_b": s["rows_b"], "last_a": s["last_a"], "first_b": s["first_b"], "hit": k in hit,
                     "cos": round(float(B @ A), 3) if A is not None and B is not None else np.nan, "team_score_before": round(ts(B), 3), "team_score_after": round(ts(A), 3)})
df = pd.DataFrame(rows)
df["kit_flips"] = np.sign(df.team_score_before) != np.sign(df.team_score_after)
print(df[df.cross].sort_values(["hit", "cos"]).to_string(index=False))
print("cross-team switches:", int(df.cross.sum()), "; appearance agrees the kit flips:", int((df.cross & df.kit_flips).sum()), "; of those hit:", int((df.cross & df.kit_flips & df.hit).sum()))
print("cross-team by annotation but the kit does NOT flip:", int((df.cross & ~df.kit_flips).sum()), "; of those hit:", int((df.cross & ~df.kit_flips & df.hit).sum()))
print("same-team switches:", int((~df.cross).sum()), "; kit flips (should be ~0):", int((~df.cross & df.kit_flips).sum()))
