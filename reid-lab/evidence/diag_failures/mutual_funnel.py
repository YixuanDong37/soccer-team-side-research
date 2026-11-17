"""Where the mutual rule loses the scored returns: for each return whose row B sits in the
person's own piece, the pairs (earlier own piece of the person, B's piece) and what the rule
did with them. DIAGNOSIS ONLY (rule 5c). -> out/mutual_funnel.csv

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I mutual_funnel.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ctx, mutual_status, pair_gate  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
ret = pd.read_csv(OUT / "returns.csv")
ORDER = ["linked", "margin < 0.05", "no runner-up", "not mutual best"]
rows = []
for r in ret[ret.status == "scored"].itertuples():
    c = ctx(r.clip)
    P = c.P
    rid = int(r.ref_id)
    row = {"clip": r.clip, "ret": r.ret}
    if r.b_kind != "own":
        row["funnel"] = f"B row in a {r.b_kind} piece (no link of known pieces helps)"
        rows.append(row)
        continue
    b = P[int(r.piece_b)]
    earlier = [q for q in P.values() if q["ident"] == rid and q["piece"] != b["piece"] and q["first"] < b["first"]]
    if not earlier:
        row["funnel"] = "no earlier own piece of the person"
        rows.append(row)
        continue
    cands = [q for q in earlier if (q["piece"], b["piece"]) in c.cand]
    row["earlier_own_pieces"] = len(earlier)
    row["earlier_own_candidates"] = len(cands)
    if not cands:
        near = max(earlier, key=lambda q: q["last"])
        gate = pair_gate(near, b)
        row["funnel"] = "candidate generation"
        row["gate_nearest_own"] = "; ".join(gate)
        row["nearest_own"] = near["piece"]
        # the single deciding gate, in a fixed order, to count returns once
        for key in ("overlap in time", "gap > 10 s", "team unverified", "clean rows < 5", "role", "verified teams differ", "out of reach"):
            if any(key in g for g in gate):
                row["gate_first"] = key
                break
        row["gates_n"] = len({("team" if "team" in g else "clean" if "clean" in g else g) for g in gate})
        rows.append(row)
        continue
    best = None
    for q in cands:
        ms = mutual_status(c.cand, q["piece"], b["piece"])
        if best is None or ORDER.index(ms["status"]) < ORDER.index(best[1]["status"]):
            best = (q, ms)
    q, ms = best
    row["funnel"] = "decision: " + ms["status"]
    row["best_true_earlier"] = q["piece"]
    row["true_cos"] = round(c.cand[(q["piece"], b["piece"])], 4)
    row["rank_true_for_b"], row["rank_b_for_true"] = ms["rank_of_a_for_b"], ms["rank_of_b_for_a"]
    row["margin_b_side"] = None if ms["margin_b_side"] is None else round(ms["margin_b_side"], 4)
    row["margin_a_side"] = None if ms["margin_a_side"] is None else round(ms["margin_a_side"], 4)
    # who competes: the strongest other candidate on each side, and whose piece it is
    for side, key, mine in (("b", lambda k: k[1] == b["piece"], lambda k: k[0]), ("a", lambda k: k[0] == q["piece"], lambda k: k[1])):
        other = [(v, mine(k)) for k, v in c.cand.items() if key(k) and mine(k) not in (q["piece"], b["piece"])]
        if other:
            v, pid = max(other)
            who = P[pid]["ident_any"]
            row[f"rival_{side}_side"] = pid
            row[f"rival_{side}_cos"] = round(v, 4)
            row[f"rival_{side}_is"] = "same person" if who == rid else ("nobody annotated" if who is None else "another person")
    rows.append(row)
df = pd.DataFrame(rows)
df = df.rename(columns={k: "fn_" + k for k in df.columns if k not in ("clip", "ret", "funnel")})
df.to_csv(OUT / "mutual_funnel.csv", index=False)
m = ret.merge(df, on=["clip", "ret"], how="left")
sc = m[m.status == "scored"]
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
print(pd.crosstab(sc.funnel, sc.out_mutual, margins=True).to_string())
print(pd.crosstab(sc.funnel, sc.ceiling_class, margins=True).to_string())
g = sc[sc.funnel == "candidate generation"]
print("first gate:"); print(g.fn_gate_first.value_counts().to_string()); print("all gates:"); print(g.fn_gate_nearest_own.value_counts().to_string())
print("number of distinct gates failing:", g.fn_gates_n.value_counts().to_dict())
d = sc[sc.funnel.fillna("").str.startswith("decision") & (sc.funnel != "decision: linked")]
cols = ["true_cos", "rank_true_for_b", "rank_b_for_true", "margin_b_side", "margin_a_side", "rival_b_side", "rival_b_cos", "rival_b_is", "rival_a_side", "rival_a_cos", "rival_a_is", "earlier_own_candidates"]
print(d[["clip", "ret", "ref_id", "funnel", "out_mutual"] + ["fn_" + k for k in cols]].to_string())
print(pd.crosstab(d.funnel, [d.fn_rival_b_is.fillna("none"), d.fn_rival_a_is.fillna("none")]).to_string())
