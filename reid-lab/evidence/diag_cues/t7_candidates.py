"""(7) The cues together: for a later piece B, which earlier pieces survive each gate, and does the
appearance ranking then find B's person (diagnosis, in-sample; NOT a method result).

Candidates of B: EVERY earlier piece of the clip (last frame < B's first frame), with or without
a known identity, as a method would see them. Labels are used only to say which candidate is
B's person. Gates, cumulative:
  role   gbf role labels equal
  team   gbf team labels equal, or one missing
  speed  anchors (median of 5 placed rows) satisfy d <= 8 m/s * T + 3 m (passes when no anchor)
  side   not (A's last box touches the left edge and B's first box the right edge, or the reverse)
  path   the straight path between the anchors is >= 30 px inside the picture on <= 30 % of the
         gap frames (passes when it cannot be computed)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import camview as V  # noqa: E402
import common as C  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
data = C.data
res: dict = {}
W, H = C.W, C.H


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def side_of(r):
    s = [k for k, c in (("left", r.x1 <= data.EDGE_PX), ("right", r.x2 >= W - 1 - data.EDGE_PX), ("top", r.y1 <= data.EDGE_PX), ("bottom", r.y2 >= H - 1 - data.EDGE_PX)) if c]
    return "+".join(s) if s else "none"


rows = []
for seq in C.CLIPS:
    t = data.tracks(seq)
    emb = data.embeddings(seq)
    has = ~np.isnan(emb[:, 0])
    lab = t.team.to_numpy(object)
    cen = emb - emb[has].mean(0)
    for name in ("left", "right"):
        sel = has & (lab == name)
        if sel.any():
            cen[lab == name] = emb[lab == name] - emb[sel].mean(0)
    cen = unit(cen)
    P = C.piece_ident(seq).reset_index(drop=True)
    tt = t[t.track_id >= 0].sort_values(["track_id", "frame"])
    info = {}
    for pid, g in tt.groupby("track_id"):
        idx = g.index.to_numpy()
        idx = idx[has[idx]]
        pl = g[g.pitch_usable & g.in_pitch & g.pitch_x.notna()]
        d = {"s_side": side_of(g.iloc[0]), "e_side": side_of(g.iloc[-1]), "raw": unit(emb[idx].mean(0)) if len(idx) else None, "tc": unit(cen[idx].mean(0)) if len(idx) else None}
        if len(pl):
            h, tl = pl.head(5), pl.tail(5)
            d.update({"sx": float(h.pitch_x.median()), "sy": float(h.pitch_y.median()), "sf": float(h.frame.median()),
                      "ex": float(tl.pitch_x.median()), "ey": float(tl.pitch_y.median()), "ef": float(tl.frame.median())})  # fmt: skip
        info[int(pid)] = d
    first, last = P["first"].to_numpy(), P["last"].to_numpy()
    rid = P.ref_id.to_numpy()
    known = P.ref_id.notna().to_numpy()
    plgk = P.role_ref.isin(["player", "goalkeeper"]).to_numpy()
    for j in range(len(P)):
        if not (known[j] and plgk[j]) or first[j] < 10:
            continue
        bj = info[int(P.piece.iat[j])]
        if bj["raw"] is None:
            continue
        earlier = np.where(last < first[j])[0]
        returning = bool(((rid[earlier] == rid[j]) & known[earlier]).any())
        for i in earlier:
            ai = info[int(P.piece.iat[i])]
            if ai["raw"] is None:
                continue
            row = {"seq": seq, "b": int(P.piece.iat[j]), "a": int(P.piece.iat[i]), "b_returning": returning, "b_edge": bool(P.starts_at_edge.iat[j]), "a_edge": bool(P.ends_at_edge.iat[i]),
                   "label": "true" if (known[i] and rid[i] == rid[j]) else ("unknown" if not known[i] else "wrong"), "gap": int(first[j] - last[i] - 1),
                   "role_ok": bool(P.role.iat[i] == P.role.iat[j]), "team_ok": bool(pd.isna(P.team.iat[i]) or pd.isna(P.team.iat[j]) or P.team.iat[i] == P.team.iat[j]),
                   "cos_raw": float(ai["raw"] @ bj["raw"]), "cos_tc": float(ai["tc"] @ bj["tc"]), "a_rows": int(P.rows.iat[i]), "b_rows": int(P.rows.iat[j])}  # fmt: skip
            row["side_ok"] = not ({ai["e_side"], bj["s_side"]} == {"left", "right"})
            speed_ok, path_ok = True, True
            if "ex" in ai and "sx" in bj:
                T = (bj["sf"] - ai["ef"]) / C.FPS
                dd = float(np.hypot(bj["sx"] - ai["ex"], bj["sy"] - ai["ey"]))
                speed_ok = dd <= 8 * max(T, 0) + 3
                row["d"], row["T"] = dd, T
                f0, f1 = int(round(ai["ef"])), int(round(bj["sf"]))
                gf = np.arange(f0 + 1, f1)
                if len(gf) and row["role_ok"] and row["team_ok"] and speed_ok and row["side_ok"]:  # only where the earlier gates pass (cost)
                    w_ = (gf - f0) / (f1 - f0)
                    lv, _ = V.person_visible(seq, gf, ai["ex"] + w_ * (bj["sx"] - ai["ex"]), ai["ey"] + w_ * (bj["sy"] - ai["ey"]), margin_px=30)
                    if np.isfinite(lv).any():
                        row["line30"] = float(np.nanmean(lv))
                        path_ok = row["line30"] <= 0.3
            row["speed_ok"], row["path_ok"] = bool(speed_ok), bool(path_ok)
            rows.append(row)
CP = pd.DataFrame(rows)
CP.to_parquet(C.OUT / "t7_candidates.parquet")
nb = CP.groupby(["seq", "b"]).ngroups
print("later pieces B (identity known, player/GK, starting at frame >= 10, with earlier pieces):", nb, "; candidate pairs:", len(CP))
print("labels:", CP.label.value_counts().to_dict())

GATES = [("no gate", []), ("role", ["role_ok"]), ("+ team", ["role_ok", "team_ok"]), ("+ speed", ["role_ok", "team_ok", "speed_ok"]),
         ("+ side", ["role_ok", "team_ok", "speed_ok", "side_ok"]), ("+ path", ["role_ok", "team_ok", "speed_ok", "side_ok", "path_ok"])]  # fmt: skip


def gate_table(D, title):
    print(f"\n--- {title} ---")
    out = []
    Bs = D.groupby(["seq", "b"]).agg(returning=("b_returning", "first"))
    n_ret, n_new = int(Bs.returning.sum()), int((~Bs.returning).sum())
    for name, cols in GATES:
        g = D if not cols else D[D[cols].all(axis=1)]
        ret = g[g.b_returning]
        new = g[~g.b_returning]
        per = ret.groupby(["seq", "b"])
        has_true = per.label.apply(lambda s: (s == "true").any())
        top_raw = per.apply(lambda x: x.loc[x.cos_raw.idxmax(), "label"] == "true", include_groups=False)
        top_tc = per.apply(lambda x: x.loc[x.cos_tc.idxmax(), "label"] == "true", include_groups=False)
        out.append({"gates": name, "returning B": n_ret, "true candidate kept": int(has_true.sum()), "kept share": has_true.sum() / n_ret,
                    "candidates per B p50": float(per.size().median()), "true:wrong+unknown": f"1:{(ret.label != 'true').sum() / max((ret.label == 'true').sum(), 1):.1f}",
                    "top-1 is the person (raw cos)": int(top_raw.sum()), "share raw": top_raw.sum() / n_ret,
                    "top-1 is the person (team-centred)": int(top_tc.sum()), "share tc": top_tc.sum() / n_ret,
                    "new B": n_new, "new B left without candidate": n_new - new.groupby(["seq", "b"]).ngroups})  # fmt: skip
    T = pd.DataFrame(out)
    print(T.round(3).to_string())
    return T


T1 = gate_table(CP, "all later pieces B")
T1.to_csv(C.OUT / "t7_gates_all.csv", index=False)
T2 = gate_table(CP[CP.b_edge], "B starts at an image edge (re-entry candidates)")
T2.to_csv(C.OUT / "t7_gates_edge.csv", index=False)
T3 = gate_table(CP[CP.b_edge & (CP.gap >= 10)], "B starts at an image edge, candidates with a gap >= 10 frames")
T3.to_csv(C.OUT / "t7_gates_edge_gap10.csv", index=False)
res["gates_all"] = T1.round(4).to_dict("records")
res["gates_edge"] = T2.round(4).to_dict("records")

# prior: how often is a new piece a returning person at all?
Bs = CP.groupby(["seq", "b"]).agg(returning=("b_returning", "first"), edge=("b_edge", "first"))
res["prior"] = {"B_pieces": len(Bs), "returning_share": round(float(Bs.returning.mean()), 3), "B_at_edge": int(Bs.edge.sum()),
                "returning_share_at_edge": round(float(Bs.returning[Bs.edge].mean()), 3), "returning_share_not_at_edge": round(float(Bs.returning[~Bs.edge].mean()), 3)}  # fmt: skip
print("\nprior that a later piece is a person seen before:", res["prior"])

# open-set: best candidate score of returning B (true top-1) against new B, after the gates up to 'side'
print("\n=== accept the top-1 candidate when cosine >= bar and it leads the runner-up by >= margin (gates role+team+speed+side) ===")
G = CP[CP[["role_ok", "team_ok", "speed_ok", "side_ok"]].all(axis=1)]


def decide(D, col, bar, margin, mutual):
    ok = wrong = unj = 0
    best_later = None
    if mutual:
        best_later = D.loc[D.groupby(["seq", "a"])[col].idxmax()].set_index(["seq", "a"]).b.to_dict()
    for (seq, b), g in D.groupby(["seq", "b"]):
        s = g.sort_values(col, ascending=False)
        top = s.iloc[0]
        lead = top[col] - (s[col].iat[1] if len(s) > 1 else -1.0)
        if top[col] < bar or lead < margin:
            continue
        if mutual and best_later.get((seq, top.a)) != b:
            continue
        if top.label == "true":
            ok += 1
        elif top.label == "unknown":
            unj += 1
        else:
            wrong += 1
    return ok, wrong, unj


tabs = []
for col, bars in (("cos_raw", (0.0, 0.90, 0.93, 0.95)), ("cos_tc", (-1.0, 0.5, 0.6, 0.7, 0.8))):
    for bar in bars:
        for margin in (0.0, 0.03, 0.05, 0.1, 0.15):
            for mutual in (False, True):
                ok, wrong, unj = decide(G, col, bar, margin, mutual)
                tabs.append({"score": col, "bar": bar, "margin": margin, "mutual best": mutual, "right links": ok, "wrong links": wrong, "to a piece of unknown identity": unj,
                             "precision (judged)": ok / max(ok + wrong, 1)})  # fmt: skip
DT = pd.DataFrame(tabs)
print(DT.round(3).to_string())
DT.to_csv(C.OUT / "t7_decision_grid.csv", index=False)
res["decision_grid"] = DT.round(4).to_dict("records")
n_ret = int(Bs.returning.sum())
print("returning B in total (upper bound of right links):", n_ret)
res["returning_B"] = n_ret
(C.OUT / "t7_candidates.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
