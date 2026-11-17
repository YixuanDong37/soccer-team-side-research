"""(2) Position: gbf pitch_x / pitch_y against the annotation's bbox_pitch (diagnosis; uses labels).

Part A, matched rows: the deviation |gbf ground point - annotated ground point| overall, per clip,
by distance from the camera, by image edge, by calib_status and box height, and split into a
camera part (gbf's camera applied to the ANNOTATED foot pixel vs the annotated ground point) and
a foot-point part (gbf's own foot pixel vs the annotated foot pixel, both through gbf's camera).

Part B, returns: displacement between the exit and the re-entry position against the gap time,
in the annotation (what people really do) and in gbf's rows A and B (what a method sees), and how
a gate  d <= v * T + e  behaves on true returns and on wrong candidate pairs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import common as C  # noqa: E402
from gbf.geometry import pixel_to_ground  # noqa: E402
from gbf.sngs_import import cameras_by_frame  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
data = C.data
res: dict = {}
Q = (0.5, 0.75, 0.9, 0.95, 0.99)


def qrow(name, x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"set": name, "n": 0}
    return {"set": name, "n": len(x), **{f"p{int(q * 100)}": float(np.quantile(x, q)) for q in Q}, "share>2m": float((x > 2).mean()), "share>5m": float((x > 5).mean())}


# ------------------------------------------------------------------ part A
rows_all = []
conv = []
for seq in C.CLIPS:
    t = data.tracks(seq)
    m = C.matched(seq).copy()
    for c in ("pitch_x", "pitch_y", "pitch_usable", "in_pitch", "calib_status", "x1", "y1", "x2", "y2", "foot_u", "foot_v", "role"):
        m[c] = t[c].to_numpy()[m.trow.to_numpy()]
    m["seq"] = seq
    cams = cameras_by_frame(data.calib(seq), usable_only=False)
    cal = data.calib(seq).set_index("frame_index")
    # annotated foot pixel (1080p bottom middle) in gbf's 720p pixel grid, through gbf's camera
    au = ((m.x1_ref + m.x2_ref) / 2 + 0.5) / C.SCALE - 0.5
    av = (m.y2_ref + 0.5) / C.SCALE - 0.5
    gx = np.full(len(m), np.nan)
    gy = np.full(len(m), np.nan)
    cd = np.full(len(m), np.nan)
    fr = m.frame.to_numpy()
    for f in np.unique(fr):
        cam = cams.get(int(f))
        if cam is None:
            continue
        idx = np.where(fr == f)[0]
        pts, ok = pixel_to_ground(cam, np.stack([au.to_numpy()[idx], av.to_numpy()[idx]], axis=1))
        gx[idx] = np.where(ok, pts[:, 0], np.nan)
        gy[idx] = np.where(ok, pts[:, 1], np.nan)
        cd[idx] = np.sqrt((m.pitch_x_ref.to_numpy()[idx] - cam.C[0]) ** 2 + (m.pitch_y_ref.to_numpy()[idx] - cam.C[1]) ** 2 + cam.C[2] ** 2)
    m["ann_foot_gx"], m["ann_foot_gy"], m["cam_dist"] = gx, gy, cd
    m["rep_err"] = cal.rep_err.reindex(fr).to_numpy()
    m["calib_age"] = cal.calib_age.reindex(fr).to_numpy()
    rows_all.append(m)
    ok = m.pitch_x.notna() & m.pitch_x_ref.notna()
    e = {}
    for sx in (1, -1):
        for sy in (1, -1):
            e[f"x*{sx:+d},y*{sy:+d}"] = float(np.median(np.hypot(sx * m.pitch_x[ok] - m.pitch_x_ref[ok], sy * m.pitch_y[ok] - m.pitch_y_ref[ok])))
    conv.append({"seq": seq, **e})
R = pd.concat(rows_all, ignore_index=True)
cv = pd.DataFrame(conv)
print("=== coordinate convention: median deviation (m) under sign flips of gbf's x / y ===")
print(cv.round(2).to_string())
cv.to_csv(C.OUT / "t2_convention.csv", index=False)

R["dx"] = R.pitch_x - R.pitch_x_ref
R["dy"] = R.pitch_y - R.pitch_y_ref
R["err"] = np.hypot(R.dx, R.dy)
R["err_cam"] = np.hypot(R.ann_foot_gx - R.pitch_x_ref, R.ann_foot_gy - R.pitch_y_ref)
R["err_foot"] = np.hypot(R.pitch_x - R.ann_foot_gx, R.pitch_y - R.ann_foot_gy)
side = C.touch_sides(R.x1, R.y1, R.x2, R.y2)
for k, v in side.items():
    R["edge_" + k] = v
R["edge_any"] = side["left"] | side["right"] | side["top"] | side["bottom"]
R["h"] = R.y2 - R.y1
R["placed"] = R.pitch_usable.astype(bool) & R.pitch_x.notna()
print("\nmatched rows:", len(R), "; with a gbf ground point:", int(R.pitch_x.notna().sum()), "; pitch_usable:", int(R.pitch_usable.sum()),
      "; in_pitch:", int(R.in_pitch.sum()), "; annotation has a ground point:", int(R.pitch_x_ref.notna().sum()))  # fmt: skip
print("calib_status:", R.calib_status.value_counts(dropna=False).to_dict())
V = R[R.pitch_x.notna() & R.pitch_x_ref.notna()]

print("\n=== A. deviation of gbf's ground point from the annotated one (m), matched rows ===")
tabs = [qrow("all matched rows", V.err),
        qrow("  camera part (ann. foot pixel through gbf's camera)", V.err_cam),
        qrow("  foot-point part (gbf foot vs ann. foot, gbf's camera)", V.err_foot),
        qrow("pitch_usable rows", V.err[V.pitch_usable.astype(bool)]),
        qrow("not pitch_usable", V.err[~V.pitch_usable.astype(bool)]),
        qrow("in_pitch rows", V.err[V.in_pitch.astype(bool)]),
        qrow("not in_pitch", V.err[~V.in_pitch.astype(bool)])]  # fmt: skip
for st in sorted(V.calib_status.dropna().unique()):
    tabs.append(qrow(f"calib_status = {st}", V.err[V.calib_status == st]))
tabs += [qrow("box away from every image edge", V.err[~V.edge_any]),
         qrow("box at an image edge (any)", V.err[V.edge_any]),
         qrow("  at the left or right edge only", V.err[(V.edge_left | V.edge_right) & ~V.edge_bottom & ~V.edge_top]),
         qrow("  at the bottom edge", V.err[V.edge_bottom]),
         qrow("  at the top edge", V.err[V.edge_top]),
         qrow("  bottom edge: foot-point part", V.err_foot[V.edge_bottom]),
         qrow("  left/right edge: foot-point part", V.err_foot[(V.edge_left | V.edge_right) & ~V.edge_bottom & ~V.edge_top])]  # fmt: skip
for lo, hi in ((0, 50), (50, 65), (65, 80), (80, 95), (95, 110), (110, 400)):
    tabs.append(qrow(f"distance from camera {lo}-{hi} m", V.err[(V.cam_dist >= lo) & (V.cam_dist < hi)]))
for lo, hi in ((0, 30), (30, 40), (40, 60), (60, 90), (90, 2000)):
    tabs.append(qrow(f"box height {lo}-{hi} px (720p)", V.err[(V.h >= lo) & (V.h < hi)]))
for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 1000)):
    tabs.append(qrow(f"frame rep_err {lo}-{hi} px", V.err[(V.rep_err >= lo) & (V.rep_err < hi)]))
tabs.append(qrow("frame rep_err missing", V.err[V.rep_err.isna()]))
tabs.append(qrow("calib_age = 0", V.err[V.calib_age == 0]))
tabs.append(qrow("calib_age > 0", V.err[V.calib_age > 0]))
A = pd.DataFrame(tabs)
print(A.round(2).to_string())
A.to_csv(C.OUT / "t2_position_error.csv", index=False)
res["position_error"] = A.round(3).to_dict("records")
pc = pd.DataFrame([{**qrow(seq, g.err), "median_dx": g.dx.median(), "median_dy": g.dy.median(), "cam_part_p50": g.err_cam.median(),
                    "foot_part_p50": g.err_foot.median(), "cam_y": float(np.asarray(data.calib(seq).C.iloc[0])[1]),
                    "cam_height": -float(np.asarray(data.calib(seq).C.iloc[0])[2])} for seq, g in V.groupby("seq")])  # fmt: skip
print("\nper clip:")
print(pc.round(2).to_string())
pc.to_csv(C.OUT / "t2_position_error_per_clip.csv", index=False)
print("\nsigned deviation dy (gbf - annotation; + = nearer the camera) by distance from the camera:")
for lo, hi in ((0, 50), (50, 65), (65, 80), (80, 95), (95, 110), (110, 400)):
    g = V[(V.cam_dist >= lo) & (V.cam_dist < hi)]
    print(f"  {lo}-{hi} m: n={len(g)} median dx={g.dx.median():+.2f} dy={g.dy.median():+.2f}  |dx| p90={g.dx.abs().quantile(0.9):.2f} |dy| p90={g.dy.abs().quantile(0.9):.2f}")
print("annotation positions outside the pitch (|x|>52.5 or |y|>34):", round(float(((V.pitch_x_ref.abs() > 52.5) | (V.pitch_y_ref.abs() > 34)).mean()), 4))

# ------------------------------------------------------------------ part B
print("\n=== B. returns: exit -> re-entry displacement against gap time ===")
ret_rows = []
for seq in C.CLIPS:
    t = data.tracks(seq)
    a = C.ann(seq)
    m = C.matched(seq)
    akey = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(a.frame, a.ref_id))}
    mrow = {int(tr): i for i, tr in enumerate(m.trow)}
    tid = t.track_id.to_numpy()
    fr = t.frame.to_numpy()
    px, py = t.pitch_x.to_numpy(), t.pitch_y.to_numpy()
    usable = t.pitch_usable.to_numpy(bool) & ~np.isnan(px)
    x1, y1, x2, y2 = (t[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
    sd = C.touch_sides(x1, y1, x2, y2)

    def sides(i):
        return "+".join(k for k in ("left", "right", "top", "bottom") if sd[k][i]) or "none"

    def robust(piece, frame, step, n=5):
        """Median ground point of up to n placed rows of the piece at/before (step=-1) or at/after the frame."""
        sel = np.where((tid == piece) & usable & ((fr <= frame) if step < 0 else (fr >= frame)))[0]
        if len(sel) == 0:
            return np.nan, np.nan, 0
        order = sel[np.argsort(fr[sel])]
        pick = order[-n:] if step < 0 else order[:n]
        return float(np.median(px[pick])), float(np.median(py[pick])), len(pick)

    for r in C.returns(seq).itertuples():
        ia, ib = akey[(r.last, r.ref_id)], akey[(r.first, r.ref_id)]
        row = {"seq": seq, "ref_id": r.ref_id, "gap": r.gap, "T_ann": (r.first - r.last) / C.FPS, "scored": r.scored,
               "ax_ann": a.pitch_x.iat[ia], "ay_ann": a.pitch_y.iat[ia], "bx_ann": a.pitch_x.iat[ib], "by_ann": a.pitch_y.iat[ib]}  # fmt: skip
        row["d_ann"] = float(np.hypot(row["bx_ann"] - row["ax_ann"], row["by_ann"] - row["ay_ann"]))
        if r.scored:
            ta, tb = r.trow_a, r.trow_b
            row.update({"fa": int(fr[ta]), "fb": int(fr[tb]), "T_gbf": (fr[tb] - fr[ta]) / C.FPS, "piece_a": r.piece_a, "piece_b": r.piece_b,
                        "a_sides": sides(ta), "b_sides": sides(tb), "a_usable": bool(usable[ta]), "b_usable": bool(usable[tb]),
                        "ax": px[ta] if usable[ta] else np.nan, "ay": py[ta] if usable[ta] else np.nan,
                        "bx": px[tb] if usable[tb] else np.nan, "by": py[tb] if usable[tb] else np.nan})  # fmt: skip
            row["d_gbf"] = float(np.hypot(row["bx"] - row["ax"], row["by"] - row["ay"]))
            rax, ray, na = robust(r.piece_a, fr[ta], -1)
            rbx, rby, nb = robust(r.piece_b, fr[tb], +1)
            row["d_gbf5"] = float(np.hypot(rbx - rax, rby - ray))
            # position deviation at the rows A and B themselves
            ma, mb = mrow[ta], mrow[tb]
            row["err_a"] = float(np.hypot(px[ta] - m.pitch_x_ref.iat[ma], py[ta] - m.pitch_y_ref.iat[ma])) if usable[ta] else np.nan
            row["err_b"] = float(np.hypot(px[tb] - m.pitch_x_ref.iat[mb], py[tb] - m.pitch_y_ref.iat[mb])) if usable[tb] else np.nan
        ret_rows.append(row)
B = pd.DataFrame(ret_rows)
B["v_ann"] = B.d_ann / B.T_ann
B["v_gbf"] = B.d_gbf / B.T_gbf
B.to_csv(C.OUT / "t2_returns.csv", index=False)
S = B[B.scored]
print("returns:", len(B), "scored:", len(S), "; rows A/B with a usable gbf ground point:", int(S.a_usable.sum()), int(S.b_usable.sum()), "; both:", int((S.a_usable & S.b_usable).sum()))
print("gap time (s), all 167:", C.qtable(B.T_ann, (0.1, 0.25, 0.5, 0.75, 0.9)))
print("share of returns with a gap > 10 s:", round(float((B.T_ann > 10).mean()), 3), " (scored:", round(float((S.T_ann > 10).mean()), 3), ")")
print("annotation: displacement (m):", C.qtable(B.d_ann))
print("annotation: mean speed over the gap d/T (m/s):", C.qtable(B.v_ann))
print("gbf rows A,B: displacement (m):", C.qtable(S.d_gbf))
print("gbf rows A,B: d/T (m/s):", C.qtable(S.v_gbf))
print("gbf deviation at row A (m):", C.qtable(S.err_a), " at row B:", C.qtable(S.err_b))
print("exit side (gbf box of row A):", S.a_sides.value_counts().to_dict())
print("re-entry side (gbf box of row B):", S.b_sides.value_counts().to_dict())
res["returns"] = {"n": len(B), "scored": len(S), "gap_s": C.qtable(B.T_ann, (0.1, 0.25, 0.5, 0.75, 0.9)), "share_gap_over_10s_all": round(float((B.T_ann > 10).mean()), 3),
                  "share_gap_over_10s_scored": round(float((S.T_ann > 10).mean()), 3),
                  "d_ann_m": C.qtable(B.d_ann), "v_ann_mps": C.qtable(B.v_ann), "d_gbf_m": C.qtable(S.d_gbf), "v_gbf_mps": C.qtable(S.v_gbf),
                  "err_at_A_m": C.qtable(S.err_a), "err_at_B_m": C.qtable(S.err_b)}  # fmt: skip
bins = [(0.4, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 31)]
bt = []
for lo, hi in bins:
    ga = B[(B.T_ann >= lo) & (B.T_ann < hi)]
    gs = S[(S.T_gbf >= lo) & (S.T_gbf < hi)]
    bt.append({"gap (s)": f"{lo}-{hi}", "returns (ann.)": len(ga), "d_ann p50": ga.d_ann.median(), "d_ann p90": ga.d_ann.quantile(0.9), "d_ann max": ga.d_ann.max(),
               "v_ann p50": ga.v_ann.median(), "v_ann p90": ga.v_ann.quantile(0.9), "v_ann max": ga.v_ann.max(),
               "scored": len(gs), "d_gbf p50": gs.d_gbf.median(), "d_gbf p90": gs.d_gbf.quantile(0.9), "d_gbf max": gs.d_gbf.max(), "v_gbf max": gs.v_gbf.max()})  # fmt: skip
BT = pd.DataFrame(bt)
print(BT.round(2).to_string())
BT.to_csv(C.OUT / "t2_returns_by_gap.csv", index=False)
res["returns_by_gap"] = BT.round(3).to_dict("records")

print("\ngate d <= v*T + e on the true returns: share passing")
vs = (3, 4, 5, 6, 7, 8, 10, 12)
es = (0, 1, 2, 3, 4, 5, 6, 8)
for label, d, T in (("annotation (167)", B.d_ann, B.T_ann), ("gbf rows A,B (both placed)", S.d_gbf[S.d_gbf.notna()], S.T_gbf[S.d_gbf.notna()]),
                    ("gbf median of 5 rows", S.d_gbf5[S.d_gbf5.notna()], S.T_gbf[S.d_gbf5.notna()])):  # fmt: skip
    g = pd.DataFrame({f"e={e}": [float((d <= v * T + e).mean()) for v in vs] for e in es}, index=[f"v={v}" for v in vs])
    print(label, "n =", len(d))
    print(g.round(3).to_string())
    g.reset_index().to_csv(C.OUT / f"t2_gate_true_{label.split()[0]}_{len(d)}.csv", index=False)
    need = {f"v={v}": {"e for 100%": round(float((d - v * T).max()), 2), "e for 99%": round(float(np.quantile(d - v * T, 0.99)), 2), "e for 95%": round(float(np.quantile(d - v * T, 0.95)), 2)} for v in vs}
    print("  smallest e (m) that passes 100% / 99% / 95%:", need)
    res[f"gate_need_{label}"] = need
print("returns the gate v=8, e=3 would reject (gbf rows):")
print(S[S.d_gbf > 8 * S.T_gbf + 3][["seq", "ref_id", "gap", "T_gbf", "d_gbf", "d_gbf5", "d_ann", "err_a", "err_b", "a_sides", "b_sides"]].round(2).to_string())

# false candidate pairs: an earlier piece A and a later piece B of DIFFERENT annotated persons
print()
print("=== gate on piece pairs (what a method sees: anchors of the pieces) ===")
print("anchor = median ground point of the piece's last / first 5 placed rows (pitch_usable, in_pitch), with the median frame of those rows as its time")


def anchors(seq):
    t = data.tracks(seq)
    t = t[(t.track_id >= 0) & t.pitch_usable & t.in_pitch & t.pitch_x.notna()].sort_values(["track_id", "frame"])
    out = {}
    for pid, g in t.groupby("track_id"):
        h, tl = g.head(5), g.tail(5)
        out[int(pid)] = {"sx": float(h.pitch_x.median()), "sy": float(h.pitch_y.median()), "sf": float(h.frame.median()),
                         "ex": float(tl.pitch_x.median()), "ey": float(tl.pitch_y.median()), "ef": float(tl.frame.median())}  # fmt: skip
    return out


pairs = []
stale = []
for seq in C.CLIPS:
    p = C.piece_ident(seq)
    an = anchors(seq)
    m = C.matched(seq)
    ends = {}
    for pid, g in m[m.piece >= 0].sort_values("frame").groupby("piece"):
        ends[int(pid)] = (set(g.ref_id.head(5)), set(g.ref_id.tail(5)))
    for r in p.itertuples():
        a = an.get(r.piece)
        stale.append({"seq": seq, "piece": r.piece, "has_anchor": a is not None, "end_stale": (r.last - a["ef"]) if a else np.nan, "start_stale": (a["sf"] - r.first) if a else np.nan})
    p = p[p.ref_id.notna()].reset_index(drop=True)
    f, l, rid = p["first"].to_numpy(), p["last"].to_numpy(), p.ref_id.to_numpy()
    nxt = {}
    for rr, g in p.groupby("ref_id"):
        ids = g.sort_values("first").index.tolist()
        for i in range(len(ids) - 1):
            nxt[ids[i]] = ids[i + 1]
    for i in range(len(p)):
        ai = an.get(int(p.piece.iat[i]))
        if ai is None:
            continue
        for j in range(len(p)):
            bj = an.get(int(p.piece.iat[j]))
            if l[i] >= f[j] or bj is None:
                continue
            same = rid[i] == rid[j]
            if same and nxt.get(i) != j:
                continue  # true pairs: consecutive pieces of the person only
            ea, eb = ends[int(p.piece.iat[i])], ends[int(p.piece.iat[j])]
            pairs.append({"seq": seq, "a": p.piece.iat[i], "b": p.piece.iat[j], "same": bool(same), "gap": int(f[j] - l[i] - 1), "T": (bj["sf"] - ai["ef"]) / C.FPS,
                          "d": float(np.hypot(bj["sx"] - ai["ex"], bj["sy"] - ai["ey"])), "a_edge": bool(p.ends_at_edge.iat[i]), "b_edge": bool(p.starts_at_edge.iat[j]),
                          "team_a": p.team.iat[i], "team_b": p.team.iat[j], "role_a": p.role.iat[i], "role_b": p.role.iat[j],
                          "clean_ends": ea[1] == {int(rid[i])} and eb[0] == {int(rid[j])}})  # fmt: skip
ST = pd.DataFrame(stale)
print("pieces:", len(ST), "; without any placed row (no anchor):", int((~ST.has_anchor).sum()), "; end anchor older than 5 frames:", int((ST.end_stale > 5).sum()),
      "; start anchor later than 5 frames:", int((ST.start_stale > 5).sum()))  # fmt: skip
res["piece_anchors"] = {"pieces": len(ST), "no_anchor": int((~ST.has_anchor).sum()), "end_anchor_older_than_5_frames": int((ST.end_stale > 5).sum()),
                        "start_anchor_later_than_5_frames": int((ST.start_stale > 5).sum())}  # fmt: skip
PP = pd.DataFrame(pairs)
PP["team_ok"] = PP.team_a.isna() | PP.team_b.isna() | (PP.team_a == PP.team_b)
PP["role_ok"] = PP.role_a == PP.role_b
PP.to_csv(C.OUT / "t2_piece_pairs.csv", index=False)
print("pairs:", len(PP), "; true (consecutive pieces of one person):", int(PP.same.sum()), "of which with clean ends (the 5 matched rows at A's end and at B's start are all the person):",
      int((PP.same & PP.clean_ends).sum()), "; wrong:", int((~PP.same).sum()))  # fmt: skip
gate_rows = []
gbins = [(0, 10), (10, 25), (25, 50), (50, 125), (125, 250), (250, 750)]
for v, e in ((12, 4), (8, 3), (8, 5), (7, 3), (6, 3), (5, 4), (4, 4), (3, 5)):
    for lo, hi in gbins:
        g = PP[(PP.gap >= lo) & (PP.gap < hi)]
        tr, fa = g[g.same], g[~g.same]
        trc = tr[tr.clean_ends]
        fa_t = fa[fa.team_ok & fa.role_ok]
        ok = lambda x: float((x.d <= v * x["T"] + e).mean()) if len(x) else np.nan  # noqa: E731
        gate_rows.append({"v": v, "e": e, "gap frames": f"{lo}-{hi}", "true pairs": len(tr), "true pass": ok(tr), "true, clean ends": len(trc), "clean pass": ok(trc),
                          "wrong pairs": len(fa), "wrong pass": ok(fa), "wrong pairs (team+role ok)": len(fa_t), "wrong pass (team+role ok)": ok(fa_t)})  # fmt: skip
G = pd.DataFrame(gate_rows)
print(G.round(3).to_string())
G.to_csv(C.OUT / "t2_gate_piece_pairs.csv", index=False)
res["gate_piece_pairs"] = G.round(4).to_dict("records")
print("true consecutive pairs with clean ends: d - v*T quantiles (m):")
cl = PP[PP.same & PP.clean_ends]
for v in (4, 5, 6, 7, 8, 12):
    x = cl.d - v * cl["T"]
    print(f"  v={v}: p90={np.quantile(x, 0.9):.2f} p95={np.quantile(x, 0.95):.2f} p99={np.quantile(x, 0.99):.2f} max={x.max():.2f}")
print("clean true pairs failing v=8, e=3:")
print(cl[cl.d > 8 * cl["T"] + 3][["seq", "a", "b", "gap", "T", "d", "a_edge", "b_edge"]].round(2).to_string())
print("true consecutive pairs with gap < 10 frames (the plugin lost the person for a moment):", int((PP.same & (PP.gap < 10)).sum()),
      " d quantiles:", C.qtable(PP.d[PP.same & (PP.gap < 10)]))  # fmt: skip
print("top annotated mean speeds over a return gap (m/s):")
print(B.sort_values("v_ann", ascending=False).head(6)[["seq", "ref_id", "gap", "T_ann", "d_ann", "v_ann", "scored"]].round(2).to_string())
res["fastest_annotated_returns"] = B.sort_values("v_ann", ascending=False).head(6)[["seq", "ref_id", "gap", "T_ann", "d_ann", "v_ann", "scored"]].round(2).to_dict("records")
(C.OUT / "t2_position.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
