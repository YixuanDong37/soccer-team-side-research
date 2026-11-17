"""(3) Camera view: what the per-frame calibration says about a returning player (diagnosis).

A. The view per frame: visible pitch area, visible x-range, pan speed, zoom.
B. The 167 annotated returns: exit side against re-entry side; is the exit caused by the camera
   panning away or by the player walking out; is the exit ground point out of the picture
   during the gap ("he stood still, the camera left and came back"); is the straight path from
   exit to re-entry out of the picture during the gap.
C. Piece pairs (what a method sees): the same two cues - same side, path out of the picture -
   on true pairs (consecutive pieces of one annotated person) and on wrong pairs, alone and
   after the speed gate and the team / role agreement.
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
print("projection check against gbf.geometry.project (px):", V.check())

# ------------------------------------------------------------------ A. the view per frame
print("\n=== A. the camera view per frame ===")
vt = []
for seq in C.CLIPS:
    v = V.view_table(seq).copy()
    v["seq"] = seq
    # pan speed: change of the image centre's ground point x over +-5 frames (m/s)
    cx = v.cx.to_numpy()
    k = 5
    pan = np.full(len(v), np.nan)
    pan[k:-k] = (cx[2 * k:] - cx[:-2 * k]) / (2 * k / C.FPS)
    v["pan_mps"] = pan
    v["width_mid"] = v.xr_mid - v.xl_mid
    vt.append(v)
VT = pd.concat(vt, ignore_index=True)
VT.to_csv(C.OUT / "t3_view_per_frame.csv", index=False)
per = VT.groupby("seq").agg(frames=("frame", "size"), with_camera=("area_m2", lambda s: int(s.notna().sum())), area_p10=("area_m2", lambda s: s.quantile(0.1)),
                            area_p50=("area_m2", "median"), area_p90=("area_m2", lambda s: s.quantile(0.9)), width_mid_p50=("width_mid", "median"),
                            focal_p50=("focal", "median"), focal_min=("focal", "min"), focal_max=("focal", "max"),
                            pan_abs_p50=("pan_mps", lambda s: s.abs().median()), pan_abs_p90=("pan_mps", lambda s: s.abs().quantile(0.9)), pan_abs_max=("pan_mps", lambda s: s.abs().max()),
                            cx_min=("cx", "min"), cx_max=("cx", "max"))  # fmt: skip
per["area_share_p50"] = per.area_p50 / 7140
print(per.round(1).to_string())
per.reset_index().to_csv(C.OUT / "t3_view_per_clip.csv", index=False)
res["view"] = {"visible_pitch_share": C.qtable(VT.area_m2 / 7140, (0.1, 0.5, 0.9)), "visible_width_at_halfway_axis_m": C.qtable(VT.width_mid, (0.1, 0.5, 0.9)),
               "abs_pan_speed_mps": C.qtable(VT.pan_mps.abs(), (0.5, 0.9, 0.99)), "frames_without_camera": int(VT.area_m2.isna().sum()), "frames": len(VT)}  # fmt: skip
print(res["view"])

# ------------------------------------------------------------------ B. annotated returns
print("\n=== B. annotated returns and the camera ===")


def side_of(x1, y1, x2, y2, w, h, px):
    s = [k for k, c in (("left", x1 <= px), ("right", x2 >= w - 1 - px), ("top", y1 <= px), ("bottom", y2 >= h - 1 - px)) if c]
    return "+".join(s) if s else "none"


def lateral(side, x1, x2, w):
    """left / right: the touched lateral edge, else the nearer one by box centre."""
    if "left" in side and "right" not in side:
        return "left"
    if "right" in side and "left" not in side:
        return "right"
    return "left" if (x1 + x2) / 2 < (w - 1) / 2 else "right"


rows = []
for seq in C.CLIPS:
    a = C.ann(seq)
    akey = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(a.frame, a.ref_id))}
    byp = {int(r): g.sort_values("frame") for r, g in a.groupby("ref_id")}
    vtab = V.view_table(seq).set_index("frame")
    t = data.tracks(seq)
    for r in C.returns(seq).itertuples():
        ia, ib = akey[(r.last, r.ref_id)], akey[(r.first, r.ref_id)]
        A, B = a.iloc[ia], a.iloc[ib]
        sa = side_of(A.x1, A.y1, A.x2, A.y2, 1920, 1080, 15.0)
        sb = side_of(B.x1, B.y1, B.x2, B.y2, 1920, 1080, 15.0)
        row = {"seq": seq, "ref_id": r.ref_id, "last": r.last, "first": r.first, "gap": r.gap, "T": (r.first - r.last) / C.FPS, "scored": r.scored,
               "exit_side": sa, "entry_side": sb, "exit_lat": lateral(sa, A.x1, A.x2, 1920), "entry_lat": lateral(sb, B.x1, B.x2, 1920),
               "ax": A.pitch_x, "ay": A.pitch_y, "bx": B.pitch_x, "by": B.pitch_y}  # fmt: skip
        row["d"] = float(np.hypot(row["bx"] - row["ax"], row["by"] - row["ay"]))
        # player velocity along x before the exit and after the re-entry (annotation, up to 10 frames)
        g = byp[r.ref_id]
        pre = g[(g.frame <= r.last) & (g.frame >= r.last - 10)]
        post = g[(g.frame >= r.first) & (g.frame <= r.first + 10)]
        row["vx_player_exit"] = float((pre.pitch_x.iat[-1] - pre.pitch_x.iat[0]) / ((pre.frame.iat[-1] - pre.frame.iat[0]) / C.FPS)) if len(pre) > 1 else np.nan
        row["vx_player_entry"] = float((post.pitch_x.iat[-1] - post.pitch_x.iat[0]) / ((post.frame.iat[-1] - post.frame.iat[0]) / C.FPS)) if len(post) > 1 else np.nan
        # velocity of the view boundary on the exit side at the player's depth (m/s), +-5 frames around the exit / re-entry
        for tag, fr0, yy, lat in (("exit", r.last, A.pitch_y, row["exit_lat"]), ("entry", r.first, B.pitch_y, row["entry_lat"])):
            f0, f1 = max(fr0 - 5, 0), min(fr0 + 5, len(vtab) - 1)
            xl, xr = V.lateral_bounds(seq, [f0, f1], yy)
            b = xl if lat == "left" else xr
            row[f"v_boundary_{tag}"] = float((b[1] - b[0]) / ((f1 - f0) / C.FPS)) if np.isfinite(b).all() else np.nan
        row["pan_exit"] = float(vtab.cx.get(min(r.last + 5, len(vtab) - 1), np.nan) - vtab.cx.get(max(r.last - 5, 0), np.nan)) / (10 / C.FPS)
        # during the gap: is the exit point (standing still) in the picture? is the straight path?
        gf = np.arange(r.last + 1, r.first)
        if len(gf):
            vis0, _ = V.person_visible(seq, gf, np.full(len(gf), A.pitch_x), np.full(len(gf), A.pitch_y), margin_px=0)
            vis30, _ = V.person_visible(seq, gf, np.full(len(gf), A.pitch_x), np.full(len(gf), A.pitch_y), margin_px=30)
            w_ = (gf - r.last) / (r.first - r.last)
            lx, ly = A.pitch_x + w_ * (B.pitch_x - A.pitch_x), A.pitch_y + w_ * (B.pitch_y - A.pitch_y)
            lin0, _ = V.person_visible(seq, gf, lx, ly, margin_px=0)
            lin30, _ = V.person_visible(seq, gf, lx, ly, margin_px=30)
            row.update({"still_visible_share": float(np.nanmean(vis0)) if np.isfinite(vis0).any() else np.nan,
                        "still_visible30_share": float(np.nanmean(vis30)) if np.isfinite(vis30).any() else np.nan,
                        "still_visible30_frames": float(np.nansum(vis30)),
                        "line_visible_share": float(np.nanmean(lin0)) if np.isfinite(lin0).any() else np.nan,
                        "line_visible30_share": float(np.nanmean(lin30)) if np.isfinite(lin30).any() else np.nan,
                        "line_visible30_frames": float(np.nansum(lin30)), "gap_frames_with_camera": int(np.isfinite(vis0).sum())})  # fmt: skip
        # at the re-entry frame: is the exit point in the picture (did he come back to where he left)?
        vb, _ = V.person_visible(seq, [r.first], [A.pitch_x], [A.pitch_y], margin_px=0)
        row["exit_point_visible_at_entry"] = float(vb[0])
        rows.append(row)
RT = pd.DataFrame(rows)
RT.to_csv(C.OUT / "t3_returns.csv", index=False)
print("returns:", len(RT))
print("exit side (annotated box within 15 px at 1080p):", RT.exit_side.value_counts().to_dict())
print("re-entry side:", RT.entry_side.value_counts().to_dict())
ct = pd.crosstab(RT.exit_side, RT.entry_side)
print("exit side (rows) x re-entry side (cols), touching edges only:")
print(ct.to_string())
ct.to_csv(C.OUT / "t3_exit_entry_sides.csv")
ctl = pd.crosstab(RT.exit_lat, RT.entry_lat)
print("lateral side (touched edge, else nearer edge): exit (rows) x re-entry (cols):")
print(ctl.to_string())
same_lat = RT.exit_lat == RT.entry_lat
both_touch = RT.exit_side.isin(["left", "right"]) & RT.entry_side.isin(["left", "right"])
res["sides"] = {"returns": len(RT), "same_lateral_side": int(same_lat.sum()), "share": round(float(same_lat.mean()), 3),
                "both_boxes_touch_left_or_right": int(both_touch.sum()), "of_those_same_side": int((RT.exit_side == RT.entry_side)[both_touch].sum()),
                "scored_same_lateral_side": int(same_lat[RT.scored].sum()), "scored": int(RT.scored.sum())}  # fmt: skip
print(res["sides"])
print("returns re-entering on the OTHER lateral side:")
print(RT[~same_lat][["seq", "ref_id", "gap", "T", "d", "exit_side", "entry_side", "ax", "bx", "still_visible30_share", "line_visible30_share", "scored"]].round(2).to_string())

# camera-driven or player-driven exits
RT["rel_exit"] = np.where(RT.exit_lat == "left", RT.v_boundary_exit - RT.vx_player_exit, RT.vx_player_exit - RT.v_boundary_exit)  # > 0: boundary overtakes the player
RT["cam_part_exit"] = np.where(RT.exit_lat == "left", RT.v_boundary_exit, -RT.v_boundary_exit)   # > 0: the view moves away from the player
RT["ply_part_exit"] = np.where(RT.exit_lat == "left", -RT.vx_player_exit, RT.vx_player_exit)      # > 0: the player moves outwards
lat_exit = RT[RT.exit_side.isin(["left", "right"])]
print("\nexits at the left / right edge:", len(lat_exit))
print("  view boundary speed away from the player (m/s):", C.qtable(lat_exit.cam_part_exit, (0.1, 0.25, 0.5, 0.75, 0.9)))
print("  player speed outwards (m/s):", C.qtable(lat_exit.ply_part_exit, (0.1, 0.25, 0.5, 0.75, 0.9)))
cam_driven = (lat_exit.cam_part_exit > lat_exit.ply_part_exit.clip(lower=0))
print("  the view moved away faster than the player moved out (camera-driven):", int(cam_driven.sum()), "of", len(lat_exit))
print("  the player was moving INWARDS (against the exit) when the view left him:", int((lat_exit.ply_part_exit < 0).sum()))
res["exit_cause"] = {"lateral_exits": len(lat_exit), "camera_driven": int(cam_driven.sum()), "player_moving_inwards": int((lat_exit.ply_part_exit < 0).sum()),
                     "boundary_speed_away_mps": C.qtable(lat_exit.cam_part_exit, (0.1, 0.5, 0.9)), "player_speed_outwards_mps": C.qtable(lat_exit.ply_part_exit, (0.1, 0.5, 0.9))}  # fmt: skip

# pan direction (ground point of the image centre, m/s along x, +-5 frames) against the exit side
pan = lat_exit.pan_exit
away = np.where(lat_exit.exit_lat == "left", pan, -pan)   # > 0: the camera pans away from the side the player left on
print("  camera pan at the exit, towards the opposite side (m/s):", C.qtable(away, (0.1, 0.25, 0.5, 0.75, 0.9)))
print("  pans away > 1 m/s:", int((away > 1).sum()), "; about still (|pan| <= 1):", int((np.abs(away) <= 1).sum()),
      "; pans towards the player's side > 1 m/s:", int((away < -1).sum()), "; no camera:", int(np.isnan(away).sum()))  # fmt: skip
res["pan_at_exit"] = {"pans_away_gt_1mps": int((away > 1).sum()), "about_still": int((np.abs(away) <= 1).sum()), "pans_towards_gt_1mps": int((away < -1).sum()),
                      "no_camera": int(np.isnan(away).sum())}  # fmt: skip
lat_in = RT[RT.entry_side.isin(["left", "right"])]
cam_in = np.where(lat_in.entry_lat == "left", -lat_in.v_boundary_entry, lat_in.v_boundary_entry)   # > 0: the view grows towards the player
ply_in = np.where(lat_in.entry_lat == "left", lat_in.vx_player_entry, -lat_in.vx_player_entry)      # > 0: the player moves into the picture
print("re-entries at the left / right edge:", len(lat_in))
print("  view boundary speed towards the player (m/s):", C.qtable(cam_in, (0.1, 0.25, 0.5, 0.75, 0.9)))
print("  player speed inwards (m/s):", C.qtable(ply_in, (0.1, 0.25, 0.5, 0.75, 0.9)))
print("  the view came to the player faster than he came in (camera-driven):", int((cam_in > np.clip(ply_in, 0, None)).sum()), "of", len(lat_in))
res["entry_cause"] = {"lateral_entries": len(lat_in), "camera_driven": int((cam_in > np.clip(ply_in, 0, None)).sum())}

print()
print("during the gap (annotated exit and re-entry ground points):")
for name, col in (("exit point (standing still) in the picture, share of gap frames", "still_visible_share"),
                  ("  ... at least 30 px inside the left/right edges", "still_visible30_share"),
                  ("straight exit->re-entry path in the picture, share of gap frames", "line_visible_share"),
                  ("  ... at least 30 px inside", "line_visible30_share")):  # fmt: skip
    x = RT[col]
    print(f"  {name}: n={x.notna().sum()} share==0: {(x == 0).mean():.3f}  <=0.1: {(x <= 0.1).mean():.3f}  <=0.25: {(x <= 0.25).mean():.3f}  <=0.5: {(x <= 0.5).mean():.3f}  mean {x.mean():.3f}")
    res[col] = {"share_eq_0": round(float((x == 0).mean()), 3), "share_le_0.1": round(float((x <= 0.1).mean()), 3), "share_le_0.25": round(float((x <= 0.25).mean()), 3),
                "share_le_0.5": round(float((x <= 0.5).mean()), 3), "mean": round(float(x.mean()), 3)}  # fmt: skip
print("  frames (not share) the straight path is >= 30 px inside:", C.qtable(RT.line_visible30_frames, (0.5, 0.75, 0.9, 0.95)))
print("  exit point back in the picture at the re-entry frame:", round(float(RT.exit_point_visible_at_entry.mean()), 3))
res["exit_point_visible_at_entry"] = round(float(RT.exit_point_visible_at_entry.mean()), 3)
print("by gap length:")
RT["gbin"] = pd.cut(RT["T"], [0, 2, 5, 10, 40])
print(RT.groupby("gbin", observed=True).agg(n=("T", "size"), same_side=("exit_lat", lambda s: float((s == RT.loc[s.index, "entry_lat"]).mean())),
                                             still30_mean=("still_visible30_share", "mean"), line30_mean=("line_visible30_share", "mean"),
                                             line30_le_01=("line_visible30_share", lambda s: float((s <= 0.1).mean())),
                                             back_at_exit_point=("exit_point_visible_at_entry", "mean"), d_p50=("d", "median")).round(3).to_string())  # fmt: skip

# ------------------------------------------------------------------ C. piece pairs
print("\n=== C. piece pairs: side and path cues (label-free features; labels only say true / wrong) ===")


def anchors(seq):
    t = data.tracks(seq)
    t = t[t.track_id >= 0].sort_values(["track_id", "frame"])
    out = {}
    for pid, g in t.groupby("track_id"):
        pl = g[g.pitch_usable & g.in_pitch & g.pitch_x.notna()]
        first, last = g.iloc[0], g.iloc[-1]
        d = {"s_side": side_of(first.x1, first.y1, first.x2, first.y2, W, H, data.EDGE_PX), "e_side": side_of(last.x1, last.y1, last.x2, last.y2, W, H, data.EDGE_PX)}
        d["s_lat"] = lateral(d["s_side"], first.x1, first.x2, W)
        d["e_lat"] = lateral(d["e_side"], last.x1, last.x2, W)
        if len(pl):
            h, tl = pl.head(5), pl.tail(5)
            d.update({"sx": float(h.pitch_x.median()), "sy": float(h.pitch_y.median()), "sf": float(h.frame.median()),
                      "ex": float(tl.pitch_x.median()), "ey": float(tl.pitch_y.median()), "ef": float(tl.frame.median())})  # fmt: skip
        out[int(pid)] = d
    return out


pairs = []
for seq in C.CLIPS:
    p = C.piece_ident(seq)
    an = anchors(seq)
    p = p[p.ref_id.notna()].reset_index(drop=True)
    f, l, rid = p["first"].to_numpy(), p["last"].to_numpy(), p.ref_id.to_numpy()
    nxt = {}
    for rr, g in p.groupby("ref_id"):
        ids = g.sort_values("first").index.tolist()
        for i in range(len(ids) - 1):
            nxt[ids[i]] = ids[i + 1]
    for i in range(len(p)):
        ai = an[int(p.piece.iat[i])]
        if "ex" not in ai:
            continue
        for j in range(len(p)):
            bj = an[int(p.piece.iat[j])]
            gapf = f[j] - l[i] - 1
            if gapf < 10 or "sx" not in bj:
                continue
            if not (p.ends_at_edge.iat[i] or p.starts_at_edge.iat[j]):
                continue  # the split rule's candidates: a gap of >= 10 frames next to an image edge
            same = rid[i] == rid[j]
            if same and nxt.get(i) != j:
                continue
            f0, f1 = int(round(ai["ef"])), int(round(bj["sf"]))
            gf = np.arange(f0 + 1, f1)
            lin30 = still30 = np.nan
            if len(gf):
                w_ = (gf - f0) / (f1 - f0)
                lv, _ = V.person_visible(seq, gf, ai["ex"] + w_ * (bj["sx"] - ai["ex"]), ai["ey"] + w_ * (bj["sy"] - ai["ey"]), margin_px=30)
                sv, _ = V.person_visible(seq, gf, np.full(len(gf), ai["ex"]), np.full(len(gf), ai["ey"]), margin_px=30)
                lin30 = float(np.nanmean(lv)) if np.isfinite(lv).any() else np.nan
                still30 = float(np.nanmean(sv)) if np.isfinite(sv).any() else np.nan
            T = (bj["sf"] - ai["ef"]) / C.FPS
            pairs.append({"seq": seq, "a": int(p.piece.iat[i]), "b": int(p.piece.iat[j]), "same": bool(same), "gap": int(gapf), "T": T,
                          "d": float(np.hypot(bj["sx"] - ai["ex"], bj["sy"] - ai["ey"])), "e_side": ai["e_side"], "s_side": bj["s_side"], "e_lat": ai["e_lat"], "s_lat": bj["s_lat"],
                          "line30": lin30, "still30": still30, "a_edge": bool(p.ends_at_edge.iat[i]), "b_edge": bool(p.starts_at_edge.iat[j]),
                          "team_ok": bool(pd.isna(p.team.iat[i]) or pd.isna(p.team.iat[j]) or p.team.iat[i] == p.team.iat[j]), "role_ok": bool(p.role.iat[i] == p.role.iat[j])})  # fmt: skip
PP = pd.DataFrame(pairs)
PP.to_csv(C.OUT / "t3_piece_pairs.csv", index=False)
PP["speed_ok"] = PP.d <= 8 * PP["T"] + 3
PP["same_side"] = PP.e_lat == PP.s_lat
PP["both_touch"] = PP.e_side.isin(["left", "right"]) & PP.s_side.isin(["left", "right"])
print("candidate pairs (gap >= 10 frames, A ends or B starts at an image edge, both anchored):", len(PP), "; true:", int(PP.same.sum()), "; wrong:", int((~PP.same).sum()))


def cue_line(name, g):
    tr, fa = g[g.same], g[~g.same]
    out = {"set": name, "true": len(tr), "wrong": len(fa)}
    for cue, col in (("same side", "same_side"), ("speed gate 8,3", "speed_ok")):
        out[f"{cue}: true pass"] = float(tr[col].mean()) if len(tr) else np.nan
        out[f"{cue}: wrong pass"] = float(fa[col].mean()) if len(fa) else np.nan
    for thr in (0.1, 0.3):
        out[f"path<= {thr}: true pass"] = float((tr.line30.fillna(0) <= thr).mean()) if len(tr) else np.nan
        out[f"path<= {thr}: wrong pass"] = float((fa.line30.fillna(0) <= thr).mean()) if len(fa) else np.nan
    out["AUC(-path share)"] = C.auc(-tr.line30.fillna(0), -fa.line30.fillna(0))
    return out


lines = [cue_line("all candidates", PP),
         cue_line("both end boxes touch left/right", PP[PP.both_touch]),
         cue_line("gap 10-49 frames", PP[PP.gap < 50]),
         cue_line("gap 50-124", PP[(PP.gap >= 50) & (PP.gap < 125)]),
         cue_line("gap 125-249", PP[(PP.gap >= 125) & (PP.gap < 250)]),
         cue_line("gap >= 250", PP[PP.gap >= 250]),
         cue_line("after team+role ok", PP[PP.team_ok & PP.role_ok]),
         cue_line("after team+role ok and speed gate", PP[PP.team_ok & PP.role_ok & PP.speed_ok]),
         cue_line("... and gap >= 125", PP[PP.team_ok & PP.role_ok & PP.speed_ok & (PP.gap >= 125)]),
         cue_line("after team+role, speed, same side", PP[PP.team_ok & PP.role_ok & PP.speed_ok & PP.same_side])]  # fmt: skip
CL = pd.DataFrame(lines)
print(CL.round(3).to_string())
CL.to_csv(C.OUT / "t3_pair_cues.csv", index=False)
res["pair_cues"] = CL.round(4).to_dict("records")
# funnel: how many wrong candidates per true pair survive each gate
fun = []
g = PP
for name, cond in (("candidates", np.ones(len(PP), bool)), ("+ team and role agree (or team missing)", PP.team_ok & PP.role_ok), ("+ speed gate v=8, e=3", PP.speed_ok),
                   ("+ same lateral side", PP.same_side), ("+ straight path <= 30% in the picture", PP.line30.fillna(0) <= 0.3)):  # fmt: skip
    g = g[cond.reindex(g.index) if hasattr(cond, "reindex") else cond[g.index]]
    fun.append({"gate": name, "true pairs kept": int(g.same.sum()), "wrong pairs kept": int((~g.same).sum()), "wrong per true": round(float((~g.same).sum() / max(g.same.sum(), 1)), 2)})
FU = pd.DataFrame(fun)
print(FU.to_string())
FU.to_csv(C.OUT / "t3_funnel.csv", index=False)
res["funnel"] = FU.to_dict("records")
# per later piece B: how many earlier candidates survive all gates, and is the true one among them
surv = g
by_b = surv.groupby(["seq", "b"]).agg(n=("same", "size"), has_true=("same", "any"))
print("later pieces with surviving candidates:", len(by_b), "; with exactly one candidate:", int((by_b.n == 1).sum()), "of which it is the true one:", int(((by_b.n == 1) & by_b.has_true).sum()))
res["after_all_gates"] = {"later_pieces": len(by_b), "single_candidate": int((by_b.n == 1).sum()), "single_and_true": int(((by_b.n == 1) & by_b.has_true).sum()),
                          "candidates_per_later_piece_p50": float(by_b.n.median()), "p90": float(by_b.n.quantile(0.9))}  # fmt: skip
(C.OUT / "t3_camera.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
