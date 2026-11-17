"""Label-free evidence about the plugin's re-entry joins (agent m_keep).

gbf cut a plugin track wherever it left the picture and came back (refine.json["tracks"]
["reentry_pieces"]).  For every such cut this module describes the pair

    a = the piece of the same plugin id right before the cut,   b = the re-entry piece

i.e. linking [a, b] restores the plugin's own join.  Nothing here reads an annotation.

    import sys; sys.path.insert(0, "C:/Research/plugins/tools")
    sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
    import features
    df = features.pair_features("SNGS-107")      # one row per plugin re-entry join

Columns (a, b first; NaN where the evidence does not exist):
  plugin, k, gap_f, gap_s            plugin id, index of b among the plugin id's pieces, gap
  rows_a/b, span_a/b                 boxes and frame span of the two pieces
  clean_a/b, emb_a/b                 clean rows (data.clean_rows) and rows with an embedding
  role_a/b, role_same, team_a/b, team_same (1 same, 0 unknown, -1 differ), teamv_a/b (verified)
  kit_d                              Lab distance of the two pieces' kit colours (gbf team check), NaN if absent
  kit_rel                            kit_d / distance of the two team prototypes
  jersey_a/b, jersey_same            most frequent confident jersey read (1 same, 0 unknown, -1 differ)
  cos                                cosine of the two piece descriptors (descriptor = mean of the
                                     clean rows' embeddings when >= MIN_CLEAN, else of all rows with one)
  cos_clean                          the same, only when both pieces have >= MIN_CLEAN clean rows
  cos_all                            from all rows with an embedding on both sides
  cos_near                           last NEAR embedded rows of a against first NEAR of b
  q_pair                             2 both descriptors clean, 1 one clean, 0 none clean
  imp_pct                            share of the clip's same-team co-visible piece pairs (two different
                                     people, same kit) whose cosine is >= cos   (small = unlike strangers)
  imp_z                              (cos - impostor median) / (impostor p95 - impostor median)
  alt_b, margin_b, rank_b, n_alt_b   best other earlier-ended compatible piece for b: its cosine to b,
                                     cos - alt_b, rank of a among b's earlier candidates (1 = best), count
  alt_a, margin_a, rank_a, n_alt_a   the same for a among later-starting pieces
  margin                             min(margin_a, margin_b)  (NaN -> the other; both NaN -> NaN)
  alt_b_edge, margin_b_edge          as alt_b but only earlier pieces that ended at the image edge
  dist_m, has_pos                    metres between a's last and b's first pitch position
  speed_req                          dist_m / gap_s
  excess_m                           max(0, dist_m - SPEED * gap_s - 2 * POS_ERR): metres beyond reach
  side_a, side_b, side_same          image side a leaves by / b enters by (L R T B or ""), 1 same 0 unknown -1 differ
  edge_a, edge_b                     a's last / b's first box touches the image edge
  dx_img                             horizontal image distance (px) between a's last and b's first box centre
  pan_deg                            camera pan change over the gap (degrees, from calib R), NaN if unusable
  stay_in_view                       share of gap frames in which a's last pitch position, had he stood
                                     still, would have been inside the picture (margin 30 px)
  n_rival_geo                        other earlier-ended compatible pieces that ended within reach of b's start
  d_rival_geo                        distance (m) of the nearest such rival's end to b's start minus dist_m

pair_features(seq, "all") gives a lighter table for every (earlier, later) pair of pieces with the
flag plugin_join, for fusion with other pair scores.
"""

from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data  # noqa: E402

FPS, W, H = data.FPS, data.WIDTH, data.HEIGHT
MIN_CLEAN = 5
NEAR = 25
SPEED, POS_ERR = 12.0, 2.0
COVISIBLE = 5
EDGE = data.EDGE_PX


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _sides(r, m=EDGE) -> str:
    s = ""
    if r.x1 <= m:
        s += "L"
    if r.x2 >= W - 1 - m:
        s += "R"
    if r.y1 <= m:
        s += "T"
    if r.y2 >= H - 1 - m:
        s += "B"
    return s


@lru_cache(maxsize=None)
def piece_table(seq: str) -> dict:
    """Per piece: rows (frame order), frames, descriptors, role, team, edge sides, pitch ends."""
    t = data.tracks(seq)
    E = data.embeddings(seq)
    clean = data.clean_rows(seq)
    has = ~np.isnan(E[:, 0])
    frame = t.frame.to_numpy()
    team_entry = {e["track_id"]: e for e in data.refine(seq)["teams"]["tracks"]}
    out = {}
    for tid, rows in t.groupby("track_id").indices.items():
        if tid < 0:
            continue
        rows = np.asarray(rows)
        rows = rows[np.argsort(frame[rows], kind="stable")]
        fr = frame[rows]
        sub = t.iloc[rows]
        c_rows, e_rows = rows[clean[rows]], rows[has[rows]]
        d_clean = _unit(E[c_rows].mean(axis=0)) if len(c_rows) >= MIN_CLEAN else None
        d_all = _unit(E[e_rows].mean(axis=0)) if len(e_rows) >= 1 else None
        placed = sub[sub.pitch_usable & sub.in_pitch & sub.pitch_x.notna()]
        head, tail = placed.head(5), placed.tail(5)
        e = team_entry.get(int(tid))
        verified = None
        if e and e["verdict"] == "consistent" and not e.get("samples_against") and not e.get("cleared") and e.get("nearer_team") == e["team"]:
            verified = e["team"]
        jer = sub[sub.jersey.notna() & (sub.jersey_conf.fillna(0) >= 0.5)]
        jersey = jer.jersey.value_counts().index[0] if len(jer) else None
        out[int(tid)] = {
            "id": int(tid), "plugin": int(sub.source_track_id.iloc[0]), "rows": rows, "frames": fr,
            "first": int(fr[0]), "last": int(fr[-1]), "n": len(rows),
            "n_clean": len(c_rows), "n_emb": len(e_rows),
            "d_clean": d_clean, "d_all": d_all, "d": d_clean if d_clean is not None else d_all,
            "d_head": _unit(E[e_rows[:NEAR]].mean(axis=0)) if len(e_rows) else None,
            "d_tail": _unit(E[e_rows[-NEAR:]].mean(axis=0)) if len(e_rows) else None,
            "role": sub.role.mode().iloc[0] if sub.role.notna().any() else None,
            "team": sub.team.mode().iloc[0] if sub.team.notna().any() else None,
            "teamv": verified, "lab": np.array(e["lab"], float) if e and e.get("lab") else None,
            "jersey": jersey, "jersey_n": int(len(jer)),
            "first_box": sub.iloc[0], "last_box": sub.iloc[-1],
            "start_xy": (float(head.pitch_x.median()), float(head.pitch_y.median())) if len(head) else None,
            "end_xy": (float(tail.pitch_x.median()), float(tail.pitch_y.median())) if len(tail) else None,
            "start_f": int(head.frame.iloc[0]) if len(head) else None,
            "end_f": int(tail.frame.iloc[-1]) if len(tail) else None,
        }  # fmt: skip
    return out


def _cls(p) -> str:
    return "ref" if p["role"] == "referee" else ("team" if p["role"] in ("player", "goalkeeper") else "other")


def compatible(p, q) -> bool:
    """Could p and q be one person as far as role and team labels say?"""
    if _cls(p) != _cls(q):
        return False
    if p["team"] is not None and q["team"] is not None and p["team"] != q["team"]:
        return False
    return True


def cos(p, q, key="d"):
    if p[key] is None or q[key] is None:
        return np.nan
    return float(p[key] @ q[key])


@lru_cache(maxsize=None)
def impostors(seq: str) -> np.ndarray:
    """Cosines of same-team piece pairs seen together for >= COVISIBLE frames (different people)."""
    P = [p for p in piece_table(seq).values() if p["d_clean"] is not None and p["team"] is not None and _cls(p) == "team"]
    out = []
    for i, p in enumerate(P):
        fp = set(p["frames"].tolist())
        for q in P[i + 1:]:
            if p["team"] != q["team"] or q["first"] > p["last"] or p["first"] > q["last"]:
                continue
            if len(fp.intersection(q["frames"].tolist())) >= COVISIBLE:
                out.append(float(p["d_clean"] @ q["d_clean"]))
    return np.array(out)


@lru_cache(maxsize=None)
def _cameras(seq: str):
    c = data.calib(seq).sort_values("frame_index")
    cams = {}
    for r in c.itertuples():
        if not r.pitch_usable:
            continue
        cams[int(r.frame_index)] = (np.asarray(r.K, float).reshape(3, 3), np.asarray(r.R, float).reshape(3, 3), np.asarray(r.C, float))
    return cams


def _project(cam, x, y):
    K, R, C = cam
    p = K @ (R @ (np.array([x, y, 0.0]) - C))
    if p[2] <= 1e-6:
        return None
    return p[0] / p[2], p[1] / p[2]


def _pan(cam) -> float:
    """Pan angle (degrees) of the optical axis around the vertical."""
    _, R, _ = cam
    z = R[2]  # optical axis in world coordinates
    return float(np.degrees(np.arctan2(z[0], z[1])))


def reentry_pairs(seq: str) -> list[tuple[int, int, int]]:
    """(a, b, gap_frames) for every re-entry cut gbf made: a = previous piece of the plugin id."""
    P = piece_table(seq)
    out = []
    for e in data.refine(seq)["tracks"]["reentry_pieces"]:
        b = int(e["track_id"])
        a = b - 100000
        if a in P and b in P:
            out.append((a, b, int(e["gap_frames"])))
    return sorted(out, key=lambda x: (x[1] % 100000, x[1]))


def describe_pair(seq: str, a_id: int, b_id: int) -> dict:
    """All features of an (earlier a, later b) pair of pieces; works for any pair, not only re-entries."""
    P = piece_table(seq)
    a, b = P[a_id], P[b_id]
    imp = impostors(seq)
    gap_f = b["first"] - a["last"] - 1
    gap_s = max(gap_f, 1) / FPS
    f = {"a": a_id, "b": b_id, "plugin": b["plugin"], "k": b_id // 100000, "same_plugin": int(a["plugin"] == b["plugin"]),
         "gap_f": gap_f, "gap_s": gap_s,
         "rows_a": a["n"], "rows_b": b["n"], "span_a": a["last"] - a["first"] + 1, "span_b": b["last"] - b["first"] + 1,
         "clean_a": a["n_clean"], "clean_b": b["n_clean"], "emb_a": a["n_emb"], "emb_b": b["n_emb"],
         "role_a": a["role"], "role_b": b["role"], "role_same": int(a["role"] == b["role"]),
         "team_a": a["team"], "team_b": b["team"],
         "team_same": 0 if a["team"] is None or b["team"] is None else (1 if a["team"] == b["team"] else -1),
         "teamv_a": a["teamv"], "teamv_b": b["teamv"]}  # fmt: skip
    # kit colour (gbf team check)
    f["kit_d"] = float(np.linalg.norm(a["lab"] - b["lab"])) if a["lab"] is not None and b["lab"] is not None else np.nan
    proto = data.refine(seq)["teams"].get("prototypes") or {}
    labs = [np.array(v["lab"], float) for v in proto.values() if v and v.get("lab")]
    pd_ = float(np.linalg.norm(labs[0] - labs[1])) if len(labs) == 2 else np.nan
    f["kit_rel"] = f["kit_d"] / pd_ if pd_ and np.isfinite(pd_) and pd_ > 0 else np.nan
    f["jersey_a"], f["jersey_b"] = a["jersey"], b["jersey"]
    f["jersey_same"] = 0 if a["jersey"] is None or b["jersey"] is None else (1 if a["jersey"] == b["jersey"] else -1)

    # appearance
    s = cos(a, b)
    f["cos"] = s
    f["cos_clean"] = cos(a, b, "d_clean")
    f["cos_all"] = cos(a, b, "d_all")
    f["cos_near"] = float(a["d_tail"] @ b["d_head"]) if a["d_tail"] is not None and b["d_head"] is not None else np.nan
    f["q_pair"] = int(a["d_clean"] is not None) + int(b["d_clean"] is not None)
    if len(imp) and np.isfinite(s):
        f["imp_pct"] = float((imp >= s).mean())
        med, p95 = np.percentile(imp, 50), np.percentile(imp, 95)
        f["imp_z"] = float((s - med) / max(p95 - med, 1e-6))
    else:
        f["imp_pct"], f["imp_z"] = np.nan, np.nan

    others = [p for p in P.values() if p["id"] not in (a_id, b_id) and p["d"] is not None]
    earlier = [p for p in others if p["last"] < b["first"] and compatible(p, b)]
    later = [p for p in others if p["first"] > a["last"] and compatible(p, a)]
    sb = np.array([cos(p, b) for p in earlier])
    sa = np.array([cos(a, p) for p in later])
    f["n_alt_b"], f["n_alt_a"] = len(sb), len(sa)
    f["alt_b"] = float(sb.max()) if len(sb) else np.nan
    f["alt_a"] = float(sa.max()) if len(sa) else np.nan
    f["margin_b"] = s - f["alt_b"] if len(sb) and np.isfinite(s) else np.nan
    f["margin_a"] = s - f["alt_a"] if len(sa) and np.isfinite(s) else np.nan
    f["rank_b"] = 1 + int((sb > s).sum()) if np.isfinite(s) else np.nan
    f["rank_a"] = 1 + int((sa > s).sum()) if np.isfinite(s) else np.nan
    ms = [m for m in (f["margin_a"], f["margin_b"]) if np.isfinite(m)]
    f["margin"] = min(ms) if ms else np.nan
    edge_earlier = [p for p in earlier if _sides(p["last_box"])]
    se = np.array([cos(p, b) for p in edge_earlier])
    f["alt_b_edge"] = float(se.max()) if len(se) else np.nan
    f["margin_b_edge"] = s - f["alt_b_edge"] if len(se) and np.isfinite(s) else np.nan

    # geometry
    f["has_pos"] = int(a["end_xy"] is not None and b["start_xy"] is not None)
    if f["has_pos"]:
        d = float(np.hypot(b["start_xy"][0] - a["end_xy"][0], b["start_xy"][1] - a["end_xy"][1]))
        el = max(b["start_f"] - a["end_f"], 1) / FPS
        f["dist_m"], f["speed_req"] = d, d / el
        f["excess_m"] = max(0.0, d - SPEED * el - 2 * POS_ERR)
    else:
        f["dist_m"] = f["speed_req"] = f["excess_m"] = np.nan
    la, fb = a["last_box"], b["first_box"]
    f["side_a"], f["side_b"] = _sides(la), _sides(fb)
    f["edge_a"], f["edge_b"] = int(bool(f["side_a"])), int(bool(f["side_b"]))
    lr_a = {c for c in f["side_a"] if c in "LR"}
    lr_b = {c for c in f["side_b"] if c in "LR"}
    f["side_same"] = 0 if not lr_a or not lr_b else (1 if lr_a & lr_b else -1)
    f["dx_img"] = float(abs((fb.x1 + fb.x2) / 2 - (la.x1 + la.x2) / 2))
    cams = _cameras(seq)
    ca, cb = cams.get(a["last"]), cams.get(b["first"])
    f["pan_deg"] = abs((_pan(cb) - _pan(ca) + 180.0) % 360.0 - 180.0) if ca is not None and cb is not None else np.nan
    if a["end_xy"] is not None and gap_f >= 1:
        inside = n = 0
        for fr in range(a["last"] + 1, b["first"]):
            cam = cams.get(fr)
            if cam is None:
                continue
            n += 1
            uv = _project(cam, *a["end_xy"])
            if uv is not None and 30 <= uv[0] <= W - 30 and 0 <= uv[1] <= H + 30:
                inside += 1
        f["stay_in_view"] = inside / n if n else np.nan
    else:
        f["stay_in_view"] = np.nan
    # geometric rivals for b: other earlier-ended compatible pieces within reach of b's start
    riv = []
    if b["start_xy"] is not None:
        for p in P.values():
            if p["id"] in (a_id, b_id) or p["last"] >= b["first"] or not compatible(p, b) or p["end_xy"] is None:
                continue
            d = float(np.hypot(b["start_xy"][0] - p["end_xy"][0], b["start_xy"][1] - p["end_xy"][1]))
            el = max(b["start_f"] - p["end_f"], 1) / FPS
            if d <= SPEED * el + 2 * POS_ERR:
                riv.append(d)
    f["n_rival_geo"] = len(riv)
    f["d_rival_geo"] = (min(riv) - f["dist_m"]) if riv and f["has_pos"] else np.nan
    return f


@lru_cache(maxsize=None)
def _pair_features(seq: str) -> pd.DataFrame:
    rows = [describe_pair(seq, a, b) for a, b, _ in reentry_pairs(seq)]
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def _all_pair_features(seq: str) -> pd.DataFrame:
    P = piece_table(seq)
    ids = sorted(P, key=lambda i: (P[i]["first"], i))
    n = len(ids)
    first = np.array([P[i]["first"] for i in ids])
    last = np.array([P[i]["last"] for i in ids])
    has = np.array([P[i]["d"] is not None for i in ids])
    D = np.stack([P[i]["d"] if P[i]["d"] is not None else np.full(256, np.nan) for i in ids])
    S = D @ D.T
    cls = np.array([_cls(P[i]) for i in ids])
    team = np.array([P[i]["team"] or "" for i in ids])
    comp = (cls[:, None] == cls[None]) & ~((team[:, None] != "") & (team[None] != "") & (team[:, None] != team[None]))
    fwd = last[:, None] < first[None]
    cand = fwd & comp & has[:, None] & has[None]
    ex = np.array([P[i]["end_xy"] if P[i]["end_xy"] is not None else (np.nan, np.nan) for i in ids])
    sx = np.array([P[i]["start_xy"] if P[i]["start_xy"] is not None else (np.nan, np.nan) for i in ids])
    ef = np.array([P[i]["end_f"] if P[i]["end_f"] is not None else -1 for i in ids])
    sf = np.array([P[i]["start_f"] if P[i]["start_f"] is not None else -1 for i in ids])
    dist = np.hypot(sx[None, :, 0] - ex[:, None, 0], sx[None, :, 1] - ex[:, None, 1])
    el = np.maximum(sf[None] - ef[:, None], 1) / FPS
    imp = impostors(seq)
    reentry = {(a, b) for a, b, _ in reentry_pairs(seq)}
    Sm = np.where(cand, S, -np.inf)
    rows = []
    for ia in range(n):
        for ib in np.where(fwd[ia] & has[ia] & has)[0]:
            a, b = ids[ia], ids[ib]
            s_ab = S[ia, ib]
            col = np.delete(Sm[:, ib], ia)          # earlier candidates of b other than a
            row = np.delete(Sm[ia], ib)             # later candidates of a other than b
            alt_b = col.max() if np.isfinite(col.max()) else np.nan
            alt_a = row.max() if np.isfinite(row.max()) else np.nan
            rows.append({
                "a": a, "b": b, "plugin_join": int((a, b) in reentry), "same_plugin": int(P[a]["plugin"] == P[b]["plugin"]),
                "compatible": int(comp[ia, ib]), "gap_s": max(first[ib] - last[ia] - 1, 1) / FPS,
                "rows_a": P[a]["n"], "rows_b": P[b]["n"], "clean_a": P[a]["n_clean"], "clean_b": P[b]["n_clean"],
                "q_pair": int(P[a]["d_clean"] is not None) + int(P[b]["d_clean"] is not None),
                "team_same": 0 if not team[ia] or not team[ib] else (1 if team[ia] == team[ib] else -1),
                "role_same": int(P[a]["role"] == P[b]["role"]),
                "cos": s_ab, "imp_pct": float((imp >= s_ab).mean()) if len(imp) else np.nan,
                "alt_a": alt_a, "alt_b": alt_b, "margin_a": s_ab - alt_a, "margin_b": s_ab - alt_b,
                "rank_a": 1 + int((row > s_ab).sum()), "rank_b": 1 + int((col > s_ab).sum()),
                "dist_m": dist[ia, ib], "speed_req": dist[ia, ib] / el[ia, ib],
                "excess_m": max(0.0, dist[ia, ib] - SPEED * el[ia, ib] - 2 * POS_ERR) if np.isfinite(dist[ia, ib]) else np.nan,
                "keep_cos": s_ab if (a, b) in reentry else np.nan,
            })  # fmt: skip
    return pd.DataFrame(rows)


def pair_features(seq: str, which: str = "reentry") -> pd.DataFrame:
    """Label-free pair evidence.

    which="reentry" (default): one row per plugin re-entry join gbf cut (a = previous piece of the
        plugin id, b = the re-entry piece), with the full feature set of the module docstring.
        m_keep's rule: keep the join when cos >= t_keep (about 0.85; strict: 0.89 and both
        pieces >= 15 boxes).
    which="all": one row per (earlier piece a, later piece b) of the clip that both have a
        descriptor (a ends before b starts), with a lighter feature set and the flag
        plugin_join (1 when [a, b] is a plugin re-entry join; keep_cos = cos for those, else NaN).
    """
    if which == "reentry":
        return _pair_features(seq).copy()
    if which == "all":
        return _all_pair_features(seq).copy()
    raise ValueError(which)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    for s in data.DEV_CLIPS:
        df = pair_features(s)
        print(s, len(df))
        print(df[["a", "b", "gap_s", "rows_a", "rows_b", "cos", "q_pair", "imp_pct", "margin_a", "margin_b", "dist_m", "speed_req", "side_a", "side_b", "team_same", "stay_in_view"]].round(3).to_string())
