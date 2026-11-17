"""Shared helpers of the diag_failures diagnosis (rule 5c: labels are read freely HERE; nothing
in this directory is a method, and none of its numbers is a method result).

Per clip, Ctx holds: the tracks, embeddings, clean rows, the label matching recomputed exactly as
reid_lab.score does it (with the predicted row kept), the annotation attributes, and one
extended row per gbf piece (label-free facts + its annotated identity).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
from gbf.evaluate import gsr  # noqa: E402
from reid_lab import data, score  # noqa: E402

FPS = 25.0
W720, H720 = 1280, 720
MIN_CLEAN, MAX_GAP_S, SPEED, POS_ERR = 5, 10.0, 12.0, 2.0   # appearance_match.py's constants
MIN_MATCHED = score.MIN_MATCHED


class Ctx:
    pass


def _sides(x1, y1, x2, y2, w, h, m):
    s = [n for n, hit in (("left", x1 <= m), ("right", x2 >= w - 1 - m), ("top", y1 <= m), ("bottom", y2 >= h - 1 - m)) if hit]
    return "+".join(s) if s else "none"


def sides_1080(r, ref):
    return _sides(r.x1, r.y1, r.x2, r.y2, ref.width, ref.height, score.EDGE_PX)


def sides_720(r, m=data.EDGE_PX):
    return _sides(r.x1, r.y1, r.x2, r.y2, W720, H720, m)


@lru_cache(maxsize=None)
def ctx(seq: str) -> Ctx:
    c = Ctx()
    c.seq = seq
    c.t = data.tracks(seq)
    c.emb = data.embeddings(seq)
    c.clean = data.clean_rows(seq)
    c.has_emb = ~np.isnan(c.emb[:, 0])
    c.clip = score.clip(seq)
    c.ref = gsr.load_reference(score.TRAIN / seq / "Labels-GameState.json")
    # the same matching as score.clip, keeping the predicted row and the IoU
    pred = c.t[["frame", "track_id", "x1", "y1", "x2", "y2"]].copy()
    for col in ("x1", "y1", "x2", "y2"):
        pred[col] = pred[col] * score.SCALE
    c.pred1080 = pred
    pairs, c.on_distractor = gsr.match(c.ref, pred)
    assert (pairs["frame"].to_numpy() == c.clip.frames).all() and (pairs["ref_id"].to_numpy() == c.clip.refs).all()
    c.pairs = pairs.reset_index(drop=True)
    c.pred_row = pairs["pred_row"].to_numpy(np.int64)
    c.ref_of_predrow = np.full(len(c.t), -1, np.int64)
    c.ref_of_predrow[c.pred_row] = c.clip.refs
    c.iou_of_predrow = np.full(len(c.t), np.nan)
    c.iou_of_predrow[c.pred_row] = pairs["iou"].to_numpy()
    # annotation attributes per person
    doc = json.loads((score.TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    att: dict[int, dict] = {}
    for a in doc["annotations"]:
        if a.get("supercategory") != "object":
            continue
        d = att.setdefault(int(a["track_id"]), {"role": Counter(), "team": Counter(), "jersey": Counter()})
        at = a.get("attributes") or {}
        for k in ("role", "team", "jersey"):
            d[k][at.get(k)] += 1
    c.att = {rid: {k: v.most_common(1)[0][0] for k, v in d.items()} for rid, d in att.items()}
    c.people = c.ref.people
    c.people_by = {rid: g.sort_values("frame").reset_index(drop=True) for rid, g in c.people.groupby("ref_id")}
    c.team_entry = {int(e["track_id"]): e for e in data.refine(seq)["teams"]["tracks"]}
    c.P = _pieces(c)
    c.piece_rows = {int(p): np.sort(rows) for p, rows in c.t.groupby("track_id").indices.items() if p >= 0}
    cand = data.appearance(seq).get("candidate_pairs", [])
    c.cand = {(int(a), int(b)): float(s) for a, b, s in cand}
    return c


def verified_team(c: Ctx, pid: int):
    e = c.team_entry.get(pid)
    if not e or e["verdict"] != "consistent" or e.get("samples_against") or e.get("cleared"):
        return None
    return e["team"] if e.get("nearer_team") == e["team"] else None


def _unit_mean(v):
    d = v.mean(axis=0)
    return d / np.linalg.norm(d)


def _pieces(c: Ctx) -> dict[int, dict]:
    t = c.t
    out = {}
    frames = t.frame.to_numpy()
    for pid, rows in t.groupby("track_id").indices.items():
        if pid < 0:
            continue
        rows = np.sort(rows)
        rows = rows[np.argsort(frames[rows], kind="stable")]
        g = t.iloc[rows]
        first, last = g.iloc[0], g.iloc[-1]
        placed = g[g.pitch_usable & g.in_pitch & g.pitch_x.notna()]
        good = rows[c.clean[rows]]
        anyemb = rows[c.has_emb[rows]]
        refs = c.ref_of_predrow[rows]
        cnt = Counter(refs[refs >= 0].tolist())
        top = cnt.most_common(2)
        e = c.team_entry.get(int(pid))
        jers = g.jersey.dropna()
        role = g.role.value_counts().index[0] if g.role.notna().any() else None
        team = g.team.mode().iloc[0] if g.team.notna().any() else None
        out[int(pid)] = {
            "piece": int(pid), "plugin": int(first.source_track_id), "k": int(pid) // 100000,
            "first": int(first.frame), "last": int(last.frame), "rows": len(rows),
            "role": role, "team": team, "vteam": verified_team(c, int(pid)),
            "team_verdict": (e or {}).get("verdict"), "team_cleared": (e or {}).get("cleared"),
            "n_clean": int(len(good)), "n_emb": int(len(anyemb)),
            "desc": _unit_mean(c.emb[good]) if len(good) >= MIN_CLEAN else None,
            "desc_any": (_unit_mean(c.emb[good]) if len(good) >= MIN_CLEAN else (_unit_mean(c.emb[anyemb]) if len(anyemb) else None)),
            "desc_kind": "strict" if len(good) >= MIN_CLEAN else ("relaxed" if len(anyemb) else "none"),
            "starts_at_edge": sides_720(first) != "none", "ends_at_edge": sides_720(last) != "none",
            "start_side": sides_720(first), "end_side": sides_720(last),
            "first_pos": (int(placed.iloc[0].frame), float(placed.iloc[0].pitch_x), float(placed.iloc[0].pitch_y)) if len(placed) else None,
            "last_pos": (int(placed.iloc[-1].frame), float(placed.iloc[-1].pitch_x), float(placed.iloc[-1].pitch_y)) if len(placed) else None,
            "start_xy": (float(placed.head(5).pitch_x.median()), float(placed.head(5).pitch_y.median())) if len(placed) else None,
            "end_xy": (float(placed.tail(5).pitch_x.median()), float(placed.tail(5).pitch_y.median())) if len(placed) else None,
            "med_h": float((g.y2 - g.y1).median()),
            "jersey": jers.value_counts().index[0] if len(jers) else None,
            "jersey_reads": int((g.jersey_conf.fillna(0) > 0).sum()),
            "matched": int(sum(cnt.values())),
            "ident": int(top[0][0]) if top and top[0][1] >= MIN_MATCHED else None,
            "ident_any": int(top[0][0]) if top else None,
            "ident_n": int(top[0][1]) if top else 0,
            "purity": round(top[0][1] / sum(cnt.values()), 3) if top else None,
            "second": int(top[1][0]) if len(top) > 1 else None,
            "second_n": int(top[1][1]) if len(top) > 1 else 0,
        }  # fmt: skip
    return out


def eligible_why(p: dict) -> list[str]:
    """Why appearance_match.py does not take a piece as a candidate end ([] = eligible)."""
    why = []
    if p["role"] != "player":
        why.append(f"role {p['role']}")
    if p["vteam"] is None:
        why.append("team unverified")
    if p["desc"] is None:
        why.append("clean rows < 5")
    return why


def reach_ok(a: dict, b: dict) -> bool:
    if a["last_pos"] is None or b["first_pos"] is None:
        return True
    (fa, xa, ya), (fb, xb, yb) = a["last_pos"], b["first_pos"]
    return bool(np.hypot(xb - xa, yb - ya) <= SPEED * (fb - fa) / FPS + 2 * POS_ERR)


def pair_gate(a: dict, b: dict) -> list[str]:
    """Every reason why (a earlier, b later) is not an appearance_match.py candidate pair."""
    if a["piece"] == b["piece"]:
        return ["same piece"]
    why = [f"A {w}" for w in eligible_why(a)] + [f"B {w}" for w in eligible_why(b)]
    if a["vteam"] is not None and b["vteam"] is not None and a["vteam"] != b["vteam"]:
        why.append("verified teams differ")
    if b["first"] <= a["last"]:
        why.append("overlap in time")
    elif (b["first"] - a["last"]) / FPS > MAX_GAP_S:
        why.append("gap > 10 s")
    if b["first"] > a["last"] and not reach_ok(a, b):
        why.append("out of reach")
    return why


def cos(a: dict, b: dict, key="desc"):
    if a[key] is None or b[key] is None:
        return None
    return float(a[key] @ b[key])


def mutual_status(cand: dict, a_id: int, b_id: int, margin=0.05):
    """Status of the pair under the mutual rule + ranks and margins on both sides."""
    s = cand.get((a_id, b_id))
    later = sorted(((v, k[1]) for k, v in cand.items() if k[0] == a_id), reverse=True)
    earlier = sorted(((v, k[0]) for k, v in cand.items() if k[1] == b_id), reverse=True)
    out = {"n_later_of_a": len(later), "n_earlier_of_b": len(earlier),
           "a_best_later": later[0][1] if later else None, "a_best_later_cos": later[0][0] if later else None,
           "b_best_earlier": earlier[0][1] if earlier else None, "b_best_earlier_cos": earlier[0][0] if earlier else None,
           "rank_of_b_for_a": None, "rank_of_a_for_b": None, "margin_a_side": None, "margin_b_side": None}  # fmt: skip
    if s is None:
        out["status"] = "not a candidate pair"
        return out
    out["rank_of_b_for_a"] = 1 + sum(v > s for v, _ in later)
    out["rank_of_a_for_b"] = 1 + sum(v > s for v, _ in earlier)
    oth_a = [v for v, k in later if k != b_id]
    oth_b = [v for v, k in earlier if k != a_id]
    out["margin_a_side"] = s - max(oth_a) if oth_a else None
    out["margin_b_side"] = s - max(oth_b) if oth_b else None
    if out["rank_of_b_for_a"] > 1 or out["rank_of_a_for_b"] > 1:
        out["status"] = "not mutual best"
    elif not oth_a or not oth_b:
        out["status"] = "no runner-up"
    elif min(out["margin_a_side"], out["margin_b_side"]) < margin:
        out["status"] = "margin < 0.05"
    else:
        out["status"] = "linked"
    return out
