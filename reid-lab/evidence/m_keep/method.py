"""Label-free linking rules of agent m_keep.  Nothing here reads an annotation.

(a) keep_links(seq, t, ...):   which of the plugin's re-entry joins to keep.
        keep [a, b]  <=>  cos(a, b) >= t      (rule "cos";  features.pair_features)
    optional length condition: and both pieces have at least n_min boxes (n_min = 1: none)
    optional veto (rule "cos+reach"): not when the pitch distance between a's last and b's first
    position is beyond SPEED m/s * elapsed + 2 * POS_ERR (features.excess_m > 0)
    rule "imp": keep when imp_pct <= t (cosine calibrated on the clip's same-team strangers).

(b) appearance_links(seq, kept, margin, bar, ...): appearance links between what is left.
    The kept joins chain pieces into groups.  For groups X (earlier) and Y (later):
      - both are players/goalkeepers with the same majority team label, each with a descriptor
        from >= MIN_CLEAN clean rows; Y starts after X ends, at most max_gap_s later; when both
        are placed on the pitch Y's first position is within reach of X's last;
      - link when X is Y's most similar earlier candidate and Y is X's most similar later one,
        cosine >= bar, and on both sides the best beats the runner-up by >= margin
        (no runner-up on a side: abstain, like the mutual baseline).
    A link joins X's last piece to Y's first piece, so every piece gets at most one earlier and
    one later partner and linked groups never overlap in time (every link goes forward in time).

    source="lab" uses appearance_match.py's candidate pairs instead (the mutual baseline's own
    candidates: verified teams, gap <= 10 s), minus pairs made impossible by the kept joins.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
from reid_lab import data  # noqa: E402

import features  # noqa: E402

FPS = data.FPS
MIN_CLEAN = features.MIN_CLEAN
SPEED, POS_ERR = features.SPEED, features.POS_ERR


# ------------------------------------------------------------------ (a) selective keep
def keep_score(seq: str, rule: str = "cos"):
    """DataFrame a, b, score (higher = keep) and veto (bool) for every plugin re-entry join."""
    f = features.pair_features(seq)
    if not len(f):
        return f.assign(score=[], veto=[])
    if rule in ("cos", "cos+reach", "cos+team"):
        f["score"] = f["cos"]
    elif rule == "imp":
        f["score"] = -f["imp_pct"]
    elif rule == "near":
        f["score"] = f["cos_near"]
    elif rule == "min":
        f["score"] = np.fmin(f["cos"], f["cos_near"])
    else:
        raise ValueError(rule)
    f["veto"] = False
    if rule == "cos+reach":
        f["veto"] = f["excess_m"].fillna(0) > 0
    if rule == "cos+team":
        f["veto"] = f["team_same"] < 0
    return f


def keep_links(seq: str, t: float, rule: str = "cos", n_min: int = 1) -> list[list[int]]:
    """Kept plugin joins: score >= t, no veto, and both pieces have at least n_min boxes
    (n_min = 1: no length condition)."""
    f = keep_score(seq, rule)
    if not len(f):
        return []
    ok = (f["score"] >= t) & ~f["veto"] & f["score"].notna() & (f[["rows_a", "rows_b"]].min(axis=1) >= n_min)
    return [[int(a), int(b)] for a, b in f.loc[ok, ["a", "b"]].to_numpy()]


# ------------------------------------------------------------------ groups
def chains(seq: str, links) -> list[list[int]]:
    """Pieces chained by links (each link goes earlier -> later), every piece in one chain."""
    P = features.piece_table(seq)
    nxt, has_prev = {}, set()
    for a, b in links:
        a, b = int(a), int(b)
        if a in nxt or b in has_prev:
            raise ValueError(f"{seq}: piece linked twice on one side: {a} -> {b}")
        if P[a]["last"] >= P[b]["first"]:
            raise ValueError(f"{seq}: link {a} -> {b} does not go forward in time")
        nxt[a] = b
        has_prev.add(b)
    out = []
    for pid in sorted(P, key=lambda i: (P[i]["first"], i)):
        if pid in has_prev:
            continue
        ch = [pid]
        while ch[-1] in nxt:
            ch.append(nxt[ch[-1]])
        out.append(ch)
    return out


def group_table(seq: str, links) -> list[dict]:
    P = features.piece_table(seq)
    E = data.embeddings(seq)
    clean = data.clean_rows(seq)
    t = data.tracks(seq)
    out = []
    for ch in chains(seq, links):
        rows = np.concatenate([P[i]["rows"] for i in ch])
        c_rows = rows[clean[rows]]
        d = features._unit(E[c_rows].mean(axis=0)) if len(c_rows) >= MIN_CLEAN else None
        sub = t.iloc[rows]
        first, last = P[ch[0]], P[ch[-1]]
        out.append({
            "pieces": ch, "first": first["first"], "last": last["last"], "n_clean": len(c_rows), "d": d,
            "role": sub.role.mode().iloc[0] if sub.role.notna().any() else None,
            "team": sub.team.mode().iloc[0] if sub.team.notna().any() else None,
            "start_xy": first["start_xy"], "start_f": first["start_f"], "end_xy": last["end_xy"], "end_f": last["end_f"],
            "head": ch[0], "tail": ch[-1],
        })  # fmt: skip
    return out


def _reach_ok(x, y) -> bool:
    if x["end_xy"] is None or y["start_xy"] is None:
        return True
    d = float(np.hypot(y["start_xy"][0] - x["end_xy"][0], y["start_xy"][1] - x["end_xy"][1]))
    return d <= SPEED * (y["start_f"] - x["end_f"]) / FPS + 2 * POS_ERR


def group_candidates(seq: str, links, max_gap_s: float | None = 10.0, team: str = "label"):
    """(X, Y, cosine) for all allowed (earlier group X, later group Y)."""
    G = [g for g in group_table(seq, links) if g["d"] is not None and g["role"] in ("player", "goalkeeper")]
    if team == "label":
        G = [g for g in G if g["team"] is not None]
    out = []
    for x in G:
        for y in G:
            if x is y or y["first"] <= x["last"]:
                continue
            if x["team"] != y["team"] and not (team == "loose" and (x["team"] is None or y["team"] is None)):
                continue
            if max_gap_s is not None and (y["first"] - x["last"]) / FPS > max_gap_s:
                continue
            if not _reach_ok(x, y):
                continue
            out.append((x["tail"], y["head"], float(x["d"] @ y["d"])))
    return out


def lab_candidates(seq: str, links):
    """appearance_match.py's candidate pairs that the kept joins leave possible: the earlier
    piece has no kept later partner, the later piece no kept earlier partner."""
    has_next = {int(a) for a, b in links}
    has_prev = {int(b) for a, b in links}
    return [(int(a), int(b), float(s)) for a, b, s in data.appearance(seq).get("candidate_pairs", [])
            if int(a) not in has_next and int(b) not in has_prev]  # fmt: skip


def decide(pairs, bar: float, margin: float, lone: bool = False):
    """Mutual best, cosine >= bar, best beats runner-up by >= margin on both sides.
    lone=True also links a pair that has no runner-up on a side (the baseline abstains)."""
    later, earlier = {}, {}
    for a, b, s in pairs:
        later.setdefault(a, []).append((s, b))
        earlier.setdefault(b, []).append((s, a))
    out = []
    for a, b, s in pairs:
        ob, oa = sorted(later[a], reverse=True), sorted(earlier[b], reverse=True)
        if ob[0][1] != b or oa[0][1] != a or s < bar:
            continue
        if (len(ob) < 2 or len(oa) < 2) and not lone:
            continue
        gb = s - ob[1][0] if len(ob) > 1 else np.inf
        ga = s - oa[1][0] if len(oa) > 1 else np.inf
        if min(ga, gb) >= margin:
            out.append([int(a), int(b)])
    return out


def appearance_links(seq: str, kept, margin: float = 0.05, bar: float = -1.0, source: str = "groups",
                     max_gap_s: float | None = 10.0, team: str = "label", lone: bool = False) -> list[list[int]]:
    pairs = lab_candidates(seq, kept) if source == "lab" else group_candidates(seq, kept, max_gap_s, team)
    return decide(pairs, bar, margin, lone)


def combined(seq: str, t_keep: float, margin: float = 0.05, bar: float = -1.0, rule: str = "cos", **kw) -> list[list[int]]:
    kept = keep_links(seq, t_keep, rule)
    links = kept + appearance_links(seq, kept, margin, bar, **kw)
    chains(seq, links)  # raises on a conflict
    return links


def check_no_conflict(seq: str, links) -> None:
    """Every piece at most one earlier and one later partner (chains() raises otherwise);
    the pieces of a linked group follow each other in time without sharing a frame."""
    P = features.piece_table(seq)
    for ch in chains(seq, links):
        for i, j in zip(ch, ch[1:]):
            if P[i]["last"] >= P[j]["first"]:
                raise ValueError(f"{seq}: chain {ch} overlaps in time at {i} -> {j}")


def combined_iter(seq: str, t_keep: float, margin: float = 0.05, bar: float = -1.0, rule: str = "cos",
                  rounds: int = 10, **kw) -> list[list[int]]:
    """As combined(), but the appearance step is repeated on the merged groups until nothing
    new links (rounds=1 is combined() with source="groups")."""
    links = keep_links(seq, t_keep, rule)
    for _ in range(rounds):
        new = decide(group_candidates(seq, links, kw.get("max_gap_s", 10.0), kw.get("team", "label")), bar, margin, kw.get("lone", False))
        if not new:
            break
        links = links + new
        chains(seq, links)
    return links
