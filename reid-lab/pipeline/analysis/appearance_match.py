"""Appearance matching of gbf pieces (prototype, audit only): could re-id embeddings re-join pieces?

Run in gbf's .venv: python -I appearance_match.py RUN_DIR EMB.npz AUDIT_LISTALL_RUN OUT.json [LABELS.json]

Pieces, teams, roles, shots and pitch positions come from the gbf run (refine stage); the per-
detection re-id embeddings (export_embeddings.py) are joined to the pieces' rows by frame, plugin
track and box. Nothing is changed: the output says which pairs the rule would join, and how its
thresholds were chosen. No frame is read.

Rule, for pieces A (earlier) and B (later) of one shot; all must hold:
- both are players with a verified team (team check consistent, no samples against, not cleared,
  nearer team = own team), on the same team, each with at least MIN_CLEAN clean rows (box at
  least MIN_BOX_H px tall, not touching the frame edge, not overlapping any other box);
- B starts after A ends, within MAX_GAP_S; when both are placed on the pitch, B's first position
  lies within reach of A's last (SPEED m/s x elapsed + 2 x POS_ERR m);
- cosine(desc A, desc B) >= t_abs, B is A's most similar candidate and A is B's, and on both
  sides the best beats the runner-up by at least margin.
desc: L2-normalised mean of the L2-normalised embeddings of the piece's clean rows.

Thresholds come from the same clip, without labels:
- impostors: two pieces of the same verified team seen together for >= COVISIBLE frames (two
  different people, the hardest kind: same kit);
- genuine: long verified pieces cut in two around their middle with an artificial gap;
- simulation: each such piece is replaced by its two halves among the real pieces and the rule
  runs; a link of the two halves is right, a link of a half to another piece is counted as wrong
  (conservatively: the other piece could be the same person), otherwise the rule abstains.

Human labels (optional LABELS.json, schema gbf-human-labels/1) are checked against this run (sha256
of refine/tracks.parquet, plugin id and first/last frame of every piece named) and are NOT used to
choose thresholds. The output's "labels" section says what the rule did with each labelled pair,
which pieces a label proves to be other people (seen at the same time as the labelled person),
what the rule decides once each labelled person's known appearances are taken out of everyone
else's candidates ("after_labels"), and the facts needed to ask about the plugin's re-entries.
"""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

RUN, EMB, LISTALL, OUT = (Path(p) for p in sys.argv[1:5])
LABELS = Path(sys.argv[5]) if len(sys.argv) > 5 else None
FPS, W, H = 25.0, 1280, 720
MIN_BOX_H, MIN_CLEAN, MAX_GAP_S = 40.0, 5, 10.0
SPEED, POS_ERR, COVISIBLE = 12.0, 2.0, 5
GAPS_S = (0.5, 1.0, 2.0, 4.0)

tracks = pd.read_parquet(RUN / "refine/tracks.parquet").reset_index(drop=True)
refine = json.loads((RUN / "refine/refine.json").read_text(encoding="utf-8"))
cuts = json.loads((RUN / "cuts.json").read_text(encoding="utf-8"))
bounds = np.array([0, *cuts["cuts"], cuts["frame_count"]])

# ------------------------------------------------------------------ embeddings onto rows
z = np.load(EMB)
milli = lambda v: int(np.rint(float(v) * 1000))  # noqa: E731
lookup = {
    (int(f), int(t), *(milli(v) for v in b)): i
    for i, (f, t, b) in enumerate(zip(z["frame"], z["track"], z["box"]))
}
emb_row = np.full(len(tracks), -1)
for r in tracks.itertuples():
    k = (int(r.frame), int(r.source_track_id), milli(r.x1), milli(r.y1), milli(r.x2), milli(r.y2))
    emb_row[r.Index] = lookup.get(k, -1)
unit = z["emb"].astype(np.float64)
unit /= np.linalg.norm(unit, axis=1, keepdims=True)

# ------------------------------------------------------------------ clean rows
x1, y1, x2, y2 = (tracks[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
edge = (x1 <= 1) | (y1 <= 1) | (x2 >= W - 2) | (y2 >= H - 2)
overlap = np.zeros(len(tracks), bool)
for _, idx in tracks.groupby("frame").indices.items():
    if len(idx) < 2:
        continue
    ix = np.minimum(x2[idx][:, None], x2[idx][None]) - np.maximum(x1[idx][:, None], x1[idx][None])
    iy = np.minimum(y2[idx][:, None], y2[idx][None]) - np.maximum(y1[idx][:, None], y1[idx][None])
    hit = (ix > 0) & (iy > 0)
    np.fill_diagonal(hit, False)
    overlap[idx] = hit.any(axis=1)
clean = (~edge) & (~overlap) & ((y2 - y1) >= MIN_BOX_H) & (emb_row >= 0)

# ------------------------------------------------------------------ pieces
team_entry = {e["track_id"]: e for e in refine["teams"]["tracks"]}


def verified_team(tid):
    e = team_entry.get(tid)
    if not e or e["verdict"] != "consistent" or e.get("samples_against") or e.get("cleared"):
        return None
    return e["team"] if e.get("nearer_team") == e["team"] else None


def describe(rows):
    good = rows[clean[rows]]
    if len(good) < MIN_CLEAN:
        return None
    d = unit[emb_row[good]].mean(axis=0)
    return d / np.linalg.norm(d)


def placed(rows):
    sub = tracks.loc[rows]
    sub = sub[sub.pitch_usable & sub.in_pitch & sub.pitch_x.notna()]
    if not len(sub):
        return None, None
    a, b = sub.iloc[0], sub.iloc[-1]
    return (int(a.frame), float(a.pitch_x), float(a.pitch_y)), (int(b.frame), float(b.pitch_x), float(b.pitch_y))


pieces = {}
for tid, rows in tracks.groupby("track_id").indices.items():
    if tid < 0:
        continue
    rows = np.sort(rows)
    frames = tracks.frame.to_numpy()[rows]
    order = np.argsort(frames, kind="stable")
    rows, frames = rows[order], frames[order]
    first_pos, last_pos = placed(rows)
    pieces[int(tid)] = {
        "id": int(tid), "rows": rows, "first": int(frames[0]), "last": int(frames[-1]),
        "frames": set(frames.tolist()),
        "shot": int(np.searchsorted(bounds, frames[0], side="right") - 1),
        "role": tracks.role.iloc[rows].value_counts().index[0],
        "team": verified_team(int(tid)), "desc": describe(rows),
        "n_clean": int(clean[rows].sum()), "first_pos": first_pos, "last_pos": last_pos,
    }  # fmt: skip


def eligible(p):
    return p["role"] == "player" and p["team"] is not None and p["desc"] is not None


def reach_ok(a, b):
    if a["last_pos"] is None or b["first_pos"] is None:
        return True
    (fa, xa, ya), (fb, xb, yb) = a["last_pos"], b["first_pos"]
    return np.hypot(xb - xa, yb - ya) <= SPEED * (fb - fa) / FPS + 2 * POS_ERR


def candidates(pool):
    """All allowed (A, B) pairs among the pool's eligible pieces of one shot."""
    pairs = []
    by_shot = {}
    for p in pool:
        if eligible(p):
            by_shot.setdefault(p["shot"], []).append(p)
    for group in by_shot.values():
        for a, b in product(group, group):
            if a is b or a["team"] != b["team"] or b["first"] <= a["last"]:
                continue
            if (b["first"] - a["last"]) / FPS > MAX_GAP_S or not reach_ok(a, b):
                continue
            pairs.append((a, b, float(a["desc"] @ b["desc"])))
    return pairs


def decide(pairs, t_abs, margin):
    """Links of the rule: mutual best, above t_abs, beating the runner-up by margin on both sides."""
    best_b, best_a = {}, {}
    for a, b, s in pairs:
        best_b.setdefault(a["id"], []).append((s, b["id"]))
        best_a.setdefault(b["id"], []).append((s, a["id"]))
    links = []
    for a, b, s in pairs:
        ob = sorted(best_b[a["id"]], reverse=True)
        oa = sorted(best_a[b["id"]], reverse=True)
        if ob[0][1] != b["id"] or oa[0][1] != a["id"] or s < t_abs:
            continue
        # without a runner-up there is nothing to beat: the pair is not distinguished from
        # other people, and same-team impostors reach above t_abs, so the rule abstains
        if len(ob) < 2 or len(oa) < 2:
            continue
        gap_b = s - ob[1][0]
        gap_a = s - oa[1][0]
        if min(gap_a, gap_b) >= margin:
            links.append({"earlier": a["id"], "later": b["id"], "cos": round(s, 4),
                          "margin": round(min(gap_a, gap_b), 4),
                          "gap_s": round((b["first"] - a["last"]) / FPS, 2)})  # fmt: skip
    return links


real = list(pieces.values())

# ------------------------------------------------------------------ impostors (label-free negatives)
impostor, cross_team = [], []
elig = [p for p in real if eligible(p)]
for i, p in enumerate(elig):
    for q in elig[i + 1:]:
        if p["shot"] != q["shot"] or len(p["frames"] & q["frames"]) < COVISIBLE:
            continue
        (impostor if p["team"] == q["team"] else cross_team).append(float(p["desc"] @ q["desc"]))

# ------------------------------------------------------------------ genuine halves and simulation
def halves(p, gap_s):
    rows, frames = p["rows"], tracks.frame.to_numpy()[p["rows"]]
    mid = np.median(frames)
    half_gap = gap_s * FPS / 2
    r1, r2 = rows[frames < mid - half_gap], rows[frames > mid + half_gap]
    if len(r1) == 0 or len(r2) == 0:
        return None
    d1, d2 = describe(r1), describe(r2)
    if d1 is None or d2 is None:
        return None
    f1, f2 = tracks.frame.to_numpy()[r1], tracks.frame.to_numpy()[r2]
    out = []
    for rr, ff, d, tag in ((r1, f1, d1, "a"), (r2, f2, d2, "b")):
        fp, lp = placed(rr)
        out.append(dict(p, id=f"{p['id']}{tag}", rows=rr, first=int(ff.min()), last=int(ff.max()),
                        frames=set(ff.tolist()), desc=d, first_pos=fp, last_pos=lp,
                        n_clean=int(clean[rr].sum())))  # fmt: skip
    return out


genuine, splits = {g: [] for g in GAPS_S}, []
for p in elig:
    for g in GAPS_S:
        h = halves(p, g)
        if h is None:
            continue
        genuine[g].append(float(h[0]["desc"] @ h[1]["desc"]))
        splits.append((p, g, h))

imp = np.array(impostor)
# no same-team pair seen together -> no impostor reference -> no threshold, the rule abstains
t_grid = sorted({round(float(np.percentile(imp, q)), 4) for q in (90, 95, 97.5, 99, 99.5, 100)}) if len(imp) else []
m_grid = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
simulation = []
for t_abs, margin in product(t_grid, m_grid):
    right = wrong = abstain = 0
    by_gap = {g: [0, 0, 0] for g in GAPS_S}
    for p, g, (h1, h2) in splits:
        pool = [q for q in real if q["id"] != p["id"]] + [h1, h2]
        involved = [lk for lk in decide(candidates(pool), t_abs, margin)
                    if h1["id"] in (lk["earlier"], lk["later"]) or h2["id"] in (lk["earlier"], lk["later"])]  # fmt: skip
        if any(lk["earlier"] == h1["id"] and lk["later"] == h2["id"] for lk in involved):
            right += 1; by_gap[g][0] += 1  # noqa: E702
        if any(not (lk["earlier"] == h1["id"] and lk["later"] == h2["id"]) for lk in involved):
            wrong += 1; by_gap[g][1] += 1  # noqa: E702
        if not involved:
            abstain += 1; by_gap[g][2] += 1  # noqa: E702
    simulation.append({"t_abs": t_abs, "margin": margin, "splits": len(splits), "right": right,
                       "wrong": wrong, "abstain": abstain,
                       "by_gap": {str(g): v for g, v in by_gap.items()}})  # fmt: skip

safe = [s for s in simulation if s["wrong"] == 0]
chosen = max(safe, key=lambda s: (s["right"], -s["t_abs"], -s["margin"])) if safe else None

# ------------------------------------------------------------------ checks asked for by the review
def upper95(n):
    """Exact one-sided 95% upper bound on a rate when 0 events were seen in n trials."""
    return round(1 - 0.05 ** (1 / n), 3) if n else None


def pair_status(pairs, a_id, b_id, t_abs, margin):
    """Why the rule does or does not link a_id -> b_id among these candidate pairs."""
    s_ab = next((s for a, b, s in pairs if a["id"] == a_id and b["id"] == b_id), None)
    if s_ab is None:
        return "not a candidate pair"
    later = sorted((s for a, b, s in pairs if a["id"] == a_id), reverse=True)
    earlier = sorted((s for a, b, s in pairs if b["id"] == b_id), reverse=True)
    if later[0] > s_ab or earlier[0] > s_ab:
        return "not mutual best"
    if s_ab < t_abs:
        return "below t_abs"
    if len(later) < 2 or len(earlier) < 2:
        return "no runner-up"
    if min(s_ab - later[1], s_ab - earlier[1]) < margin:
        return "margin too small"
    return "linked"


def simulate(split_list, t_abs, margin):
    right = wrong = abstain = 0
    for p, g, (h1, h2) in split_list:
        pool = [q for q in real if q["id"] != p["id"]] + [h1, h2]
        ids = (h1["id"], h2["id"])
        involved = [lk for lk in decide(candidates(pool), t_abs, margin)
                    if lk["earlier"] in ids or lk["later"] in ids]  # fmt: skip
        right += any(lk["earlier"] == h1["id"] and lk["later"] == h2["id"] for lk in involved)
        wrong += any(not (lk["earlier"] == h1["id"] and lk["later"] == h2["id"]) for lk in involved)
        abstain += not involved
    return right, wrong, abstain


checks = {}
if chosen:
    T, MG = chosen["t_abs"], chosen["margin"]
    per_piece, reasons = {}, {}
    for p, g, (h1, h2) in splits:
        pool = [q for q in real if q["id"] != p["id"]] + [h1, h2]
        pairs = candidates(pool)
        ids = (h1["id"], h2["id"])
        involved = [lk for lk in decide(pairs, T, MG) if lk["earlier"] in ids or lk["later"] in ids]
        rec = per_piece.setdefault(p["id"], {"shot": p["shot"], "right": 0, "wrong": 0, "abstain": 0})
        if any(lk["earlier"] == h1["id"] and lk["later"] == h2["id"] for lk in involved):
            rec["right"] += 1
        if any(not (lk["earlier"] == h1["id"] and lk["later"] == h2["id"]) for lk in involved):
            rec["wrong"] += 1
        if not involved:
            rec["abstain"] += 1
            why = pair_status(pairs, h1["id"], h2["id"], T, MG)
            reasons[why] = reasons.get(why, 0) + 1
    pieces_right = sorted(i for i, r in per_piece.items() if r["right"])
    pieces_wrong = sorted(i for i, r in per_piece.items() if r["wrong"])
    checks["clustered"] = {
        "split_pieces": len(per_piece), "pieces_ever_right": len(pieces_right),
        "pieces_ever_wrong": len(pieces_wrong), "pieces_right_ids": pieces_right,
        "upper95_wrong_per_link_if_independent": upper95(chosen["right"]),
        "upper95_wrong_per_linked_piece": upper95(len(pieces_right)),
        "abstain_reasons": reasons,
    }  # fmt: skip

    # the true partner is absent: keep one half only; every link of it is an upper bound on a false link
    absent, trials = [], 0
    for p, g, (h1, h2) in splits:
        for keep, tag in ((h1, "first half"), (h2, "second half")):
            trials += 1
            pool = [q for q in real if q["id"] != p["id"]] + [keep]
            for lk in decide(candidates(pool), T, MG):
                if keep["id"] in (lk["earlier"], lk["later"]):
                    other = lk["later"] if lk["earlier"] == keep["id"] else lk["earlier"]
                    absent.append({"piece": p["id"], "gap_s": g, "kept": tag, "other": other,
                                   "cos": lk["cos"], "pair_gap_s": lk["gap_s"]})  # fmt: skip
    distinct = sorted({(a["piece"], a["other"]) for a in absent})
    checks["absent_partner"] = {
        "trials": trials, "trials_with_a_link": len(absent),
        "distinct_pairs": [{"piece": pc, "other": ot,
                            "pair_gap_s": min(a["pair_gap_s"] for a in absent if (a["piece"], a["other"]) == (pc, ot)),
                            "cos": max(a["cos"] for a in absent if (a["piece"], a["other"]) == (pc, ot))}
                           for pc, ot in distinct],
    }  # fmt: skip

    # leave one shot out: pick (t_abs, margin) on the other shots, test on the held-out shot
    imp_shots = []
    for i, p in enumerate(elig):
        for q in elig[i + 1:]:
            if p["shot"] == q["shot"] and p["team"] == q["team"] and len(p["frames"] & q["frames"]) >= COVISIBLE:
                imp_shots.append((p["shot"], float(p["desc"] @ q["desc"])))
    loso = []
    for held in sorted({p["shot"] for p, g, h in splits}):
        train = [x for x in splits if x[0]["shot"] != held]
        test = [x for x in splits if x[0]["shot"] == held]
        imp_train = np.array([s for sh, s in imp_shots if sh != held])
        if not len(train) or not len(test) or not len(imp_train):
            continue
        grid = sorted({round(float(np.percentile(imp_train, q)), 4) for q in (90, 95, 97.5, 99, 99.5, 100)})
        best = None
        for t_abs, margin in product(grid, m_grid):
            r, w, a = simulate(train, t_abs, margin)
            if w == 0 and (best is None or r > best[2]):
                best = (t_abs, margin, r)
        if best is None:
            loso.append({"held_out_shot": held, "chosen": None})
            continue
        r, w, a = simulate(test, best[0], best[1])
        loso.append({"held_out_shot": held, "train_splits": len(train), "test_splits": len(test),
                     "chosen_on_train": {"t_abs": best[0], "margin": best[1], "right": best[2]},
                     "test": {"right": r, "wrong": w, "abstain": a}})  # fmt: skip
    checks["leave_one_shot_out"] = loso


def boot_desc(p, rng):
    good = p["rows"][clean[p["rows"]]]
    pick = rng.choice(good, size=len(good), replace=True)
    d = unit[emb_row[pick]].mean(axis=0)
    return d / np.linalg.norm(d)


# ------------------------------------------------------------------ real pairs
listall = json.loads((LISTALL / "refine/refine.json").read_text(encoding="utf-8"))["stitch_audit"]
audit_reason = {(e["earlier"], e["later"]): e.get("veto") or e.get("geometry") or e["decision"]
                for e in listall["abstentions"]}  # fmt: skip
real_pairs = candidates(real)
links = decide(real_pairs, chosen["t_abs"], chosen["margin"]) if chosen else []
jersey = tracks[tracks.jersey.notna()].groupby("track_id")["jersey"].first().to_dict()
for lk in links:
    lk["audit"] = audit_reason.get((lk["earlier"], lk["later"]))
    lk["rows"] = [len(pieces[lk["earlier"]]["rows"]), len(pieces[lk["later"]]["rows"])]
    lk["clean_rows"] = [pieces[lk["earlier"]]["n_clean"], pieces[lk["later"]]["n_clean"]]
    lk["jersey"] = [jersey.get(lk["earlier"]), jersey.get(lk["later"])]

reentry = {(p["source_track_id"] if p["track_id"] < 200000 else p["track_id"] - 100000, p["track_id"])
           for p in refine["tracks"]["reentry_pieces"]}  # fmt: skip
reentry_view = []
for a_id, b_id in sorted(reentry):
    a, b = pieces.get(a_id), pieces.get(b_id)
    why = []
    if a is None or b is None:
        why.append("piece missing")
    else:
        for tag, p in (("A", a), ("B", b)):
            if p["role"] != "player":
                why.append(f"{tag} is {p['role']}")
            elif p["team"] is None:
                why.append(f"{tag} team not verified")
            if p["desc"] is None:
                why.append(f"{tag} has {p['n_clean']} clean rows (< {MIN_CLEAN})")
    s = float(a["desc"] @ b["desc"]) if a and b and a["desc"] is not None and b["desc"] is not None else None
    reentry_view.append({"earlier": a_id, "later": b_id, "cos": None if s is None else round(s, 4),
                         "blocked_by": why, "linked": any(lk["earlier"] == a_id and lk["later"] == b_id for lk in links)})  # fmt: skip

if chosen and links:
    rng = np.random.default_rng(0)
    stability = []
    for lk in links:
        a_id, b_id = lk["earlier"], lk["later"]
        outcome, coss = {}, []
        for _ in range(1000):
            pool = [dict(q) for q in real]
            for q in pool:
                if q["id"] in (a_id, b_id):
                    q["desc"] = boot_desc(pieces[q["id"]], rng)
            pairs = candidates(pool)
            why = pair_status(pairs, a_id, b_id, chosen["t_abs"], chosen["margin"])
            outcome[why] = outcome.get(why, 0) + 1
            coss.append(next(s for a, b, s in pairs if a["id"] == a_id and b["id"] == b_id))
        stability.append({"pair": [a_id, b_id], "resamples": 1000, "outcomes": outcome,
                          "cos_p5_p50_p95": [round(float(np.percentile(coss, q)), 4) for q in (5, 50, 95)]})  # fmt: skip
    checks["link_stability_bootstrap"] = stability


def explain(a_id, b_id, top=3):
    """The best candidates on both sides of a pair, from the real candidate pairs."""
    later = sorted(((s, b["id"]) for a, b, s in real_pairs if a["id"] == a_id), reverse=True)[:top]
    earlier = sorted(((s, a["id"]) for a, b, s in real_pairs if b["id"] == b_id), reverse=True)[:top]
    return {"pair": [a_id, b_id], "best_later_for_A": [(i, round(s, 4)) for s, i in later],
            "best_earlier_for_B": [(i, round(s, 4)) for s, i in earlier]}  # fmt: skip


explained = [explain(lk["earlier"], lk["later"]) for lk in links]
explained += [explain(r["earlier"], r["later"]) for r in reentry_view if r["cos"] is not None and not r["blocked_by"]]
explained += [explain(100065, 77)] if 100065 in pieces and 77 in pieces else []

# cross-shot check on the pieces gbf kept a jersey number for
jersey_check = []
for tid, number in jersey.items():
    p = pieces.get(int(tid))
    if p is None or p["desc"] is None or p["team"] is None:
        continue
    others = [q for q in elig if q["shot"] != p["shot"] and q["team"] == p["team"]]
    ranked = sorted(((float(p["desc"] @ q["desc"]), q["id"]) for q in others), reverse=True)[:3]
    jersey_check.append({"piece": int(tid), "jersey": number, "shot": p["shot"],
                         "most_similar_in_other_shots": [(i, round(s, 4), jersey.get(i)) for s, i in ranked]})  # fmt: skip

# ------------------------------------------------------------------ human labels (optional)
proto = {k: v for k, v in (refine["teams"].get("prototypes") or {}).items() if v and v.get("lab")}
LIGHT = max(proto, key=lambda k: proto[k]["lab"][0]) if proto else None  # the team whose kit prototype is lighter
src_id = tracks.groupby("track_id")["source_track_id"].first()
gen_all = np.concatenate([np.array(v) for v in genuine.values()])


def where(s):
    """Where a cosine sits: share of genuine halves at or below it, same-team impostor pairs at or above it."""
    return {"genuine_halves_at_or_below": round(float((gen_all <= s).mean()), 3),
            "same_team_impostors_at_or_above": int((imp >= s).sum()), "same_team_impostor_pairs": len(imp)}  # fmt: skip


def kit(pid):
    """Shirt colour of a piece: from the team check's nearer prototype; when the check named no nearer
    team (too few samples), from the plugin's team label only, marked as such."""
    e = team_entry.get(pid)
    if LIGHT is None:
        return {"kit": None, "L": None, "team_check": e["verdict"] if e else None}
    if e and e.get("lab") and e.get("nearer_team"):
        return {"kit": "light" if e["nearer_team"] == LIGHT else "dark", "L": e["lab"][0],
                "team_check": e["verdict"]}  # fmt: skip
    col = tracks.team.iloc[pieces[pid]["rows"]].dropna()
    t = col.value_counts().index[0] if len(col) else None
    return {"kit": None if t is None else ("light" if t == LIGHT else "dark") + " (plugin label only)",
            "L": e["lab"][0] if e and e.get("lab") else None, "team_check": e["verdict"] if e else None}  # fmt: skip


NEAR_EDGE_PX = 10.0  # a box within this many pixels of the image border counts as near the edge


def ends(pid):
    """Where a piece enters and leaves: frame, time, image edges it touches (edge) or comes within
    NEAR_EDGE_PX of (near_edge), image region (thirds of the box centre), pitch position."""
    sub = tracks.loc[pieces[pid]["rows"]]  # rows are in frame order
    out = {"piece": pid, "plugin_id": int(src_id[pid]), "rows": len(sub), **kit(pid)}
    for tag, r in (("first", sub.iloc[0]), ("last", sub.iloc[-1])):
        sides = lambda m: [n for n, hit in (("left", r.x1 <= m), ("right", r.x2 >= W - 1 - m), ("top", r.y1 <= m),  # noqa: E731
                                            ("bottom", r.y2 >= H - 1 - m)) if hit]  # fmt: skip
        touch = sides(1.0)
        cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
        out[tag] = {"frame": int(r.frame), "time_s": round(r.frame / FPS, 2), "edge": touch,
                    "near_edge": sides(NEAR_EDGE_PX),
                    "region": [("left", "middle", "right")[min(2, int(3 * cx / W))],
                               ("upper", "middle", "lower")[min(2, int(3 * cy / H))]],
                    "pitch": [round(float(r.pitch_x), 1), round(float(r.pitch_y), 1)]
                    if r.pitch_usable and pd.notna(r.pitch_x) else None}  # fmt: skip
    return out


def top(pairs, pid, side, n=3):
    """The n most similar candidates of a piece among these pairs: later ones (side 'later') or earlier."""
    got = [(s, b["id"]) for a, b, s in pairs if a["id"] == pid] if side == "later" else \
          [(s, a["id"]) for a, b, s in pairs if b["id"] == pid]  # fmt: skip
    return [(i, round(s, 4)) for s, i in sorted(got, reverse=True)[:n]]


labels_out = None
if LABELS:
    if chosen is None:
        raise SystemExit("no threshold pair made zero wrong links, so there is no rule to check labels against")
    spec = json.loads(LABELS.read_text(encoding="utf-8"))
    if spec.get("schema") != "gbf-human-labels/1":
        raise SystemExit(f"{LABELS}: unknown schema {spec.get('schema')!r}")
    digest = hashlib.sha256((RUN / "refine/tracks.parquet").read_bytes()).hexdigest()
    if spec["applies_to"]["tracks_sha256"] != digest:
        raise SystemExit(f"{LABELS}: made for another refine output (tracks sha256 differs)")
    for lab in spec["labels"]:
        if lab["answer"] not in ("same_person", "different_people"):
            raise SystemExit(f"label {lab['id']}: unknown answer {lab['answer']!r}")
        if len(set(lab["pieces"])) != len(lab["pieces"]):
            raise SystemExit(f"label {lab['id']}: a piece is named twice")
        if "anchors" not in lab:
            raise SystemExit(f"label {lab['id']}: no anchors")
        if sorted(map(int, lab["anchors"])) != sorted(lab["pieces"]):
            raise SystemExit(f"label {lab['id']}: every piece needs an anchor")
        for pid, anc in lab["anchors"].items():
            p = pieces.get(int(pid))
            if p is None or (int(src_id[int(pid)]), p["first"], p["last"]) != (anc["plugin_id"], anc["first"], anc["last"]):
                raise SystemExit(f"label {lab['id']}: piece {pid} does not match this run")

    # same_person labels join pieces into people; a person is in one place at a time
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for lab in spec["labels"]:
        if lab["answer"] == "same_person":
            for q in lab["pieces"][1:]:
                parent[find(q)] = find(lab["pieces"][0])
    people = {}
    for x in list(parent):
        people.setdefault(find(x), []).append(x)
    people = [sorted(g, key=lambda i: pieces[i]["first"]) for g in people.values()]
    for g in people:
        for i, j in combinations(g, 2):
            if pieces[i]["frames"] & pieces[j]["frames"]:
                raise SystemExit(f"labels put pieces {i} and {j}, seen at the same time, into one person")
    person_of, prev_in, next_in = {}, {}, {}
    for gi, g in enumerate(people):
        for k, i in enumerate(g):
            person_of[i] = gi
            prev_in[i] = g[k - 1] if k else None
            next_in[i] = g[k + 1] if k + 1 < len(g) else None

    def person_key(x):
        """A labelled person, or the piece itself when no same_person label names it."""
        return ("person", person_of[x]) if x in person_of else ("piece", x)

    # different_people labels name two pieces; they hold for everything labelled the same as each
    different = set()
    for lab in spec["labels"]:
        if lab["answer"] != "different_people":
            continue
        if len(lab["pieces"]) != 2:
            raise SystemExit(f"label {lab['id']}: a different_people label names exactly two pieces")
        ka, kb = (person_key(x) for x in lab["pieces"])
        if ka == kb:
            raise SystemExit(f"label {lab['id']}: says different people, but same_person labels make them one")
        different.add(frozenset((ka, kb)))

    def seen_with(x, g):
        return any(x["frames"] & pieces[i]["frames"] for i in people[g])

    def why_dropped(a, b):
        """Why the labels rule out (a, b) as a still-open candidate pair; None if it stays open."""
        ga, gb = person_of.get(a["id"]), person_of.get(b["id"])
        if frozenset((person_key(a["id"]), person_key(b["id"]))) in different:
            return "labelled different people"
        if ga is not None and ga == gb:
            return "inside one labelled person (decided)"
        if ga is not None and gb is not None and any(
                pieces[i]["frames"] & pieces[j]["frames"] for i in people[ga] for j in people[gb]):
            return f"labelled people {people[ga]} and {people[gb]} are seen together"
        if ga is not None and seen_with(b, ga):
            return f"later piece seen together with labelled person {people[ga]}"
        if gb is not None and seen_with(a, gb):
            return f"earlier piece seen together with labelled person {people[gb]}"
        nb, pa = next_in.get(a["id"]), prev_in.get(b["id"])
        if nb is not None and b["last"] >= pieces[nb]["first"]:
            return f"earlier piece's next appearance is known ({nb})"
        if pa is not None and a["first"] <= pieces[pa]["last"]:
            return f"later piece's previous appearance is known ({pa})"
        return None

    T, MG = chosen["t_abs"], chosen["margin"]
    pairs_after = [(a, b, s) for a, b, s in real_pairs if why_dropped(a, b) is None]
    dropped = [{"earlier": a["id"], "later": b["id"], "cos": round(s, 4), "why": why_dropped(a, b)}
               for a, b, s in real_pairs if why_dropped(a, b) is not None]  # fmt: skip
    links_after = decide(pairs_after, T, MG)
    linked = {(lk["earlier"], lk["later"]) for lk in links}

    per_label = []
    for lab in spec["labels"]:
        rows_ = []
        ordered = sorted(lab["pieces"], key=lambda k: pieces[k]["first"])
        for i, j in combinations(ordered, 2):
            a, b = pieces[i], pieces[j]
            s = float(a["desc"] @ b["desc"]) if a["desc"] is not None and b["desc"] is not None else None
            status = pair_status(real_pairs, i, j, T, MG)
            same = lab["answer"] == "same_person"
            agreement = ("link right" if same else "link wrong") if (i, j) in linked else \
                        ("rule abstained" if same else "abstained rightly")  # fmt: skip
            rows_.append({"earlier": i, "later": j, "cos": None if s is None else round(s, 4),
                          "rule": status, "agreement": agreement,
                          "gap_s": round((b["first"] - a["last"]) / FPS, 2),
                          **({} if s is None else where(s))})  # fmt: skip
        g = person_of.get(lab["pieces"][0]) if lab["answer"] == "same_person" else None
        others = []
        if g is not None:
            shots = {pieces[i]["shot"] for i in people[g]}
            for p in real:
                if p["id"] in people[g] or p["shot"] not in shots or not seen_with(p, g):
                    continue
                others.append({"piece": p["id"], "plugin_id": int(src_id[p["id"]]), "role": p["role"],
                               **{k: v for k, v in kit(p["id"]).items() if k == "kit"}})  # fmt: skip
        per_label.append({"id": lab["id"], "answer": lab["answer"], "pieces": lab["pieces"], "pairs": rows_,
                          "seen_together_so_other_people": others,
                          "ends": [ends(i) for i in ordered]})  # fmt: skip

    # the plugin's own re-entries (one plugin id across a gap) and what the labels say about them
    plugin_reentries = []
    for a_id, b_id in sorted(reentry):
        if a_id not in pieces or b_id not in pieces:
            continue
        a, b = pieces[a_id], pieces[b_id]
        gb, ga = person_of.get(b_id), person_of.get(a_id)
        verdict = None
        if frozenset((person_key(a_id), person_key(b_id))) in different:
            verdict = "wrong: labelled different people"
        elif ga is not None and ga == gb:
            verdict = "confirmed by a label"
        elif gb is not None and seen_with(a, gb):
            verdict = f"wrong: the later piece is labelled person {people[gb]}, seen together with the earlier piece"
        elif ga is not None and seen_with(b, ga):
            verdict = f"wrong: the earlier piece is labelled person {people[ga]}, seen together with the later piece"
        s = float(a["desc"] @ b["desc"]) if a["desc"] is not None and b["desc"] is not None else None
        plugin_reentries.append({
            "earlier": a_id, "later": b_id, "gap_s": round((b["first"] - a["last"]) / FPS, 2),
            "cos": None if s is None else round(s, 4), "labels_say": verdict,
            "earlier_end": ends(a_id), "later_end": ends(b_id),
            "later_best_earlier_after_labels": top(pairs_after, b_id, "earlier"),
            "earlier_best_later_after_labels": top(pairs_after, a_id, "later"),
            "audit": audit_reason.get((a_id, b_id)),
        })  # fmt: skip

    labels_out = {
        "file": str(LABELS), "sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "people": people, "per_label": per_label,
        "after_labels": {
            "candidate_pairs": len(pairs_after), "dropped_pairs": dropped, "links": links_after,
            "new_links": [lk for lk in links_after if (lk["earlier"], lk["later"]) not in linked],
            "pieces_losing_candidates": sorted({d["earlier"] for d in dropped} | {d["later"] for d in dropped}),
        },
        "plugin_reentries": plugin_reentries,
    }  # fmt: skip


def edge_board(pairs, why=None):
    """Per shot: players and goalkeepers that leave the image at (or within NEAR_EDGE_PX of) its
    border before the shot ends, those that enter there after it starts, and for each entry the
    earlier exits that could be the same person (shirt colour family not different, within reach)
    with the cosine where both pieces have a descriptor and the rule's verdict on the pair among
    `pairs` (or, when `why` says the labels settle the pair, that reason). A review aid: team
    checks are not required here, unlike in the rule, so pieces with an inherited plugin team
    label show up. No maximum gap is applied."""
    near = lambda r: (r.x1 <= NEAR_EDGE_PX or r.y1 <= NEAR_EDGE_PX or r.x2 >= W - 1 - NEAR_EDGE_PX  # noqa: E731
                      or r.y2 >= H - 1 - NEAR_EDGE_PX)  # fmt: skip
    family = lambda pid: (kit(pid)["kit"] or "").split(" ")[0] or None  # noqa: E731
    board = []
    for s in range(len(bounds) - 1):
        lo, hi = int(bounds[s]), int(bounds[s + 1]) - 1
        ps = [p for p in real if p["shot"] == s and p["role"] in ("player", "goalkeeper")]
        exits = [p for p in ps if p["last"] < hi and near(tracks.iloc[p["rows"][-1]])]
        entries = [p for p in ps if p["first"] > lo and near(tracks.iloc[p["rows"][0]])]
        out = []
        for b in sorted(entries, key=lambda p: p["first"]):
            options = []
            for a in exits:
                fa, fb = family(a["id"]), family(b["id"])
                if a["last"] >= b["first"] or (fa and fb and fa != fb) or not reach_ok(a, b):
                    continue
                gap = (b["first"] - a["last"]) / FPS
                dist = None
                if a["last_pos"] is not None and b["first_pos"] is not None:
                    dist = float(np.hypot(b["first_pos"][1] - a["last_pos"][1], b["first_pos"][2] - a["last_pos"][2]))
                cos = None if a["desc"] is None or b["desc"] is None else float(a["desc"] @ b["desc"])
                options.append({"exit": a["id"], "gap_s": round(gap, 2), "dist_m": None if dist is None else round(dist, 1),
                                "cos": None if cos is None else round(cos, 4),
                                "rule": (f"labels: {why(a, b)}" if why and why(a, b) else
                                         pair_status(pairs, a["id"], b["id"], chosen["t_abs"], chosen["margin"]) if chosen else None),
                                **({} if cos is None else where(cos))})  # fmt: skip
            options.sort(key=lambda o: -1 if o["cos"] is None else o["cos"], reverse=True)
            out.append({"entry": b["id"], "options": options})
        board.append({"shot": s, "frames": [lo, hi], "exits": [ends(p["id"]) for p in sorted(exits, key=lambda p: p["last"])],
                      "entries": [ends(p["id"]) for p in sorted(entries, key=lambda p: p["first"])], "pairs": out})  # fmt: skip
    return board


def candidate_view(pid, pairs):
    """A piece's best candidates on both sides among these pairs, with where each one enters and leaves."""
    return {"piece": ends(pid), "best_later": top(pairs, pid, "later"), "best_earlier": top(pairs, pid, "earlier")}


if labels_out is not None:
    labels_out["candidates_after_labels"] = {
        str(pid): candidate_view(pid, pairs_after)
        for pid in sorted(set(labels_out["after_labels"]["pieces_losing_candidates"]) | {lk["later"] for lk in links_after}
                          | {lk["earlier"] for lk in links_after})
        if pid in pieces
    }  # fmt: skip

pct = lambda a, qs: {str(q): round(float(np.percentile(a, q)), 4) for q in qs} if len(a) else None  # noqa: E731
result = {
    "params": {"min_box_h": MIN_BOX_H, "min_clean_rows": MIN_CLEAN, "max_gap_s": MAX_GAP_S,
               "speed_mps": SPEED, "position_error_m": POS_ERR, "covisible_frames": COVISIBLE,
               "split_gaps_s": GAPS_S},  # fmt: skip
    "rows_with_embedding": int((emb_row >= 0).sum()), "rows": len(tracks), "clean_rows": int(clean.sum()),
    "pieces": len(pieces), "eligible_pieces": len(elig),
    "impostor_same_team": {"pairs": len(impostor), **(pct(imp, (0, 50, 90, 95, 99, 100)) or {})},
    "different_team_covisible": {"pairs": len(cross_team), **(pct(np.array(cross_team), (0, 50, 99, 100)) or {})},
    "genuine_halves": {str(g): {"pieces": len(v), **(pct(np.array(v), (0, 1, 5, 50, 100)) or {})} for g, v in genuine.items()},
    "genuine_all_gaps": {"pairs": sum(len(v) for v in genuine.values()),
                         **(pct(np.concatenate([np.array(v) for v in genuine.values()]), (0, 5, 50, 100)) or {})},
    "simulation": simulation, "chosen": chosen,
    "real_candidate_pairs": len(real_pairs), "links": links, "reentry_pairs": reentry_view,
    "explained": explained, "review_checks": checks,
    "jersey_cross_shot": jersey_check,
}
if labels_out is not None:
    result["labels"] = labels_out
result["edge_board"] = (edge_board(pairs_after, why_dropped) if labels_out is not None  # verdicts after the labels
                        else edge_board(real_pairs))
# every candidate pair the rule looked at (earlier, later, cosine), so other thresholds can be tried later
result["candidate_pairs"] = [[a["id"], b["id"], round(s, 6)] for a, b, s in real_pairs]
OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: result[k] for k in ("rows_with_embedding", "rows", "clean_rows", "pieces", "eligible_pieces",
                                         "impostor_same_team", "different_team_covisible", "genuine_halves",
                                         "chosen", "real_candidate_pairs")}, indent=1, default=str))  # fmt: skip
print("links", json.dumps(links, indent=1, default=str))
print("reentry", json.dumps(reentry_view, default=str))
print("jersey", json.dumps(jersey_check, default=str))
print("simulation (wrong == 0 rows, best first):")
for s in sorted(simulation, key=lambda s: (s["wrong"], -s["right"]))[:8]:
    print("  ", s["t_abs"], s["margin"], "right", s["right"], "wrong", s["wrong"], "abstain", s["abstain"], s["by_gap"])
if labels_out is not None:
    print("labels: people", labels_out["people"])
    for pl in labels_out["per_label"]:
        print("  ", pl["id"], pl["answer"], json.dumps(pl["pairs"]))
        print("   seen together (other people):", [o["piece"] for o in pl["seen_together_so_other_people"]])
    al = labels_out["after_labels"]
    print("after labels:", al["candidate_pairs"], "candidate pairs,", len(al["dropped_pairs"]), "dropped; links",
          json.dumps(al["links"]), "new", json.dumps(al["new_links"]))  # fmt: skip
    for r in labels_out["plugin_reentries"]:
        print("  reentry", r["earlier"], "->", r["later"], "cos", r["cos"], "|", r["labels_say"],
              "| later's best earlier", r["later_best_earlier_after_labels"])  # fmt: skip
