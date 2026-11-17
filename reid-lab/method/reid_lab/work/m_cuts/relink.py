"""Label-free: the "mutual" appearance rule (appearance_match.py's candidate pairs + mutual best
with a 0.05 margin, no absolute bar) re-run on the sub-pieces that exist after cuts.

    mutual_links(seq, cuts)  -> [[earlier id, later id], ...] between (sub-)piece ids
    python -I relink.py      -> checks that with no cuts this reproduces appearance-match.json's
                                candidate pairs on all 13 clips (no labels involved)

Same rule as analysis/appearance_match.py: both are players with a verified team (the parent
piece's team check: consistent, no samples against, not cleared, nearer team = own team), the same
team, each with >= MIN_CLEAN clean rows; the later starts after the earlier ends, within MAX_GAP_S;
when both are placed on the pitch the later's first position is within SPEED x elapsed + 2 x
POS_ERR of the earlier's last. A sub-piece inherits its parent's team verdict.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
from reid_lab import data  # noqa: E402
import features as F  # noqa: E402

FPS = 25.0
MIN_CLEAN, MAX_GAP_S, SPEED, POS_ERR = 5, 10.0, 12.0, 2.0
MARGIN = 0.05


def _units(seq: str, cuts):
    t = data.tracks(seq)
    E = data.embeddings(seq)
    clean = data.clean_rows(seq)
    team_entry = {e["track_id"]: e for e in data.refine(seq)["teams"]["tracks"]}
    P = F.piece_table(seq)
    role = t.role.to_numpy()
    placed = (t.pitch_usable & t.in_pitch & t.pitch_x.notna()).to_numpy()
    frame = t.frame.to_numpy()
    px, py = t.pitch_x.to_numpy(float), t.pitch_y.to_numpy(float)

    def verified_team(tid):
        e = team_entry.get(tid)
        if not e or e["verdict"] != "consistent" or e.get("samples_against") or e.get("cleared"):
            return None
        return e["team"] if e.get("nearer_team") == e["team"] else None

    out = []
    for sid, s in F.subpieces(seq, cuts).items():
        rows = P[s["piece"]].rows[s["lo"]: s["hi"]]
        good = rows[clean[rows]]
        desc = None
        if len(good) >= MIN_CLEAN:
            d = E[good].mean(axis=0)
            desc = d / np.linalg.norm(d)
        pl = rows[placed[rows]]
        vals, counts = np.unique(role[rows].astype(str), return_counts=True)
        # appearance_match.py: value_counts().index[0] (most frequent; pieces have one role)
        out.append({"id": sid, "first": s["first"], "last": s["last"], "role": vals[np.argmax(counts)],
                    "team": verified_team(s["piece"]), "desc": desc,
                    "first_pos": (int(frame[pl[0]]), px[pl[0]], py[pl[0]]) if len(pl) else None,
                    "last_pos": (int(frame[pl[-1]]), px[pl[-1]], py[pl[-1]]) if len(pl) else None})  # fmt: skip
    return out


def candidate_pairs(seq: str, cuts=()):
    """[(earlier id, later id, cosine)] of the rule's allowed pairs among the (sub-)pieces."""
    units = [u for u in _units(seq, cuts) if u["role"] == "player" and u["team"] is not None and u["desc"] is not None]
    pairs = []
    for a in units:
        for b in units:
            if a is b or a["team"] != b["team"] or b["first"] <= a["last"]:
                continue
            if (b["first"] - a["last"]) / FPS > MAX_GAP_S:
                continue
            if a["last_pos"] is not None and b["first_pos"] is not None:
                (fa, xa, ya), (fb, xb, yb) = a["last_pos"], b["first_pos"]
                if np.hypot(xb - xa, yb - ya) > SPEED * (fb - fa) / FPS + 2 * POS_ERR:
                    continue
            pairs.append((a["id"], b["id"], float(a["desc"] @ b["desc"])))
    return pairs


def decide(pairs, margin: float = MARGIN, t_abs: float = -1.0):
    """Mutual best with a margin over the runner-up on both sides (and at least one runner-up)."""
    later, earlier = {}, {}
    for a, b, s in pairs:
        later.setdefault(a, []).append((s, b))
        earlier.setdefault(b, []).append((s, a))
    out = []
    for a, b, s in pairs:
        ob, oa = sorted(later[a], reverse=True), sorted(earlier[b], reverse=True)
        if ob[0][1] != b or oa[0][1] != a or s < t_abs or len(ob) < 2 or len(oa) < 2:
            continue
        if min(s - ob[1][0], s - oa[1][0]) >= margin:
            out.append([int(a), int(b)])
    return out


def mutual_links(seq: str, cuts=()):
    return decide(candidate_pairs(seq, cuts))


if __name__ == "__main__":
    bad = 0
    for seq in data.DEV_CLIPS:
        mine = sorted((a, b, round(s, 6)) for a, b, s in candidate_pairs(seq))
        ref = sorted((int(a), int(b), round(float(s), 6)) for a, b, s in data.appearance(seq)["candidate_pairs"])
        same_pairs = [x[:2] for x in mine] == [x[:2] for x in ref]
        max_diff = max((abs(x[2] - y[2]) for x, y in zip(mine, ref)), default=0.0) if same_pairs else None
        same_links = sorted(decide(mine)) == sorted(decide(ref))
        print(seq, "pairs", len(mine), len(ref), "same pairs:", same_pairs, "max |cos diff|:", max_diff, "same links:", same_links, "links", len(decide(mine)))
        bad += not (same_pairs and same_links)
    print("problems:", bad)
