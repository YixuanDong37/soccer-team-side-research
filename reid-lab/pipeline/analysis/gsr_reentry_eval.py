"""Re-entry check of one SN-GSR-2025 clip against its annotation (no frame is read).

Run in gbf's .venv: python -I gsr_reentry_eval.py LABELS.json RUN_DIR APPEARANCE.json OUT.json [--scale 1.5]

RUN_DIR is a gbf run on the clip's 720p video (refine/tracks.parquet, refine/refine.json,
cuts.json); APPEARANCE.json is appearance_match.py's output on that run (a stub with no links
when that script failed). Predicted boxes are scaled by --scale to the annotation's 1920x1080 and
matched with gbf.evaluate.gsr.match (per frame, Hungarian on IoU, pairs kept at IoU >= 0.5), the
project's own matching.

Definitions
- Untracked rows (gbf track id < 0) belong to no piece and no plugin identity.
- Piece identity: the annotated person (ref_id) of most of a gbf piece's matched rows, when it
  has at least MIN_MATCHED matched rows; purity is that share.
- Return: an annotated player or goalkeeper missing from the annotation for at least GAP_MIN
  frames (gbf's 0.4 s) between two annotated frames, whose box on the frame before or after the
  gap is within EDGE_PX of the image border (gbf's re-entry test uses the same "or"). Missing
  stretches away from the border are counted as occlusions and not scored.
- For a return of person R: row A = R's matched row on the last annotated frame before the gap
  (or the latest within MATCH_SLACK frames before it), row B = R's matched row on the first frame
  after it (or the earliest within MATCH_SLACK frames after). Without both rows the return is
  "undetected"; when A or B is untracked it is "untracked"; neither is scored.
- Outcome under an identity assignment g (a group per tracked row): right if g(A) == g(B);
  otherwise, among g(B)'s matched rows before B: none -> lost; most of them R's -> right (B
  rejoined an earlier stretch of R); most of them another person's, with at least MIN_MATCHED
  rows -> wrong; else lost. "contaminated" flags a group holding at least MIN_MATCHED rows of
  another person before B, whatever the outcome.
- Assignments: plugin = the plugin's track id; split = gbf's pieces; +G = pieces joined by the
  stitch audit's stitches with decision "stitch" (path G; pairs of a disabled path are reported
  apart); +A = also joined by appearance links: the rule as run on the clip (thresholds chosen on
  the clip), the private clip's thresholds (FIXED), and mutual best with a 0.05 margin and no
  absolute bar (MUTUAL, for comparison only).
- Link precision: a link between two pieces with identities is right when they are the same
  person.
- Plugin re-entries: for each piece gbf split off at a re-entry, the plugin's join was right when
  the piece and the plugin id's earlier piece have the same identity.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from gbf.evaluate import gsr

GAP_MIN = 10
EDGE_PX = 15.0
MIN_MATCHED = 5
MATCH_SLACK = 12
FIXED = (0.9732, 0.05)
MUTUAL_MARGIN = 0.05


def decide(pairs: list[tuple[int, int, float]], t_abs: float, margin: float) -> list[tuple[int, int]]:
    """appearance_match.decide on (earlier, later, cosine) triples."""
    later, earlier = {}, {}
    for a, b, s in pairs:
        later.setdefault(a, []).append((s, b))
        earlier.setdefault(b, []).append((s, a))
    links = []
    for a, b, s in pairs:
        ob, oa = sorted(later[a], reverse=True), sorted(earlier[b], reverse=True)
        if ob[0][1] != b or oa[0][1] != a or s < t_abs or len(ob) < 2 or len(oa) < 2:
            continue
        if min(s - ob[1][0], s - oa[1][0]) >= margin:
            links.append((a, b))
    return links


class Groups:
    def __init__(self, links: list[tuple[int, int]]):
        self.parent: dict[int, int] = {}
        for a, b in links:
            self.parent[self.find(a)] = self.find(b)

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("appearance", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--scale", type=float, default=1.5)
    args = parser.parse_args()

    ref = gsr.load_reference(args.labels)
    doc = json.loads(args.labels.read_text(encoding="utf-8"))
    attrs: dict[int, list[tuple]] = {}
    for row in doc.get("annotations") or []:
        if row.get("supercategory") == "object" and row.get("track_id") is not None:
            a = row.get("attributes") or {}
            attrs.setdefault(int(row["track_id"]), []).append((a.get("role"), a.get("team"), a.get("jersey")))
    tracks = pd.read_parquet(args.run / "refine/tracks.parquet").reset_index(drop=True)
    refine = json.loads((args.run / "refine/refine.json").read_text(encoding="utf-8"))
    cuts = json.loads((args.run / "cuts.json").read_text(encoding="utf-8")).get("cuts", [])
    appearance = json.loads(args.appearance.read_text(encoding="utf-8"))

    pred = tracks[["frame", "track_id", "x1", "y1", "x2", "y2"]].copy()
    for c in ("x1", "y1", "x2", "y2"):
        pred[c] = pred[c] * args.scale
    pairs, _ = gsr.match(ref, pred)
    pred_row = pairs["pred_row"].to_numpy(dtype=np.int64)
    frames = pairs["frame"].to_numpy(dtype=np.int64)
    refs = pairs["ref_id"].to_numpy(dtype=np.int64)
    piece = pairs["track_id"].to_numpy(dtype=np.int64)
    src = tracks["source_track_id"].to_numpy(dtype=np.int64)[pred_row] if len(pred_row) else np.zeros(0, np.int64)
    tracked = piece >= 0
    row_of = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(frames, refs))}

    # piece identities (tracked rows only)
    ident: dict[int, int] = {}
    purity: dict[int, float] = {}
    for tid in np.unique(piece[tracked]):
        counts = Counter(refs[piece == tid].tolist())
        top, n = counts.most_common(1)[0]
        if n >= MIN_MATCHED:
            ident[int(tid)] = int(top)
            purity[int(tid)] = n / sum(counts.values())

    # returns of annotated players and goalkeepers
    people = ref.people
    W, H = ref.width, ref.height
    near = lambda r: bool(r.x1 <= EDGE_PX or r.y1 <= EDGE_PX or r.x2 >= W - 1 - EDGE_PX or r.y2 >= H - 1 - EDGE_PX)  # noqa: E731
    returns, occlusions = [], 0
    for rid, g in people[people["role"].isin(["player", "goalkeeper"])].groupby("ref_id"):
        g = g.sort_values("frame")
        f = g["frame"].to_numpy()
        for i in np.where(np.diff(f) - 1 >= GAP_MIN)[0]:
            before, after = g.iloc[i], g.iloc[i + 1]
            eb, ea = near(before), near(after)
            if not (eb or ea):
                occlusions += 1
                continue
            returns.append({"ref_id": int(rid), "role": str(before.role), "last": int(f[i]), "first": int(f[i + 1]),
                            "gap": int(f[i + 1] - f[i] - 1), "edge_before": eb, "edge_after": ea,
                            "spans_detected_cut": any(f[i] < c <= f[i + 1] for c in cuts)})  # fmt: skip

    def find_row(rid: int, frame: int, step: int) -> int | None:
        for k in range(MATCH_SLACK + 1):
            i = row_of.get((frame + step * k, rid))
            if i is not None:
                return i
        return None

    # identity assignments: one group id per matched row; untracked rows get their own negative id
    audit = refine["stitch_audit"]
    stitches = [(int(s["earlier"]), int(s["later"])) for s in audit.get("stitches", []) if s.get("decision") == "stitch"]
    disabled = [(int(s["earlier"]), int(s["later"])) for s in audit.get("stitches", []) if s.get("decision") != "stitch"]
    cand = [(int(a), int(b), float(s)) for a, b, s in appearance.get("candidate_pairs", [])]
    links = {
        "clip_rule": [(int(lk["earlier"]), int(lk["later"])) for lk in appearance.get("links", [])],
        "fixed_rule": decide(cand, *FIXED),
        "mutual_margin": decide(cand, -1.0, MUTUAL_MARGIN),
    }
    alone = -1 - np.arange(len(piece), dtype=np.int64)  # a distinct id per untracked row
    assignments = {"plugin": np.where(tracked & (src >= 0), src, alone), "split": np.where(tracked, piece, alone)}
    for name, extra in (("split+G", []), ("split+G+A(clip rule)", links["clip_rule"]),
                        ("split+G+A(fixed rule)", links["fixed_rule"]), ("split+G+A(mutual, no bar)", links["mutual_margin"])):
        groups = Groups(stitches + extra)
        mapped = np.array([groups.find(int(p)) for p in piece], dtype=np.int64)
        assignments[name] = np.where(tracked, mapped, alone)

    scored = []
    for r in returns:
        a, b = find_row(r["ref_id"], r["last"], -1), find_row(r["ref_id"], r["first"], +1)
        r["piece_a"] = None if a is None else int(piece[a])
        r["piece_b"] = None if b is None else int(piece[b])
        r["plugin_a"] = None if a is None else int(src[a])
        r["plugin_b"] = None if b is None else int(src[b])
        if a is None or b is None:
            r["outcome"] = "undetected"
            continue
        if not (tracked[a] and tracked[b]):
            r["outcome"] = "untracked"
            continue
        r["outcome"], r["contaminated"] = {}, {}
        for name, g in assignments.items():
            before = (frames < frames[b]) & (g == g[b])
            counts = Counter(refs[before].tolist())
            others = {k: v for k, v in counts.items() if k != r["ref_id"]}
            r["contaminated"][name] = bool(others) and max(others.values()) >= MIN_MATCHED
            if g[a] == g[b]:
                r["outcome"][name] = "right"
            elif not counts:
                r["outcome"][name] = "lost"
            else:
                top, n = counts.most_common(1)[0]
                if top == r["ref_id"]:
                    r["outcome"][name] = "right"
                elif n >= MIN_MATCHED:
                    r["outcome"][name] = "wrong"
                else:
                    r["outcome"][name] = "lost"
        scored.append(r)

    def link_check(pairs_: list[tuple[int, int]]) -> dict:
        right = wrong = unknown = 0
        for a, b in pairs_:
            if a in ident and b in ident:
                right += ident[a] == ident[b]
                wrong += ident[a] != ident[b]
            else:
                unknown += 1
        return {"links": len(pairs_), "right": int(right), "wrong": int(wrong), "unknown": int(unknown)}

    reentry_joins = Counter()
    for p in refine["tracks"].get("reentry_pieces", []):
        tid, sid = int(p["track_id"]), int(p["source_track_id"])
        earlier = tracks[(tracks["source_track_id"] == sid) & (tracks["frame"] < int(p["start_frame"]))]
        prev = earlier.sort_values("frame")["track_id"].iloc[-1] if len(earlier) else None
        if prev is None or int(prev) not in ident or tid not in ident:
            reentry_joins["unknown"] += 1
        else:
            reentry_joins["plugin right (split broke a true identity)" if ident[int(prev)] == ident[tid]
                          else "plugin wrong (split removed a false identity)"] += 1  # fmt: skip

    # does the annotation keep one id per person? two player ids with the same team and jersey, never together
    def signature(rows: list[tuple]) -> tuple | None:
        roles = Counter(r[0] for r in rows)
        teams = Counter(r[1] for r in rows if r[1] not in (None, ""))
        jerseys = Counter(r[2] for r in rows if r[2] not in (None, "", "None"))
        if not ({"player", "goalkeeper"} & set(roles)) or not teams or not jerseys:
            return None
        return teams.most_common(1)[0][0], jerseys.most_common(1)[0][0]

    sig = {k: s for k, v in attrs.items() if (s := signature(v)) is not None}
    visible = {int(k): set(g["frame"]) for k, g in people.groupby("ref_id")}
    suspects = []
    ids = sorted(sig)
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            if sig[x] == sig[y] and not (visible.get(x, set()) & visible.get(y, set())):
                suspects.append([x, y, list(sig[x])])

    outcome_names = list(assignments)
    out = {
        "labels": str(args.labels), "run": str(args.run), "scale": args.scale,
        "params": {"gap_min_frames": GAP_MIN, "edge_px_1080p": EDGE_PX, "min_matched": MIN_MATCHED,
                   "match_slack": MATCH_SLACK, "fixed_rule": FIXED, "mutual_margin": MUTUAL_MARGIN},
        "annotation": {"frames": ref.n_frames, "people_ids": int(people["ref_id"].nunique()),
                       "player_gk_ids": int(people[people["role"].isin(["player", "goalkeeper"])]["ref_id"].nunique()),
                       "returns_at_edge": len(returns), "occlusion_gaps": occlusions,
                       "ids_with_team_and_jersey": len(sig), "same_team_and_jersey_on_two_ids": suspects},
        "gbf_cuts": cuts,
        "matching": {"predicted_rows": len(pred), "untracked_rows": int((tracks["track_id"] < 0).sum()),
                     "matched_rows": int(len(pairs)), "pieces": int(tracks.loc[tracks["track_id"] >= 0, "track_id"].nunique()),
                     "pieces_with_identity": len(ident),
                     "purity_p10_p50": [round(float(np.percentile(list(purity.values()), q)), 3) for q in (10, 50)] if purity else None},
        "appearance_failed": bool(appearance.get("failed")),
        "chosen_thresholds_on_clip": appearance.get("chosen"),
        "returns": returns,
        "outcomes": {name: dict(Counter(r["outcome"][name] for r in scored)) for name in outcome_names},
        "contaminated": {name: int(sum(r["contaminated"][name] for r in scored)) for name in outcome_names},
        "undetected_returns": sum(1 for r in returns if r["outcome"] == "undetected"),
        "untracked_returns": sum(1 for r in returns if r["outcome"] == "untracked"),
        "links": {"path_G": link_check(stitches), "disabled_path": link_check(disabled), **{k: link_check(v) for k, v in links.items()}},
        "link_detail": {k: [[a, b, ident.get(a), ident.get(b)] for a, b in v] for k, v in {"path_G": stitches, **links}.items()},
        "plugin_reentry_joins": dict(reentry_joins),
        "stitch_audit_counts": audit.get("counts"),
        "abstentions_listed_all": len(audit.get("abstentions", [])) == audit.get("abstentions_total"),
        "restrictions": ["derived from SN-GSR-2025 annotations: publish numbers only, never frames"],
    }  # fmt: skip
    args.out.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("annotation", "matching", "outcomes", "undetected_returns", "untracked_returns",
                                          "links", "plugin_reentry_joins")}, default=str))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
