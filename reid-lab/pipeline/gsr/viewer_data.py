"""Data for the local SN-GSR clip viewer (RUNS_DIR/viewer.html): what each clip's returns look like.

Run in gbf's .venv: python -I viewer_data.py RUNS_DIR SELECTION_JSON TRAIN_DIR SUMMARY_JSON

For every selected clip with a re-entry check (RUNS_DIR/<SEQ>/reentry-eval.json) this writes
RUNS_DIR/viewer/<SEQ>.js: gbf's boxes per frame with the plugin id and gbf's piece id (720p), the
check's returns, the annotated boxes of the people they involve (scaled to 720p), and for each
scored return the person who held the new id before (plugin and gbf pieces), recomputed with the
check's own matching and verified against its outcomes. RUNS_DIR/viewer/index.js lists every
selected clip with its batch progress (from batch.log) and carries the pooled numbers of
SUMMARY_JSON (gsr_summary.py). The files point at the clips' local videos and hold annotation
boxes: SN-GSR-2025 is for research only, so they stay on this computer (publish numbers only).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gbf.evaluate import gsr

FPS = 25
SCALE = 1.5          # annotation (1080p) -> video (720p) is / SCALE
MIN_MATCHED = 5      # as in gsr_reentry_eval.py
MATCH_SLACK = 12     # as in gsr_reentry_eval.py
EDGE_PX = 15.0       # as in gsr_reentry_eval.py
RUN_GAP = 10         # annotated frames closer than this are one on-screen stretch (the check's GAP_MIN)
LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (start|ok|FAILED) (SNGS-\d+) \((\d+)/(\d+)\)")
DONE = re.compile(r"^(SNGS-\d+) done in (\d+) s")


def progress(runs: Path, clips: list[str]) -> dict:
    """The latest full batch in batch.log: per clip status and minutes, and a finish estimate."""
    lines = (runs / "batch.log").read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, s in enumerate(lines) if (m := LINE.match(s)) and m[2] == "start" and m[4] == "1" and int(m[5]) == len(clips)]
    if not starts:
        raise SystemExit("batch.log has no batch over the selected clips")
    began, state, seconds, failed_rc = None, {}, {}, {}
    for s in lines[starts[-1]:]:
        if m := LINE.match(s):
            t = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S")
            began = began or t
            state[m[3]] = (m[2], t)
            if m[2] == "FAILED":
                failed_rc[m[3]] = s.rsplit("exit", 1)[-1].strip()
    for s in lines:  # the pilot clips finished in an earlier batch: take each clip's longest real run
        if m := DONE.match(s):
            seconds[m[1]] = max(seconds.get(m[1], 0), int(m[2]))
    stopped_note = next((s for s in lines[starts[-1]:] if "batch stopped at the user's request" in s), None)
    interrupted = set(re.findall(r"SNGS-\d+", stopped_note.split(";", 1)[-1])) if stopped_note else set()
    now = datetime.now()
    rows, running = [], None
    for k, c in enumerate(clips, 1):
        st, t = state.get(c, ("waiting", None))
        status = {"start": "running", "ok": "done", "FAILED": "failed"}.get(st, "waiting")
        if c in interrupted:
            status = "stopped"   # killed when the batch was stopped, not a pipeline failure
        elif stopped_note and status == "waiting":
            status = "not_run"
        if status == "running":
            running = {"seq": c, "since": t.strftime("%H:%M"), "elapsed_min": round((now - t).total_seconds() / 60, 1)}
        rows.append({"seq": c, "rank": k, "status": status, "minutes": round(seconds[c] / 60, 1) if seconds.get(c) else None,
                     "exit_code": failed_rc.get(c)})  # fmt: skip
    took = [r["minutes"] for r in rows if r["status"] == "done" and r["minutes"]]
    avg = sum(took) / len(took) if took else None
    left = sum(1 for r in rows if r["status"] == "waiting")
    eta = None
    if avg is not None and (left or running) and not stopped_note:
        rest = left * avg + (max(0.0, avg - running["elapsed_min"]) if running else 0.0)
        eta = (now + pd.Timedelta(minutes=rest)).strftime("%H:%M")
    return {"snapshot": now.strftime("%Y-%m-%d %H:%M"), "began": began.strftime("%H:%M") if began else None,
            "total": len(clips), "done": sum(r["status"] == "done" for r in rows),
            "failed": sum(r["status"] == "failed" for r in rows), "running": running,
            "stopped": stopped_note[1:20] if stopped_note else None, "interrupted": sorted(interrupted),
            "avg_min": round(avg, 1) if avg else None, "eta": eta, "clips": rows}  # fmt: skip


def stretches(frames: np.ndarray) -> list[list[int]]:
    """On-screen stretches [first, last] of sorted annotated frames, joining holes under RUN_GAP."""
    out: list[list[int]] = []
    for f in frames.tolist():
        if out and f - out[-1][1] - 1 < RUN_GAP:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def side(box, w: int, h: int) -> str | None:
    if box.x1 <= EDGE_PX:
        return "left"
    if box.x2 >= w - 1 - EDGE_PX:
        return "right"
    if box.y1 <= EDGE_PX:
        return "top"
    if box.y2 >= h - 1 - EDGE_PX:
        return "bottom"
    return None


def clip_data(seq: str, runs: Path, train: Path) -> dict:
    out_dir, labels = runs / seq, train / seq / "Labels-GameState.json"
    ev = json.loads((out_dir / "reentry-eval.json").read_text(encoding="utf-8"))
    ref = gsr.load_reference(labels)
    doc = json.loads(labels.read_text(encoding="utf-8"))
    attrs: dict[int, list[tuple]] = {}
    for row in doc.get("annotations") or []:
        if row.get("supercategory") == "object" and row.get("track_id") is not None:
            a = row.get("attributes") or {}
            attrs.setdefault(int(row["track_id"]), []).append((a.get("team"), a.get("jersey")))
    tracks = pd.read_parquet(out_dir / "gbf/refine/tracks.parquet").reset_index(drop=True)
    pred = tracks[["frame", "track_id", "x1", "y1", "x2", "y2"]].copy()
    for c in ("x1", "y1", "x2", "y2"):
        pred[c] = pred[c] * SCALE
    pairs, _ = gsr.match(ref, pred)
    pred_row = pairs["pred_row"].to_numpy(dtype=np.int64)
    frames = pairs["frame"].to_numpy(dtype=np.int64)
    refs = pairs["ref_id"].to_numpy(dtype=np.int64)
    piece = pairs["track_id"].to_numpy(dtype=np.int64)
    src = tracks["source_track_id"].to_numpy(dtype=np.int64)[pred_row] if len(pred_row) else np.zeros(0, np.int64)
    tracked = piece >= 0
    alone = -1 - np.arange(len(piece), dtype=np.int64)
    ids = {"plugin": np.where(tracked & (src >= 0), src, alone), "split": np.where(tracked, piece, alone)}
    row_of = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(frames, refs))}

    def find_row(rid: int, frame: int, step: int) -> int | None:
        for k in range(MATCH_SLACK + 1):
            i = row_of.get((frame + step * k, rid))
            if i is not None:
                return i
        return None

    people = ref.people
    W, H = ref.width, ref.height
    involved: set[int] = set()
    returns = []
    for r in ev["returns"]:
        g = people[people["ref_id"] == r["ref_id"]].set_index("frame")
        v = {k: r[k] for k in ("ref_id", "role", "last", "first", "gap", "edge_before", "edge_after", "plugin_a", "plugin_b", "piece_a", "piece_b")}
        v["exit_side"], v["return_side"] = side(g.loc[r["last"]], W, H), side(g.loc[r["first"]], W, H)
        v["outcome"] = r["outcome"]
        involved.add(r["ref_id"])
        if isinstance(r["outcome"], dict):
            a, b = find_row(r["ref_id"], r["last"], -1), find_row(r["ref_id"], r["first"], +1)
            v["a_frame"], v["b_frame"] = int(frames[a]), int(frames[b])
            v["before_owner"] = {}
            for name, g_ids in ids.items():
                before = (frames < frames[b]) & (g_ids == g_ids[b])
                counts = Counter(refs[before].tolist())
                # the check's own outcome rule, recomputed here as a consistency check
                if g_ids[a] == g_ids[b]:
                    mine = "right"
                elif not counts:
                    mine = "lost"
                else:
                    top, n = counts.most_common(1)[0]
                    mine = "right" if top == r["ref_id"] else ("wrong" if n >= MIN_MATCHED else "lost")
                if mine != r["outcome"][name]:
                    raise SystemExit(f"{seq} ref {r['ref_id']} frame {r['first']}: {name} recomputed {mine}, check says {r['outcome'][name]}")
                if mine == "wrong":
                    top = counts.most_common(1)[0][0]
                    fr = frames[before & (refs == top)]
                    v["before_owner"][name] = {"ref_id": int(top), "rows": int(counts[top]), "from": int(fr.min()), "to": int(fr.max())}
                    involved.add(int(top))
        returns.append(v)

    ann = {}
    for rid in sorted(involved):
        g = people[people["ref_id"] == rid].sort_values("frame")
        team, jersey = Counter(t for t, _ in attrs.get(rid, []) if t).most_common(1), Counter(j for _, j in attrs.get(rid, []) if j).most_common(1)
        boxes = [[int(f), round(x1 / SCALE, 1), round(y1 / SCALE, 1), round(x2 / SCALE, 1), round(y2 / SCALE, 1)]
                 for f, x1, y1, x2, y2 in g[["frame", "x1", "y1", "x2", "y2"]].itertuples(index=False)]  # fmt: skip
        ann[str(rid)] = {"role": str(g["role"].iloc[0]), "team": team[0][0] if team else None, "jersey": jersey[0][0] if jersey else None,
                         "stretches": stretches(g["frame"].to_numpy()), "boxes": boxes}  # fmt: skip

    t = tracks[["frame", "source_track_id", "track_id", "x1", "y1", "x2", "y2"]]
    boxes = [[int(f), int(p), int(q), round(x1), round(y1), round(x2), round(y2)] for f, p, q, x1, y1, x2, y2 in t.itertuples(index=False)]
    return {"seq": seq, "frames": len(ref.image_numbers), "fps": FPS, "video": f"{seq}/video.mp4", "width": 1280, "height": 720,
            "outcomes": ev["outcomes"], "links": ev["links"], "chosen": ev.get("chosen_thresholds_on_clip"),
            "returns_at_edge": ev["annotation"]["returns_at_edge"], "returns": returns, "people": ann, "boxes": boxes}  # fmt: skip


def main() -> int:
    runs, selection, train, summary = (Path(a) for a in sys.argv[1:5])
    clips = json.loads(selection.read_text(encoding="utf-8"))["selected"]
    view = runs / "viewer"
    view.mkdir(exist_ok=True)
    prog = progress(runs, clips)
    made = []
    for c in clips:
        if (runs / c / "reentry-eval.json").exists() and (runs / c / "video.mp4").exists():
            data = clip_data(c, runs, train)
            (view / f"{c}.js").write_text(f"window.CLIPS = window.CLIPS || {{}};\nwindow.CLIPS[{json.dumps(c)}] = {json.dumps(data, separators=(',', ':'))};\n", encoding="utf-8")
            made.append(c)
            for r in prog["clips"]:
                if r["seq"] == c:
                    r["outcomes"], r["returns_at_edge"], r["links"] = data["outcomes"], data["returns_at_edge"], data["links"]
    s = json.loads(summary.read_text(encoding="utf-8"))
    pooled = {k: s[k] for k in ("clips_evaluated", "returns_scored", "returns_undetected", "returns_untracked", "returns_outcomes",
                                "links", "returns_by_gap", "plugin_reentry_joins", "paired_differences_vs_plugin")}  # fmt: skip
    pooled["returns_at_edge"] = s["annotation"]["returns_at_edge"]
    index = {"progress": prog, "pooled": pooled, "with_data": made}
    (view / "index.js").write_text("window.INDEX = " + json.dumps(index, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
    print(f"viewer data for {len(made)} clip(s); {prog['done']}/{prog['total']} done, {prog['failed']} failed, eta {prog['eta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
