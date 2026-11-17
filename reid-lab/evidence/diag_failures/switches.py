"""In-track identity switches: (1) the 9 scored returns that need a cut (row B, or row A and B, in a
piece that is mainly someone else or a mixed fragment): what the piece does right before the
person's rows start; (2) every gbf piece of the 13 clips holding two annotated people with >= 5
matched rows each. DIAGNOSIS ONLY (rule 5c). -> out/switch_returns.csv, out/mixed_pieces.csv

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I switches.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ctx, data, sides_720  # noqa: E402
from gbf.track_tools import boxes_touch_edge  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
ret = pd.read_csv(OUT / "returns.csv")
need = ret[ret.ceiling_class.str.contains("cut")]
rows = []
for r in need.itertuples():
    c = ctx(r.clip)
    pid = int(r.piece_b)
    idx = c.piece_rows[pid]
    t = c.t.iloc[idx].sort_values("frame")
    refs = pd.Series(c.ref_of_predrow[t.index.to_numpy()], index=t.index)
    fb = int(r.frame_b)
    here = t[t.frame == fb].iloc[0]
    prev = t[t.frame < fb]
    run = 0
    for f, ref in zip(t.frame[t.frame >= fb], refs[t.frame >= fb]):
        if ref != r.ref_id:
            break
        run += 1
    row = {"clip": r.clip, "ret": r.ret, "ref_id": r.ref_id, "piece_b": pid, "b_kind": r.b_kind, "piece_rows": len(t), "majority_person": c.P[pid]["ident_any"],
           "frame_b": fb, "person_run_rows_from_b": run, "rows_of_piece_after_run": int((t.frame > fb).sum()) - (run - 1),
           "box_b_side": sides_720(here), "out_split": r.out_split, "out_plugin": r.out_plugin, "ceiling_class": r.ceiling_class}  # fmt: skip
    if len(prev):
        p = prev.iloc[-1]
        row["frames_missing_before_b"] = int(fb - p.frame - 1)
        row["centre_jump_px"] = round(float(np.hypot((p.x1 + p.x2 - here.x1 - here.x2) / 2, (p.y1 + p.y2 - here.y1 - here.y2) / 2)), 1)
        row["prev_box_side"] = sides_720(p)
        e = boxes_touch_edge(np.array([[p.x1, p.y1, p.x2, p.y2], [here.x1, here.y1, here.x2, here.y2]]), (1280, 720))
        row["gbf_edge_touch_prev_or_b"] = bool(e.any())
        row["prev_person"] = int(refs[prev.index[-1]])
        if p.pitch_usable and here.pitch_usable and pd.notna(p.pitch_x) and pd.notna(here.pitch_x):
            d = float(np.hypot(here.pitch_x - p.pitch_x, here.pitch_y - p.pitch_y))
            row["pitch_jump_m"] = round(d, 1)
            row["implied_speed_mps"] = round(d / ((fb - p.frame) / 25), 1)
    else:
        row["frames_missing_before_b"] = None
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT / "switch_returns.csv", index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
print(df.to_string())

mixed = []
tot_rows = imp = 0
for seq in data.DEV_CLIPS:
    c = ctx(seq)
    for pid, idx in c.piece_rows.items():
        t = c.t.iloc[idx].sort_values("frame")
        refs = c.ref_of_predrow[t.index.to_numpy()]
        cnt = Counter(refs[refs >= 0].tolist())
        big = [(k, v) for k, v in cnt.most_common() if v >= 5]
        tot_rows += sum(cnt.values())
        if cnt:
            imp += sum(cnt.values()) - cnt.most_common(1)[0][1]
        if len(big) < 2:
            continue
        # the switch points: where the matched person changes between consecutive matched rows (runs >= 5)
        f = t.frame.to_numpy()[refs >= 0]
        rr = refs[refs >= 0]
        runs = []
        for i in range(len(rr)):
            if runs and runs[-1][0] == rr[i]:
                runs[-1][2] = int(f[i]); runs[-1][3] += 1
            else:
                runs.append([int(rr[i]), int(f[i]), int(f[i]), 1])
        runs = [x for x in runs if x[3] >= 5]
        for u, v in zip(runs[:-1], runs[1:]):
            if u[0] == v[0]:
                continue
            a = t[t.frame == u[2]].iloc[0]
            b = t[t.frame == v[1]].iloc[0]
            between = t[(t.frame > u[2]) & (t.frame < v[1])]
            fr_all = t.frame.to_numpy()
            seg = fr_all[(fr_all >= u[2]) & (fr_all <= v[1])]
            max_hole = int(np.diff(seg).max() - 1) if len(seg) > 1 else 0
            # was another box overlapping this track's box around the switch (a crossing)?
            cross = 0
            for fr in range(u[2] - 5, v[1] + 6):
                me = t[t.frame == fr]
                if not len(me):
                    continue
                me = me.iloc[0]
                oth = c.t[(c.t.frame == fr) & (c.t.track_id != pid)]
                ix = np.minimum(me.x2, oth.x2) - np.maximum(me.x1, oth.x1)
                iy = np.minimum(me.y2, oth.y2) - np.maximum(me.y1, oth.y1)
                cross += int(((ix > 0) & (iy > 0)).any())
            mixed.append({"clip": seq, "piece": pid, "rows": len(t), "from_person": u[0], "from_rows": u[3], "to_person": v[0], "to_rows": v[3],
                          "last_frame_from": u[2], "first_frame_to": v[1], "frames_between": v[1] - u[2] - 1, "rows_between": len(between),
                          "largest_hole_frames": max_hole, "frames_overlapping_another_box": cross,
                          "side_before": sides_720(a), "side_after": sides_720(b),
                          "centre_jump_px": round(float(np.hypot((a.x1 + a.x2 - b.x1 - b.x2) / 2, (a.y1 + a.y2 - b.y1 - b.y2) / 2)), 1),
                          "cos_segments": None})  # fmt: skip
            ra = t[(t.frame <= u[2])].index.to_numpy()
            rb = t[(t.frame >= v[1])].index.to_numpy()
            ga, gb = ra[c.clean[ra]], rb[c.clean[rb]]
            if len(ga) >= 5 and len(gb) >= 5:
                da, db = c.emb[ga].mean(0), c.emb[gb].mean(0)
                mixed[-1]["cos_segments"] = round(float(da @ db / np.linalg.norm(da) / np.linalg.norm(db)), 4)
mx = pd.DataFrame(mixed)
mx.to_csv(OUT / "mixed_pieces.csv", index=False)
print("matched rows", tot_rows, "impure", imp, round(imp / tot_rows, 4))
print("switch events (runs >= 5 rows on both sides):", len(mx), "in", mx.groupby(["clip", "piece"]).ngroups, "pieces; rows after the switches:", int(mx.to_rows.sum()))
print(mx.to_string())
print("largest hole >= 10 frames:", (mx.largest_hole_frames >= 10).sum(), "; overlapping another box around the switch:", (mx.frames_overlapping_another_box > 0).sum(), "; cos known:", mx.cos_segments.notna().sum(), "cos < 0.9:", (mx.cos_segments < 0.9).sum())
