"""Why 30 annotated returns have no matched system box next to the gap ("undetected").
DIAGNOSIS ONLY (rule 5c). -> out/undetected.csv (one row per missing side)

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I undetected.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ctx, data, gsr, score  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SLACK = score.MATCH_SLACK
ret = pd.read_csv(OUT / "returns.csv")
rows = []
for seq in data.DEV_CLIPS:
    c = ctx(seq)
    pred = c.pred1080
    by_frame = pred.groupby("frame").indices
    boxes = pred[["x1", "y1", "x2", "y2"]].to_numpy()
    W, H = c.ref.width, c.ref.height
    for i, r in enumerate(c.clip.returns):
        if r["a"] is not None and r["b"] is not None:
            continue
        rid = r["ref_id"]
        g = c.people_by[rid]
        info = ret[(ret["clip"] == seq) & (ret["ret"] == i)].iloc[0]
        for tag, miss, frame0, step in (("A", r["a"] is None, r["last"], -1), ("B", r["b"] is None, r["first"], +1)):
            if not miss:
                continue
            win = [frame0 + step * k for k in range(SLACK + 1)]
            ann = g[g.frame.isin(win)]
            best_iou, best_other, n_any_box, n_iou30 = 0.0, 0, 0, 0
            for a in ann.itertuples():
                idx = by_frame.get(a.frame)
                if idx is None:
                    continue
                iou = gsr.box_iou(np.array([[a.x1, a.y1, a.x2, a.y2]]), boxes[idx])[0]
                j = int(np.argmax(iou))
                n_any_box += iou[j] > 0
                n_iou30 += iou[j] >= 0.3
                if iou[j] > best_iou:
                    best_iou = float(iou[j])
                # a predicted box that fits this person (IoU >= 0.5) but was given to another annotated person
                fit = idx[iou >= 0.5]
                best_other += int(any(c.ref_of_predrow[k] not in (-1, rid) for k in fit))
            edge = ann.iloc[(ann.frame - frame0).abs().argsort()].iloc[0] if len(ann) else None
            w, h = (edge.x2 - edge.x1, edge.y2 - edge.y1) if edge is not None else (np.nan, np.nan)
            wh_win = float(((ann.x2 - ann.x1) / (ann.y2 - ann.y1)).median()) if len(ann) else np.nan
            # the person's usual shape, away from the border (all his annotated boxes not near the border)
            inner = g[(g.x1 > 15) & (g.x2 < W - 16) & (g.y1 > 15) & (g.y2 < H - 16)]
            wh_usual = float(((inner.x2 - inner.x1) / (inner.y2 - inner.y1)).median()) if len(inner) else np.nan
            stretch = int(info.stretch_before_frames if tag == "A" else info.stretch_after_frames)
            # matched rows of the person in the stretch on that side (any piece)
            m = c.pairs[c.pairs.ref_id == rid]
            if tag == "A":
                fm = m[(m.frame <= r["last"]) & (m.frame > r["last"] - stretch)].frame
                nearest = int(r["last"] - fm.max()) if len(fm) else None
            else:
                fm = m[(m.frame >= r["first"]) & (m.frame < r["first"] + stretch)].frame
                nearest = int(fm.min() - r["first"]) if len(fm) else None
            if nearest is None:
                why = "never matched in this visible stretch" + (" (stretch <= 12 frames)" if stretch <= SLACK else "")
            elif best_other:
                why = "a fitting box exists but is matched to another person"
            elif best_iou >= 0.3:
                why = "box found but IoU < 0.5 (partial body), matched later"
            elif best_iou > 0:
                why = "only a poorly overlapping box, matched later"
            else:
                why = "no box at all near the gap, matched later"
            rows.append({
                "clip": seq, "ret": i, "ref_id": rid, "missing_side": tag, "gap_s": info.gap_s,
                "side": info.exit_side if tag == "A" else info.reentry_side,
                "stretch_frames": stretch, "annotated_frames_in_window": len(ann),
                "frames_with_any_overlap": int(n_any_box), "frames_with_iou>=0.3": int(n_iou30), "best_iou_in_window": round(best_iou, 2),
                "frames_fitting_box_taken_by_other": best_other,
                "ann_box_h_720p": round(float(h) / 1.5, 1), "ann_box_w_720p": round(float(w) / 1.5, 1),
                "ann_w_over_h_in_window": round(wh_win, 2), "ann_w_over_h_usual": round(wh_usual, 2),
                "nearest_matched_row_frames_from_gap": nearest,
                "own_piece_on_this_side": (info.anchor_a if tag == "A" else info.anchor_b),
                "own_piece_frames_from_gap": (info.anchor_a_lead_frames if tag == "A" else info.anchor_b_delay_frames),
                "reason": why,
            })  # fmt: skip
df = pd.DataFrame(rows)
df.to_csv(OUT / "undetected.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
print(df.drop(columns=["frames_with_any_overlap"]).to_string())
print(df.reason.value_counts())
print(pd.crosstab(df.missing_side, df.reason).T.to_string())
print("returns:", df.groupby(["clip", "ret"]).ngroups)
print("narrow (w/h in window < 0.6 x usual):", (df.ann_w_over_h_in_window < 0.6 * df.ann_w_over_h_usual).sum(), "of", len(df))
print("short box (< 40 px at 720p):", (df.ann_box_h_720p < 40).sum())
u = ret[ret.status != "scored"]
print("undetected returns with own pieces on both sides:", (u.anchor_a.notna() & u.anchor_b.notna()).sum(), "of", len(u))
print(u[["clip", "ret", "ref_id", "gap_s", "status", "anchor_a", "anchor_a_lead_frames", "anchor_b", "anchor_b_delay_frames", "anchors_candidate", "anchors_mutual_status", "anchors_cos_any", "true_minus_best_impostor", "anchors_same_plugin"]].to_string())
