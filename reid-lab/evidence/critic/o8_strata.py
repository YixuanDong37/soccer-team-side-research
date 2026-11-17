"""critic / o8_strata.py -- DIAGNOSIS (labels): where do the returns of fuse_r1 go right, by the things
that will differ on the real clip: box size (resolution proxy), length of the absence, and how
many boxes the player has after he is back.

Verdict per return = the settled measure of o2_settled.py (W = 50 frames), all 167 annotated returns.
Box height = median height (720p px) of the person's matched system boxes in the 2 s after the return.
"""
import contextlib
import io
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/critic")
import common as C
from common import data, score
from gbf.evaluate import gsr

with contextlib.redirect_stdout(io.StringIO()):
    import o2_settled as S

P = C.Tee(C.OUT / "o8_strata.txt")
info = {}
for seq in data.DEV_CLIPS:
    ref = gsr.load_reference(C.TRAIN / seq / "Labels-GameState.json")
    t = data.tracks(seq)
    pred = t[["frame", "track_id", "x1", "y1", "x2", "y2"]].copy()
    for col in ("x1", "y1", "x2", "y2"):
        pred[col] = pred[col] * score.SCALE
    pairs, _ = gsr.match(ref, pred)
    prow = pairs["pred_row"].to_numpy(np.int64)
    pfr = pairs["frame"].to_numpy(np.int64)
    pref = pairs["ref_id"].to_numpy(np.int64)
    h = (t.y2.to_numpy() - t.y1.to_numpy())[prow]
    for r in score.clip(seq).returns:
        m = (pref == r["ref_id"]) & (pfr >= r["first"]) & (pfr < r["first"] + 50)
        ma = (pref == r["ref_id"]) & (pfr <= r["last"]) & (pfr > r["last"] - 50)
        info[(seq, r["ref_id"], r["last"])] = {"h": float(np.median(h[m])) if m.any() else np.nan,
                                               "ha": float(np.median(h[ma])) if ma.any() else np.nan,
                                               "n_after": int(m.sum()), "gap": r["gap"] / 25.0}


def table(name, key, bins, labels_):
    P(f"\n== {name} ==")
    for mname in ("fuse_r1 strict", "fuse_r1 balanced", "fuse_r1 recall"):
        _, per = S.settled(S.methods[mname], 50, 0)
        cnt = {lb: Counter() for lb in labels_}
        for seq, rid, last, first, o, scored in per:
            v = info[(seq, rid, last)][key]
            if v != v:
                continue
            lb = labels_[int(np.searchsorted(bins, v, side="right"))]
            cnt[lb][o] += 1
        P(f"  {mname}")
        for lb in labels_:
            c = cnt[lb]
            n = sum(c.values())
            if n:
                P(f"    {lb:14s} n {n:3d}: same {c['same']:3d} ({100 * c['same'] / n:3.0f}%), earlier {c['earlier']}, wrong {c['wrong']}, lost {c['lost']}, nobox {c['nobox']}")


hs = np.array([v["h"] for v in info.values() if v["h"] == v["h"]])
P(f"median box height after the return (720p px): quartiles {np.percentile(hs, [10, 25, 50, 75, 90]).round(0).tolist()} (10 / 25 / 50 / 75 / 90%)")
table("by box height after the return (720p px)", "h", [45, 60, 80], ["under 45 px", "45-60 px", "60-80 px", "80 px and up"])
table("by length of the absence", "gap", [2, 5, 10, 20], ["under 2 s", "2-5 s", "5-10 s", "10-20 s", "over 20 s"])
table("by matched boxes in the 2 s after the return", "n_after", [10, 25, 45], ["under 10", "10-24", "25-44", "45-50"])
