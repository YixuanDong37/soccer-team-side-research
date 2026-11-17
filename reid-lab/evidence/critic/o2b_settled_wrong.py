"""critic / o2b_settled_wrong.py -- DIAGNOSIS (labels): the returns that are "wrong" by the settled
measure (o2_settled.py, W = 50) next to the lab scorer's verdict for the same return."""
import sys
from collections import Counter

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/critic")
import common as C
from common import data, score
import o2_settled as S  # noqa: E402  (prints its tables again; harmless)

P = C.Tee(C.OUT / "o2b_settled_wrong.txt")
for name in ("fuse_r1 strict", "fuse_r1 balanced", "fuse_r1 recall", "fuse_r1 frozen strict setting"):
    m = S.methods[name]
    _, per = S.settled(m, 50, 0)
    lab_out = {}
    for seq in data.DEV_CLIPS:
        r = score.score_clip(seq, m[seq]["links"], m[seq]["cuts"])
        for ret, o in zip(score.clip(seq).returns, r["per_return"]):
            lab_out[(seq, ret["ref_id"], ret["last"])] = o
    P(f"\n== {name}: settled-wrong returns ==")
    cross = Counter()
    for seq, rid, last, first, o, scored in per:
        so = lab_out[(seq, rid, last)]
        cross[(so, o)] += 1
        if o == "wrong":
            P(f"  {seq} person {rid:2d} left {last:3d} back {first:3d} gap {(first - last - 1) / 25:5.1f} s | scorer: {so}")
    P("  scorer verdict x settled verdict:")
    for so in ("right", "wrong", "lost", "undetected"):
        P(f"    scorer {so:10s}: " + ", ".join(f"{k} {cross[(so, k)]}" for k in ("same", "earlier", "wrong", "lost", "nobox")))
    w = Counter(seq for seq, rid, last, first, o, scored in per if o == "wrong")
    P("  settled-wrong per clip:", dict(w))
