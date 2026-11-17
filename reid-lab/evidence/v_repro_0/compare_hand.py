"""v_repro / compare_hand.py -- compares hand_check.py's independent outcomes with reid_lab/score.py,
return by return (LABELS; evaluation only).

    python -I -B compare_hand.py METHOD.json HAND.json

For every dev clip: score.clip(seq).returns (person, gap) and score.score_clip(...)["per_return"]
against the hand list. Prints the differences, then the sampled returns side by side.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402

method = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["clips"]
hand = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
by_clip = {}
for k, r in enumerate(hand["returns"]):
    by_clip.setdefault(r["clip"], []).append((k, r))
diff = 0
n = 0
side = {}
tot = Counter()
for seq in data.DEV_CLIPS:
    c = score.clip(seq)
    m = method.get(seq) or {}
    s = score.score_clip(seq, m.get("links") or [], m.get("cuts") or [])
    mine = by_clip.get(seq, [])
    if len(mine) != len(c.returns):
        diff += 1
        print("DIFFERENT number of returns", seq, len(mine), len(c.returns))
        continue
    for (k, h), r, o in zip(mine, c.returns, s["per_return"]):
        n += 1
        tot[o] += 1
        same_gap = (h["ref"], h["last"], h["first"], h["gap"]) == (r["ref_id"], r["last"], r["first"], r["gap"])
        fa = None if r["a"] is None else int(c.frames[r["a"]])
        fb = None if r["b"] is None else int(c.frames[r["b"]])
        pa = None if r["a"] is None else int(c.piece[r["a"]])
        pb = None if r["b"] is None else int(c.piece[r["b"]])
        ha = h["row_a"]["frame"] if isinstance(h["row_a"], dict) else h["row_a"]
        hb = h["row_b"]["frame"] if isinstance(h["row_b"], dict) else h["row_b"]
        same_rows = (ha, hb) == (fa, fb)
        if isinstance(h["row_a"], dict) and isinstance(h["row_b"], dict):
            same_rows &= (h["row_a"]["piece"], h["row_b"]["piece"]) == (pa, pb)
        if not (same_gap and same_rows and h["outcome"] == o):
            diff += 1
            print("DIFFERENT", seq, "hand:", h["ref"], h["last"], h["first"], ha, hb, h["outcome"], "| score.py:", r["ref_id"], r["last"], r["first"], fa, fb, o)
        side[k] = {"score_outcome": o, "score_row_a_frame": fa, "score_row_b_frame": fb, "score_piece_a": pa, "score_piece_b": pb}
print("returns compared:", n, "| score.py outcomes:", dict(tot), "| differences:", diff)
print()
print("the sampled scored returns (hand vs score.py):")
ok = 0
for k in hand["sample_index"]:
    h, s = hand["returns"][k], side[k]
    agree = h["outcome"] == s["score_outcome"]
    ok += agree
    print(f"{h['clip']} person {h['ref']:>2} gone {h['last'] + 1}..{h['first'] - 1} ({h['gap']} frames; edge before/after {h['edge_before']}/{h['edge_after']}) "
          f"A: frame {h['row_a']['frame']} id {h['row_a']['id_after_cuts']} | B: frame {h['row_b']['frame']} id {h['row_b']['id_after_cuts']} "
          f"| groups {h['row_a']['group']} / {h['row_b']['group']} | hand {h['outcome']} | score.py {s['score_outcome']} | {'AGREE' if agree else 'DISAGREE'}")
print(f"{ok} of {len(hand['sample_index'])} sampled returns agree")
