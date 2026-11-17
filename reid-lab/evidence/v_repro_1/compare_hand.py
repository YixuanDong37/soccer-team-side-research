"""v_repro_1 / compare_hand.py -- compare hand_check.py's outcomes with reid_lab/score.py, return by return.
LABELS (scoring check only).

    python -I -B compare_hand.py METHOD.json HAND.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402

method = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["clips"]
hand = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
mine = {(r["clip"], r["person"], r["last"], r["first"]): r for r in hand["returns"]}
picks_path = Path(sys.argv[2]).with_suffix(".picks.json")
picks = json.loads(picks_path.read_text(encoding="utf-8")) if picks_path.exists() else []
theirs = {}
bad = []
for seq in data.DEV_CLIPS:
    m = method.get(seq) or {}
    s = score.score_clip(seq, m.get("links") or [], m.get("cuts") or [])
    c = score.clip(seq)
    for r, o in zip(c.returns, s["per_return"]):
        theirs[(seq, r["ref_id"], r["last"], r["first"])] = o
        h = mine.get((seq, r["ref_id"], r["last"], r["first"]))
        if h is not None and h["A"] is not None and r["a"] is not None:
            # the same rows next to the gap?
            if h["A"]["frame"] != int(c.frames[r["a"]]) or h["A"]["piece"] != int(c.piece[r["a"]]):
                bad.append(("row A differs", seq, r["ref_id"], r["last"]))
        if h is not None and h["B"] is not None and r["b"] is not None:
            if h["B"]["frame"] != int(c.frames[r["b"]]) or h["B"]["piece"] != int(c.piece[r["b"]]):
                bad.append(("row B differs", seq, r["ref_id"], r["last"]))
    e = hand["per_clip"][seq]
    if (e["links_right"], e["links_wrong"], e["links_unknown"]) != (s["links"]["right"], s["links"]["wrong"], s["links"]["unknown"]):
        bad.append(("link verdicts differ", seq, e, s["links"]))
    if e["impure"] != s["impure_rows"] or e["tracked"] != s["tracked_rows"]:
        bad.append(("impure / tracked rows differ", seq, e["impure"], s["impure_rows"], e["tracked"], s["tracked_rows"]))
    if e["matched"] != len(c.frames):
        bad.append(("matched pairs differ", seq, e["matched"], len(c.frames)))
print("returns: hand", len(mine), "score.py", len(theirs), "| same set of returns:", set(mine) == set(theirs))
diff = [(k, mine[k]["outcome"], theirs[k]) for k in sorted(set(mine) & set(theirs)) if mine[k]["outcome"] != theirs[k]]
print("outcomes equal:", len(set(mine) & set(theirs)) - len(diff), "different:", len(diff))
for d in diff:
    print("  DIFFERENT", d)
print("score.py:", dict(Counter(theirs.values())), "| hand:", dict(Counter(r["outcome"] for r in mine.values())))
for p in picks:
    k = (p["clip"], p["person"], p["last"], p["first"])
    print(f"  pick #{p['k']} {p['clip']} person {p['person']} gap after frame {p['last']}: hand {p['outcome']}, score.py {theirs.get(k)}",
          "OK" if theirs.get(k) == p["outcome"] else "MISMATCH")
print("other problems:", bad if bad else "none")
sys.exit(1 if (diff or bad or set(mine) != set(theirs)) else 0)
