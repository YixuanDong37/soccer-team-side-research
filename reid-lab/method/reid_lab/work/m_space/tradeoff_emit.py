"""Which kinds of links to give out: pooled LOCO held-out counts when the linker emits
(2) only pairs that left at an image border AND came back at one [the delivered setting],
(1) pairs with at least one border, (0) every pair it accepts (in-view fragments too).
Same LOCO weights (params_loco.json), thresholds fixed.

    python -I tradeoff_emit.py        (after run_position_linker.py)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data, score  # noqa: E402
import features  # noqa: E402
import linker  # noqa: E402
from run_position_linker import FLOOR, GRID, MODE, fmt  # noqa: E402

HERE = Path(__file__).resolve().parent
loco = json.loads((HERE / "params_loco.json").read_text(encoding="utf-8"))
seqs = {s: linker.peel_sequence(features.pair_features(s), loco[s], MODE, FLOOR) for s in data.DEV_CLIPS}
out = {}
for emit in (2, 1, 0):
    for t in GRID:
        p = score.evaluate({s: {"links": linker.cut_sequence(q, t, emit)} for s, q in seqs.items()})["pooled"]
        out[f"emit{emit}|{t}"] = p
        print(f"emit>={emit} theta {t:.2f}: {fmt(p)}")
(HERE / "tradeoff_emit.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
