"""Label-free: write a METHOD.json for score.py from fixed parameters (no label is read here).

    python -I make_method.py NAME [--t-keep 0.85] [--n-min 1] [--no-appearance]
                                  [--margin 0.05] [--bar 0.90] [--max-gap 10]   -> out/NAME.json

Presets (the parameters loco.py fits on all 13 dev clips; their scores are IN-SAMPLE):
    python -I make_method.py keep_balanced  --t-keep 0.85 --no-appearance
    python -I make_method.py keep_strict    --t-keep 0.89 --n-min 15 --no-appearance
    python -I make_method.py comb_balanced  --t-keep 0.85
    python -I make_method.py comb_strict    --t-keep 0.89 --n-min 15
Then:  python -I C:/Research/plugins/tools/reid_lab/score.py out/NAME.json out/NAME.score.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
from reid_lab import data  # noqa: E402

import method  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--t-keep", type=float, default=0.85)
ap.add_argument("--n-min", type=int, default=1)
ap.add_argument("--no-appearance", action="store_true")
ap.add_argument("--margin", type=float, default=0.05)
ap.add_argument("--bar", type=float, default=0.90)
ap.add_argument("--max-gap", type=float, default=10.0, help="seconds; <= 0 for no limit")
a = ap.parse_args()

clips = {}
for seq in data.DEV_CLIPS:
    kept = method.keep_links(seq, a.t_keep, "cos", a.n_min)
    links = list(kept)
    if not a.no_appearance:
        links += method.appearance_links(seq, kept, a.margin, a.bar, source="groups", max_gap_s=a.max_gap if a.max_gap > 0 else None)
    method.check_no_conflict(seq, links)
    clips[seq] = {"links": links}
out = Path(__file__).resolve().parent / "out" / f"{a.name}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"name": a.name, "params": vars(a), "clips": clips}, indent=1) + "\n", encoding="utf-8")
print(out, {s: len(v["links"]) for s, v in clips.items()})
