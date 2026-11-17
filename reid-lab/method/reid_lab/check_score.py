"""Check that score.py reproduces analysis/gsr_reentry_eval.py on every dev clip.

Run in gbf's .venv: python -I check_score.py
split (no links) must equal the check's "split", plugin re-joins its "plugin", mutual its
"split+G+A(mutual, no bar)" (path G made no stitch on these clips, which is also checked),
and the mutual links' right/wrong/unknown its link counts.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid_lab import data, score  # noqa: E402

bad = []
base = {name: score.baseline_links(name) for name in ("split", "plugin", "mutual")}
for seq in data.DEV_CLIPS:
    ev = json.loads((data.RUNS / seq / "reentry-eval.json").read_text(encoding="utf-8"))
    if ev["links"]["path_G"]["links"]:
        bad.append((seq, "path G stitched: compare against split+G instead"))
    for name, key in (("split", "split"), ("plugin", "plugin"), ("mutual", "split+G+A(mutual, no bar)")):
        s = score.score_clip(seq, base[name][seq]["links"])
        want = {o: ev["outcomes"][key].get(o, 0) for o in ("right", "wrong", "lost")}
        got = {o: s["returns"][o] for o in ("right", "wrong", "lost")}
        if want != got:
            bad.append((seq, name, "outcomes", want, got))
        if s["returns"]["undetected"] != ev["undetected_returns"] or s["returns"]["untracked"] != ev.get("untracked_returns", 0):
            bad.append((seq, name, "unscored", s["returns"]))
        if name == "mutual":
            w = ev["links"]["mutual_margin"]
            if (w["links"], w["right"], w["wrong"], w["unknown"]) != (s["links"]["n"], s["links"]["right"], s["links"]["wrong"], s["links"]["unknown"]):
                bad.append((seq, "mutual links", w, s["links"]))
for name, links in base.items():
    print(name, json.dumps(score.evaluate(links)["pooled"]))
print("problems:", bad if bad else "none")
sys.exit(1 if bad else 0)
