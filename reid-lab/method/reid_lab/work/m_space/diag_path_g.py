"""Why gbf's stitch audit (path G) made 0 stitches on the 13 dev clips: pooled counts from
refine.json stitch_audit. Label-free."""
import sys, json
from collections import Counter
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
tot = {k: Counter() for k in ("decisions", "vetoes", "geometry", "jersey")}
pairs = 0; diag = Counter()
per = {}
for seq in data.DEV_CLIPS:
    sa = data.refine(seq)["stitch_audit"]
    c = sa["counts"]; pairs += c["pairs"]
    for k in tot: tot[k].update(c[k])
    for k, v in sa["diagnostics"].items():
        if isinstance(v, (int, float)): diag[k] += v
    per[seq] = {"pairs": c["pairs"], "geometry": c["geometry"], "vetoes": c["vetoes"], "breaks": sa["diagnostics"]["camera_breaks"], "usable": sa["diagnostics"]["frames_pitch_usable"]}
    # evidence of the pairs that reached the window test
out = {"pairs": pairs, **{k: dict(v.most_common()) for k, v in tot.items()}, "diagnostics": dict(diag), "per_clip": per}
print(json.dumps(out, indent=1))
json.dump(out, open("diag_path_g.json", "w"), indent=1)
# look at the pairs that reached the per-frame test
for seq in data.DEV_CLIPS:
    sa = data.refine(seq)["stitch_audit"]
    for e in sa["abstentions"]:
        g = e.get("geometry")
        if g in ("contradiction", "missing_frames", "no_decision_frames", "too_few_decision_frames", "decision_span_short", "late_first_decision"):
            ev = e.get("geometry_evidence", {})
            print(seq, e["earlier"], e["later"], "gap", e["gap_frames"], e["plugin"], g, "cov", ev.get("covered_frames"), "dec", ev.get("decision_frames"), "unc", ev.get("uncovered"), "contr", ev.get("contradiction_frames"), (ev.get("contradictions") or [None])[0])
