"""DIAGNOSIS: impure rows per clip of the held-out operating points, against cuts only and gbf now."""
import json, sys
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score
HERE = "C:/Research/plugins/tools/reid_lab/work/fuse/"
M = {p: json.load(open(HERE + f"METHOD_{p}.json", encoding="utf-8"))["clips"] for p in ("strict", "balanced", "recall")}
tot = {k: [0, 0] for k in ("split", "cuts", "strict", "balanced", "recall")}
print("clip      rows   gbf-now  cuts-only  strict  balanced  recall   (impure rows)")
for s in data.DEV_CLIPS:
    r = {"split": score.score_clip(s, [], []), "cuts": score.score_clip(s, [], M["strict"][s]["cuts"])}
    for p in M:
        r[p] = score.score_clip(s, M[p][s]["links"], M[p][s]["cuts"])
    for k in tot:
        tot[k][0] += r[k]["impure_rows"]; tot[k][1] += r[k]["tracked_rows"]
    print(f"{s}  {r['split']['tracked_rows']:6d}  {r['split']['impure_rows']:6d}  {r['cuts']['impure_rows']:6d}  {r['strict']['impure_rows']:6d}  {r['balanced']['impure_rows']:6d}  {r['recall']['impure_rows']:6d}")
print("pooled share: " + "  ".join(f"{k} {100 * a / b:.2f}%" for k, (a, b) in tot.items()))
a = {k: [0, 0] for k in tot}
