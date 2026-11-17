"""DIAGNOSIS (labels through score): which links of a combined setting are wrong, and why."""
import sys
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
import pandas as pd
from reid_lab import data, score
import method, features
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
T, M, B, G = 0.86, float(sys.argv[1]) if len(sys.argv) > 1 else 0.03, 0.90, (None if len(sys.argv) > 2 and sys.argv[2] == "none" else 10.0)
rows = []
for s in data.DEV_CLIPS:
    kept = method.keep_links(s, T)
    links = kept + method.appearance_links(s, kept, M, B, max_gap_s=G)
    base = score.score_clip(s, [])
    full = score.score_clip(s, links)
    for a, b in links:
        r = score.score_clip(s, [[a, b]])
        lab = "right" if r["links"]["right"] else ("wrong" if r["links"]["wrong"] else "unknown")
        if lab != "wrong":
            continue
        f = features.describe_pair(s, a, b)
        rows.append({"seq": s, "a": a, "b": b, "kept": [a, b] in kept, "d_impure": r["impure_rows"] - base["impure_rows"],
                     "dw": sum(1 for x, y in zip(base["per_return"], r["per_return"]) if y == "wrong" and x != "wrong"),
                     **{k: f[k] for k in ("gap_s", "rows_a", "rows_b", "clean_a", "clean_b", "cos", "imp_pct", "dist_m", "speed_req", "side_a", "side_b", "stay_in_view")}})
    print(s, "returns", full["returns"], "wrong returns idx", [i for i, o in enumerate(full["per_return"]) if o == "wrong"], "baseline wrong idx", [i for i, o in enumerate(base["per_return"]) if o == "wrong"])
print(pd.DataFrame(rows).round(3).to_string())
