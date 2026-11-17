"""LABEL-BASED helper (uses reid_lab.score): per plugin re-entry join, what the scorer says.

Only for (a) diagnosis and (b) fitting thresholds inside leave-one-clip-out loops: a fit for
held-out clip C may read the rows of the other 12 clips only (loco.py enforces that).
Never imported by features.py or by any code that produces links for a clip from that
clip's own labels.

Per join [a, b]:
  link      "right" / "wrong" / "unknown"  (score.py's link label for the single link)
  d_right   returns that turn right when only this link is added to gbf's split
  d_wrong   returns that turn wrong when only this link is added
"""
import sys
from functools import lru_cache

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
import pandas as pd  # noqa: E402
from reid_lab import data, score  # noqa: E402

import features  # noqa: E402


@lru_cache(maxsize=None)
def link_labels(seq: str) -> pd.DataFrame:
    base = score.score_clip(seq, [])
    rows = []
    for a, b, _ in features.reentry_pairs(seq):
        s = score.score_clip(seq, [[a, b]])
        lab = "right" if s["links"]["right"] else ("wrong" if s["links"]["wrong"] else "unknown")
        dr = sum(1 for x, y in zip(base["per_return"], s["per_return"]) if y == "right" and x != "right")
        dw = sum(1 for x, y in zip(base["per_return"], s["per_return"]) if y == "wrong" and x != "wrong")
        rows.append({"seq": seq, "a": a, "b": b, "link": lab, "d_right": dr, "d_wrong": dw,
                     "d_impure": s["impure_rows"] - base["impure_rows"]})  # fmt: skip
    return pd.DataFrame(rows)


def table(clips=data.DEV_CLIPS) -> pd.DataFrame:
    """Features + labels of every plugin re-entry join of the given clips."""
    out = []
    for seq in clips:
        f = features.pair_features(seq)
        if not len(f):
            continue
        f.insert(0, "seq", seq)
        out.append(f.merge(link_labels(seq), on=["seq", "a", "b"]))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    pd.set_option("display.width", 280)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 200)
    t = table()
    print(t.link.value_counts())
    print(t[["d_right", "d_wrong", "d_impure"]].sum())
    cols = ["seq", "a", "b", "link", "d_right", "d_wrong", "d_impure", "gap_s", "rows_a", "rows_b", "clean_a", "clean_b", "cos", "q_pair", "imp_pct",
            "margin_a", "margin_b", "rank_a", "rank_b", "dist_m", "speed_req", "excess_m", "side_a", "side_b", "team_same", "role_same", "kit_rel", "jersey_same", "stay_in_view", "pan_deg", "n_rival_geo"]
    print(t.sort_values(["link", "cos"])[cols].round(3).to_string())
    t.to_csv("diag_table.csv", index=False)
