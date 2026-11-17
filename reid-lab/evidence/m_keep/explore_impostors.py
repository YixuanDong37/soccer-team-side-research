"""Label-free: per-clip distribution of same-team co-visible piece-pair cosines (strangers)."""
import sys
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
import numpy as np
from reid_lab import data
import features
for s in data.DEV_CLIPS:
    imp = features.impostors(s)
    a = data.appearance(s)
    g = a.get("genuine_all_gaps") or {}
    print(s, "n", len(imp), "p50 %.3f p90 %.3f p95 %.3f p99 %.3f max %.3f" % tuple(np.percentile(imp, q) for q in (50, 90, 95, 99, 100)) if len(imp) else "none",
          "| lab impostor p95/p99/max", a["impostor_same_team"].get("95"), a["impostor_same_team"].get("99"), a["impostor_same_team"].get("100"), "| genuine halves p5/p50", g.get("5"), g.get("50"))
