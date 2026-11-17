"""v_robust_1 / t1_probe_split_wrong.py -- (LABELS: diagnosis) which returns are wrong under split (gbf now), cuts only and the
fuse_r1 points: the same persons or new ones? And returns scored lost whose identity B is mixed (a wrong merge the return score
does not count).   -> out/t1_probe_split_wrong.txt"""
import sys
from collections import Counter
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import vr_common as vc
import vr_labels as vl
from vr_common import data
from reid_lab import score  # LABELS

M = {"split": score.baseline_links("split"), "plugin": score.baseline_links("plugin"), "mutual": score.baseline_links("mutual")}
for p in ("cuts_only", "strict", "balanced", "recall", "fixed_strict"):
    M[p] = vc.load_method(p)
out = []
wrong = {}
for name, m in M.items():
    w, lost_mixed, lost = [], 0, 0
    for seq in data.DEV_CLIPS:
        mm = m.get(seq) or {}
        clip, ids, g = vl.assign(seq, mm.get("links") or [], mm.get("cuts") or [])
        for r in vl.returns_table(seq, mm.get("links") or [], mm.get("cuts") or []):
            if r["outcome"] == "wrong":
                w.append((seq, r["ref_id"], r["last"], r["first"]))
            if r["outcome"] == "lost":
                lost += 1
                b = [x for x in clip.returns if x["ref_id"] == r["ref_id"] and x["first"] == r["first"]][0]["b"]
                cnt = Counter(clip.refs[g == g[b]].tolist())
                if any(k != r["ref_id"] and v >= 5 for k, v in cnt.items()):
                    lost_mixed += 1
    wrong[name] = set(w)
    out.append(f"{name}: wrong returns {sorted(w)}; lost {lost}, of which B's identity holds >= 5 boxes of another person: {lost_mixed}")
for p in ("strict", "balanced", "recall", "fixed_strict"):
    out.append(f"{p}: wrong returns also wrong under split: {sorted(wrong[p] & wrong['split'])}; under plugin: {sorted(wrong[p] & wrong['plugin'])}; "
               f"under mutual: {sorted(wrong[p] & wrong['mutual'])}; new (in none of the three baselines): {sorted(wrong[p] - wrong['split'] - wrong['plugin'] - wrong['mutual'])}")
(vc.OUT / "t1_probe_split_wrong.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
