"""Label-free check: fuse/linker.agglomerate without a bonus returns m_appear's links (agglo.agglomerate)
on every dev clip, with and without m_cuts' cuts."""
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse")
from reid_lab import data  # noqa: E402
import linker  # noqa: E402
import prep  # noqa: E402

prep._import("m_appear", "features")
agglo = prep._import("m_appear", "agglo")
bad = 0
for seq in data.DEV_CLIPS:
    for cuts in ((), prep.cuts_for(seq, seq)):
        for t_abs, margin in ((0.5, 0.25), (0.4, 0.15)):
            mine = sorted(map(tuple, linker.agglomerate(seq, cuts, None, t_abs=t_abs, margin=margin)))
            theirs = sorted(map(tuple, agglo.agglomerate(seq, kind="w", t_abs=t_abs, margin=margin, cuts=cuts)))
            if mine != theirs:
                bad += 1
                print("DIFFERENT", seq, len(cuts), t_abs, margin, len(mine), len(theirs))
    print(seq, "ok" if not bad else "", flush=True)
print("all equal" if not bad else f"{bad} differences")
