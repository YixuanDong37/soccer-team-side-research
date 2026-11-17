"""Debug: why are the gap-free switches not candidates?"""
import sys
sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np
import features as F

for seq, pid, fr in (("SNGS-097", 5, 491), ("SNGS-097", 8, 178), ("SNGS-162", 106, 654), ("SNGS-066", 50, 321), ("SNGS-102", 36, 347)):
    p = F.piece_table(seq)[pid]
    cs = F.boundary_cos(p)
    i = int(np.searchsorted(p.frame, fr))
    lo, hi = max(0, i - 12), min(len(p) - 1, i + 12)
    print(seq, pid, "boundary index", i, "n", len(p))
    for j in range(lo, hi):
        print(f"   row {j} frame {p.frame[j]} gap_after {p.frame[j+1]-p.frame[j]-1} clean {int(p.clean[j])} has {int(p.has[j])} cos {cs[j]:.3f} conf {p.conf[j]:.2f}")
    c = [r for r in F._candidate_rows(seq) if r[1].pid == pid]
    print("   candidates:", [(int(p.frame[r[2]]), r[3]) for r in c])
