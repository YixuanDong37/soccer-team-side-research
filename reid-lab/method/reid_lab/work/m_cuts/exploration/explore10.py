"""Debug: candidates of a piece with the rows around them."""
import sys
sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import numpy as np
import pandas as pd
import features as F
import truth as T

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 800)
seq, pid = sys.argv[1], int(sys.argv[2])
lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 10**6)
p = F.piece_table(seq)[pid]
mf, mr = T.matched(seq).get(pid, (np.zeros(0, int), np.zeros(0, int)))
ref = dict(zip(mf.tolist(), mr.tolist()))
print("rows (frame:ref clean):", " ".join(f"{int(f)}:{ref.get(int(f), '-')}{'c' if c else ('h' if h else 'x')}" for f, c, h in zip(p.frame, p.clean, p.has) if lo <= f <= hi))
c = F.candidates(seq)
c = c[(c.piece == pid) & (c.f0 >= lo) & (c.f1 <= hi)]
print(c[["i", "f0", "f1", "gap", "kind", "plateau", "cos", "nb", "na", "clean_b", "clean_a", "alt_d", "alt_r", "d_m", "dt", "rows_before", "rows_after"]].round(3).to_string())
print("true switches:", [s for s in T.switches(seq) if s["piece"] == pid])
