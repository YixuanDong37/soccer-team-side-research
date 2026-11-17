import sys, time
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
import prep
from reid_lab import data
t0 = time.time()
seq = "SNGS-162"
c = prep.cuts_for(seq, seq)
print(len(c), "cuts", round(time.time() - t0, 1))
a = prep.app_table(seq, c); print("app", a.shape, round(time.time() - t0, 1))
s, e = prep.space_tables(seq, c); print("space", s.shape, e.shape, round(time.time() - t0, 1))
print(a.columns.tolist()); print(s.columns.tolist()); print(e.columns.tolist())
# the substitution is undone
t = data.tracks(seq); print("tracks ids > SUB after restore:", int((t.track_id >= prep.SUB).sum()))
print("sub-pieces in space table:", int((e.piece >= prep.SUB).sum()), "in app:", int((a.a >= prep.SUB).sum()))
