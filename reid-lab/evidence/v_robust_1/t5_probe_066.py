"""v_robust_1 / t5_probe_066.py -- (LABELS: diagnosis) SNGS-066, the wrong return of person 21: what the pieces around it are.
-> out/t5_probe_066.txt"""
import sys
from collections import Counter
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import numpy as np
import vr_common as vc
import vr_labels as vl
from vr_common import data, prep
from reid_lab import score  # LABELS

seq = "SNGS-066"
out = []
c = score.clip(seq)
for name, cuts in (("gbf pieces (no cuts)", ()), ("after the held-out cuts", prep.cuts_for(seq, seq))):
    ids = score._ids_after_cuts(c, [list(x) for x in cuts])
    out.append(name)
    for p in (28, 10000028, 14, 6, 49, 63):
        m = ids == p
        if not m.any():
            continue
        fr = c.frames[m]
        cnt = Counter(c.refs[m].tolist())
        # persons by time thirds
        out.append(f"  piece {p}: matched boxes {int(m.sum())}, frames {int(fr.min())}-{int(fr.max())}, persons {dict(cnt.most_common(4))}")
        if p in (28,):
            for lo, hi in ((0, 297), (298, 10**6)):
                mm = m & (c.frames >= lo) & (c.frames <= hi)
                out.append(f"    frames {lo}-{hi}: {dict(Counter(c.refs[mm].tolist()).most_common(4))}")
out.append("cuts of piece 28 (held-out weights): " + str([x for x in prep.cuts_for(seq, seq) if x[0] == 28]))
t = data.tracks(seq)
ids = vc.piece_ids_after_cuts(seq, prep.cuts_for(seq, seq))
clean = data.clean_rows(seq)
for p in (28, 10000028, 14):
    m = ids == p
    out.append(f"label-free: piece {p}: boxes {int(m.sum())}, clean {int(clean[m].sum())}, frames {int(t.frame[m].min())}-{int(t.frame[m].max())}, plugin id {sorted(set(t.source_track_id[m]))}")
# the return itself
for r in vl.returns_table(seq, [], []):
    if r["ref_id"] == 21:
        out.append("split: " + str(r))
(vc.OUT / "t5_probe_066.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
