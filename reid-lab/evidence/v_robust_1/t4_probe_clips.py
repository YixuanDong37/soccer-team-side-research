"""v_robust_1 / t4_probe_clips.py -- LABEL-FREE numbers per clip (does the clip that holds most wrong returns stand out
without labels?) next to its held-out result (from out/r1_reproduce.json).   -> out/t4_probe_clips.txt"""
import json
import sys
from collections import Counter
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import numpy as np
import vr_common as vc
from vr_common import data, prep

R1 = json.loads((vc.OUT / "r1_reproduce.json").read_text(encoding="utf-8"))
rows = []
for seq in data.DEV_CLIPS:
    t = data.tracks(seq)
    cuts = prep.cuts_for(seq, seq)
    ids = vc.piece_ids_after_cuts(seq, cuts)
    n = Counter(ids[ids >= 0].tolist())
    clean = data.clean_rows(seq)
    tr = t[t.track_id >= 0]
    pl = tr[tr.role != "referee"]
    per_frame = pl.groupby("frame").size()
    M = vc.load_method("strict")[seq]
    g = vc.groups_of(M["links"])
    c = Counter(R1["methods"]["strict"]["per_clip"][seq]["per_return"])
    rows.append((seq, len(set(t.track_id[t.track_id >= 0])), len(n), sum(v >= 5 for v in n.values()), len(cuts), float(clean[ids >= 0].mean()),
                 float(per_frame.mean()), int(per_frame.max()), len(M["links"]), len(g), f"{c['right']}/{c['wrong']}/{c['lost']}"))
L = ["clip | gbf pieces | sub-pieces after cuts | solid (>= 5 boxes) | cuts | clean share of boxes | players in view mean | max | strict links | strict groups | strict right/wrong/lost"]
for r in rows:
    L.append(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.2f} | {r[6]:.1f} | {r[7]} | {r[8]} | {r[9]} | {r[10]}")
for j, name in ((1, "gbf pieces"), (3, "solid pieces"), (4, "cuts"), (5, "clean share"), (6, "players in view")):
    v = np.array([r[j] for r in rows], float)
    i = data.DEV_CLIPS.index("SNGS-162")
    L.append(f"SNGS-162 rank in '{name}': {int((v > v[i]).sum()) + 1} of 13 from the top (value {v[i]:.2f}; others {np.delete(v, i).min():.2f} - {np.delete(v, i).max():.2f})")
(vc.OUT / "t4_probe_clips.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
