"""v_robust_1 / t6_probe_fixed.py -- (reads fuse_r1's scored LOCO results) the fixed-setting rows: every clip at one setting with
the numbers of its own fold, pooled; with and without SNGS-162.   -> out/t6_probe_fixed.txt"""
import pickle
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import vr_common as vc
from vr_common import data
sys.path.insert(0, str(vc.FUSE))
import rules
RAW = pickle.load(open(vc.FUSE_CACHE / "loco_raw.pkl", "rb"))["folds"]
out = ["leaves | margin | held-out right / wrong / lost | links r / w / u | impure | without SNGS-162 | SNGS-162"]
for lv in (False, True):
    for m in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20):
        cells = {c: rules.cell(RAW[c]["res"], "r1", (lv, m), c) for c in data.DEV_CLIPS}
        p = rules.pool(cells.values())
        q = rules.pool(v for c, v in cells.items() if c != "SNGS-162")
        x = cells["SNGS-162"]
        out.append(f"{'on' if lv else 'off'} | {m:.2f} | {p['right']} / {p['wrong']} / {p['lost']} | {p['links_right']} / {p['links_wrong']} / {p['links_unknown']} | "
                   f"{100 * p['impure_row_share']:.2f}% | {q['right']} / {q['wrong']} / {q['lost']} | {x['right']} / {x['wrong']} / {x['lost']}")
(vc.OUT / "t6_probe_fixed.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
