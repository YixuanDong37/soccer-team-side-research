"""probe (reads fuse's raw LOCO results, which were scored with labels): SNGS-162 alone and the other
12 clips at every fixed margin (held-out numbers of each clip's own fold)."""
import pickle, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_0")
import vr_common as vc
from vr_common import data
raw = pickle.load(open(vc.FUSE_CACHE / "loco_raw.pkl", "rb"))["folds"]
M = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20)
print("margin | SNGS-162 r/w/l, links r/w/u | other 12 clips r/w/l, links r/w/u, impure | clips with a wrong return | clips with a wrong link")
for m in M:
    a = vc.pool([raw["SNGS-162"]["res"][("fuse", m, "SNGS-162")]])
    b = vc.pool(raw[c]["res"][("fuse", m, c)] for c in data.DEV_CLIPS if c != "SNGS-162")
    wr = [c for c in data.DEV_CLIPS if vc.counts(raw[c]["res"][("fuse", m, c)]["per_return"])["wrong"]]
    wl = [f"{c}:{raw[c]['res'][('fuse', m, c)]['lk']['wrong']}" for c in data.DEV_CLIPS if raw[c]["res"][("fuse", m, c)]["lk"]["wrong"]]
    print(f"{m:.2f} | {a['right']}/{a['wrong']}/{a['lost']}, {a['links_right']}/{a['links_wrong']}/{a['links_unknown']} | {b['right']}/{b['wrong']}/{b['lost']}, {b['links_right']}/{b['links_wrong']}/{b['links_unknown']}, {100*b['impure_row_share']:.2f}% | {wr} | {wl}")
