"""DIAGNOSIS (labels via score.evaluate): how complementary are the agents' held-out METHOD files?
Unions of their links on the uncut pieces. Not a method result."""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score
W = "C:/Research/plugins/tools/reid_lab/work/"
def load(p):
    return json.load(open(W + p, encoding="utf-8"))["clips"]
M = {
 "appear_l3": load("m_appear/out/agglo-lam3.method.json"),
 "appear_l2": load("m_appear/out/agglo-lam2.method.json"),
 "pos_strict": load("m_space/METHOD_pos_strict.json"),
 "pos_e1_t40": load("m_space/METHOD_pos_e1_t40.json"),
 "keep_strict": load("m_keep/out/keep_long_loco_lam1.json"),
 "keep_bal": load("m_keep/out/keep_loco_lam1.json"),
}
def union(*names):
    out = {}
    for s in data.DEV_CLIPS:
        L = []
        for n in names:
            L += [list(map(int, l)) for l in (M[n].get(s) or {}).get("links", [])]
        out[s] = {"links": L}
    return out
def show(label, m):
    r = score.evaluate(m)
    p = r["pooled"]
    print(f"{label:45s} {p['right']:3d}/{p['wrong']:2d}/{p['lost']:3d}  links {p['links_right']}/{p['links_wrong']}/{p['links_unknown']}  impure {p['impure_row_share']:.4f}")
    return r
for n in M:
    show(n, union(n))
show("appear_l3 + pos_strict", union("appear_l3", "pos_strict"))
show("appear_l3 + keep_strict", union("appear_l3", "keep_strict"))
show("appear_l3 + keep_bal", union("appear_l3", "keep_bal"))
show("appear_l3 + pos_strict + keep_strict", union("appear_l3", "pos_strict", "keep_strict"))
show("appear_l3 + pos_strict + keep_bal", union("appear_l3", "pos_strict", "keep_bal"))
show("appear_l2 + pos_strict + keep_bal", union("appear_l2", "pos_strict", "keep_bal"))
show("appear_l2 + pos_e1_t40 + keep_bal", union("appear_l2", "pos_e1_t40", "keep_bal"))
