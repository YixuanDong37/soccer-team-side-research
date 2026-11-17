"""Pairs of returns of one clip whose appearance says the two people were exchanged across the
gap: piece-before of person i looks like piece-after of person j AND piece-before of j looks like
piece-after of i, both clearly more than the annotated pairing. Either the appearance model
fails twice in a complementary way, or the annotation exchanged the two identities at re-entry.
No image was viewed: these are SUSPECTS for a human check, not corrections.
DIAGNOSIS ONLY (rule 5c). -> out/swap_suspects.csv

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I swap_suspects.py
"""
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ctx  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
MARGIN = 0.03
ret = pd.read_csv(OUT / "returns.csv")
ok = ret[ret.anchor_a.notna() & ret.anchor_b.notna()]
rows = []
for seq, g in ok.groupby("clip"):
    c = ctx(seq)
    P = c.P
    for i, j in combinations(range(len(g)), 2):
        ri, rj = g.iloc[i], g.iloc[j]
        if ri.ref_id == rj.ref_id or ri.ann_team != rj.ann_team:
            continue
        ai, bi, aj, bj = (P[int(x)] for x in (ri.anchor_a, ri.anchor_b, rj.anchor_a, rj.anchor_b))
        if len({ai["piece"], bi["piece"], aj["piece"], bj["piece"]}) < 4 or any(p["desc_any"] is None for p in (ai, bi, aj, bj)):
            continue
        if not (bj["first"] > ai["last"] and bi["first"] > aj["last"]):
            continue   # the exchanged pairing is impossible in time
        cii, cjj = float(ai["desc_any"] @ bi["desc_any"]), float(aj["desc_any"] @ bj["desc_any"])
        cij, cji = float(ai["desc_any"] @ bj["desc_any"]), float(aj["desc_any"] @ bi["desc_any"])
        if cij - cii >= MARGIN and cji - cjj >= MARGIN and cij - cjj >= MARGIN and cji - cii >= MARGIN:
            rows.append({"clip": seq, "ret_i": int(ri.ret), "person_i": int(ri.ref_id), "jersey_i": ri.ann_jersey, "ret_j": int(rj.ret), "person_j": int(rj.ref_id),
                         "jersey_j": rj.ann_jersey, "team": ri.ann_team, "before_i": ai["piece"], "after_i": bi["piece"], "before_j": aj["piece"], "after_j": bj["piece"],
                         "cos_annotated_i": round(cii, 3), "cos_annotated_j": round(cjj, 3), "cos_exchanged_i_to_j": round(cij, 3), "cos_exchanged_j_to_i": round(cji, 3),
                         "clean_rows": f"{ai['n_clean']}/{bi['n_clean']}/{aj['n_clean']}/{bj['n_clean']}",
                         "exit_i": int(ri.exit_frame), "reentry_i": int(ri.reentry_frame), "exit_j": int(rj.exit_frame), "reentry_j": int(rj.reentry_frame),
                         "sides_i": f"{ri.exit_side}>{ri.reentry_side}", "sides_j": f"{rj.exit_side}>{rj.reentry_side}",
                         "out_i": f"{ri.out_plugin}/{ri.out_mutual}", "out_j": f"{rj.out_plugin}/{rj.out_mutual}"})  # fmt: skip
df = pd.DataFrame(rows)
df.to_csv(OUT / "swap_suspects.csv", index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60)
print(df.to_string())
flag = set(zip(df["clip"], df.ret_i)) | set(zip(df["clip"], df.ret_j))
print("returns in a suspected exchange:", len(flag), sorted(flag))
# single-sided: the annotated pairing loses clearly to another person's piece on both sides of the return
neg = ok[ok.true_minus_best_impostor <= -MARGIN]
print("anchor pairs losing to another person's piece by >= 0.03:", len(neg))
print(neg[["clip", "ret", "ref_id", "status", "anchor_a", "anchor_b", "anchors_cos_any", "imp_best_cos_for_a", "imp_best_piece_for_a", "imp_best_cos_for_b", "imp_best_piece_for_b", "true_minus_best_impostor", "anchors_clean", "out_plugin", "out_mutual"]].to_string())
