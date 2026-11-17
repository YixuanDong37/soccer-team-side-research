"""DIAGNOSIS / ORACLE ONLY (reads labels through reid_lab.score): where are the true in-piece
identity switches, and what is the ceiling of a perfect cut detector?  Numbers from this script
are never method results.  Writes diag_switches.json (true switch list) for diagnosis scripts.

    python -I diag_oracle.py
"""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from collections import Counter
import numpy as np, pandas as pd
from reid_lab import data, score

MIN_SEG = 5   # a run of another person must have >= MIN_SEG matched rows to count as a switch


def runs_of(piece_frames, piece_refs, min_seg=MIN_SEG):
    """Greedy segmentation of a matched (frame, ref) sequence into runs of one majority person:
    short excursions (< min_seg rows) are absorbed."""
    # run-length encode
    rl = []
    for f, r in zip(piece_frames, piece_refs):
        if rl and rl[-1][0] == r:
            rl[-1][2] = f; rl[-1][3] += 1
        else:
            rl.append([r, f, f, 1])
    # drop short runs, merge neighbours with same ref
    changed = True
    while changed:
        changed = False
        keep = [x for x in rl if x[3] >= min_seg]
        if not keep:
            return []
        merged = []
        for x in keep:
            if merged and merged[-1][0] == x[0]:
                merged[-1][2] = x[2]; merged[-1][3] += x[3]
            else:
                merged.append(list(x))
        if len(merged) != len(rl):
            changed = len(merged) != len(keep)
        rl = merged
    return rl


def main():
    out = {}
    tot = Counter()
    oracle = {}
    for seq in data.DEV_CLIPS:
        c = score.clip(seq)
        t = data.tracks(seq)
        df = pd.DataFrame({"frame": c.frames, "ref": c.refs, "piece": c.piece}).sort_values(["piece", "frame"])
        df = df[df.piece >= 0]
        sw = []
        cuts = []
        for p, g in df.groupby("piece"):
            rl = runs_of(g.frame.to_numpy(), g.ref.to_numpy())
            for i in range(1, len(rl)):
                a, b = rl[i - 1], rl[i]
                # exact switch frame: first frame of run b; the cut goes at b's first frame
                sw.append({"piece": int(p), "from_ref": int(a[0]), "to_ref": int(b[0]), "last_a": int(a[2]), "first_b": int(b[1]),
                           "rows_a": int(a[3]), "rows_b": int(b[3])})
                cuts.append([int(p), int(b[1])])
        base = score.score_clip(seq)
        orc = score.score_clip(seq, [], cuts)
        oracle[seq] = {"links": [], "cuts": cuts}
        # impure rows in split, by piece
        maj = df.groupby("piece").ref.agg(lambda s: s.value_counts().index[0])
        imp = df[df.ref.to_numpy() != maj.loc[df.piece].to_numpy()]
        n_imp_pieces = imp.piece.nunique()
        print(f"{seq}: switches {len(sw)} in {len(set(s['piece'] for s in sw))} pieces; impure rows {base['impure_rows']} "
              f"({base['impure_rows']/base['tracked_rows']:.3%}) in {n_imp_pieces} pieces -> oracle cuts {orc['impure_rows']} "
              f"({orc['impure_rows']/orc['tracked_rows']:.3%}); returns {base['returns']} -> {orc['returns']}")
        for s in sw:
            print("    ", s)
        out[seq] = sw
        tot["switches"] += len(sw); tot["imp"] += base["impure_rows"]; tot["imp_oracle"] += orc["impure_rows"]; tot["rows"] += base["tracked_rows"]
    print(dict(tot), tot["imp"] / tot["rows"], tot["imp_oracle"] / tot["rows"])
    print("split      ", score.evaluate(score.baseline_links("split"))["pooled"])
    print("oracle cuts", score.evaluate(oracle)["pooled"])
    json.dump(out, open("C:/Research/plugins/tools/reid_lab/work/m_cuts/diag_switches.json", "w"), indent=1)


if __name__ == "__main__":
    main()
