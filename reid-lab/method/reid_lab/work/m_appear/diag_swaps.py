"""DIAGNOSIS (labels; no method result): returns whose annotated identities look exchanged.

For every clip and every two annotated people X, Y with a scored return each (pieces A_X -> B_X and
A_Y -> B_Y, all four with a known identity, >= 5 matched rows), compare the appearance of the four pieces.
A pair is listed when both crossed similarities (A_X ~ B_Y and A_Y ~ B_X) beat both annotated ones
(A_X ~ B_X, A_Y ~ B_Y) by at least MIN_GAIN in the team-centred cosine: the embeddings say the two people
came back with each other's annotation id. Also printed: the raw cosines, the rows behind every piece
and where (pitch y, metres) each piece ends / starts, as a second, independent cue.

No image is read. This cannot prove an annotation error; it lists what a human should look at.
"""
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from reid_lab import data, score  # noqa: E402

import fitlabels  # noqa: E402
from core import ctx  # noqa: E402
from features import protos  # noqa: E402

MIN_GAIN = 0.2


def _d(P, a, b):
    e, s = P.loc[a, "end_xy"], P.loc[b, "start_xy"]
    return "?" if e is None or s is None else f"{abs(s[1] - e[1]):.1f}"


if __name__ == "__main__":
    found = 0
    for seq in data.DEV_CLIPS:
        c = score.clip(seq)
        cx = ctx(seq)
        pr = protos(seq)
        P = cx.P.set_index("piece")
        ident = fitlabels.piece_identity(seq)
        rets = []
        for r in c.returns:
            if r["a"] is None or r["b"] is None:
                continue
            pa, pb = int(c.piece[r["a"]]), int(c.piece[r["b"]])
            if pa == pb or ident.get(pa, (None,))[0] != r["ref_id"] or ident.get(pb, (None,))[0] != r["ref_id"]:
                continue
            if pa in pr.tc and pb in pr.tc:
                rets.append((r["ref_id"], pa, pb, r))
        for i in range(len(rets)):
            for j in range(i + 1, len(rets)):
                (x, ax, bx, rx), (y, ay, by, ry) = rets[i], rets[j]
                if x == y or pr.ateam[ax] != pr.ateam[ay]:
                    continue
                if not (P.loc[ax, "last"] < P.loc[by, "first"] and P.loc[ay, "last"] < P.loc[bx, "first"]):
                    continue
                s = lambda p, q, k: float(pr.get(k)[p] @ pr.get(k)[q])  # noqa: E731
                direct = (s(ax, bx, "tc"), s(ay, by, "tc"))
                cross = (s(ax, by, "tc"), s(ay, bx, "tc"))
                if min(cross) - max(direct) < MIN_GAIN:
                    continue
                found += 1
                ypos = lambda p, key: "?" if P.loc[p, key] is None else f"{P.loc[p, key][1]:.1f}"  # noqa: E731
                print(f"{seq}: annotated people {x} and {y} (team {pr.ateam[ax]})")
                print(f"   annotated:  {ax} -> {bx} (person {x})  cos_tc {direct[0]:.3f} raw {s(ax, bx, 'raw'):.3f} | "
                      f"{ay} -> {by} (person {y})  cos_tc {direct[1]:.3f} raw {s(ay, by, 'raw'):.3f}")
                print(f"   crossed:    {ax} -> {by}  cos_tc {cross[0]:.3f} raw {s(ax, by, 'raw'):.3f} | {ay} -> {bx}  cos_tc {cross[1]:.3f} raw {s(ay, bx, 'raw'):.3f}")
                print(f"   clean rows: {ax}: {P.loc[ax, 'n_clean']}, {bx}: {P.loc[bx, 'n_clean']}, {ay}: {P.loc[ay, 'n_clean']}, {by}: {P.loc[by, 'n_clean']};"
                      f" gaps {rx['gap']} and {ry['gap']} frames")
                print(f"   pitch y at exit: {ax} {ypos(ax, 'end_xy')}, {ay} {ypos(ay, 'end_xy')}; at entry: {bx} {ypos(bx, 'start_xy')}, {by} {ypos(by, 'start_xy')}"
                      f"   (annotated |dy| {_d(P, ax, bx)} + {_d(P, ay, by)}, crossed |dy| {_d(P, ax, by)} + {_d(P, ay, bx)})")
    print("pairs listed:", found)

    # ---- second view, not tied to scored returns: later pieces that look like another annotated person
    # (best earlier candidate is person X although the piece is annotated Y, and it beats every earlier piece of
    # Y by MIN_GAIN in cos_tc), kept when the reverse also holds for the same two people.
    print("\nlater pieces that look like the other person, in both directions (all pieces, not only scored returns):")
    from features import pair_features
    for seq in data.DEV_CLIPS:
        ident = fitlabels.piece_identity(seq)
        F = pair_features(seq)
        P = ctx(seq).P.set_index("piece")
        looks = {}
        for b, g in F.groupby("b"):
            b = int(b)
            if b not in ident:
                continue
            g = g[[int(a) in ident for a in g.a]]
            if not len(g):
                continue
            top = g.loc[g.cos_tc.idxmax()]
            x, y = ident[int(top.a)][0], ident[b][0]
            own = g[[ident[int(a)][0] == y for a in g.a]].cos_tc.max()
            own = -1.0 if np.isnan(own) else own
            if x != y and top.cos_tc - own >= MIN_GAIN:
                looks.setdefault((x, y), []).append((int(top.a), b, float(top.cos_tc), float(top.cos_raw), own))
        for (x, y), v in sorted(looks.items()):
            if x < y and (y, x) in looks:
                for tag, items in ((f"annotated {y}, looks like {x}", v), (f"annotated {x}, looks like {y}", looks[(y, x)])):
                    for a, b, s_tc, s_raw, own in items:
                        print(f"  {seq} people {x}/{y}: piece {b} ({tag}): best earlier piece {a} cos_tc {s_tc:.3f} raw {s_raw:.3f}; best earlier piece of its "
                              f"annotated person cos_tc {own:.3f}; clean rows {P.loc[a, 'n_clean']}/{P.loc[b, 'n_clean']}; pitch y {a} exit "
                              f"{'?' if P.loc[a, 'end_xy'] is None else round(P.loc[a, 'end_xy'][1], 1)} -> {b} entry {'?' if P.loc[b, 'start_xy'] is None else round(P.loc[b, 'start_xy'][1], 1)}")
