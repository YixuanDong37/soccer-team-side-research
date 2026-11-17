"""ORACLE (labels; never a method result): how many returns could any linking of pieces get right?

- all:      link every two pieces whose known identities (>= 5 matched rows, majority) are equal;
- cand:     the same, restricted to m_appear's candidate pairs (features.pair_features);
- cand, both >= 5 clean rows; cand, gap <= 250; appearance_match.py's candidate pairs.
"""
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score  # noqa: E402

import fitlabels  # noqa: E402
from features import pair_features  # noqa: E402


def show(tag, method):
    p = score.evaluate(method)["pooled"]
    print(f"{tag:58s} right {p['right']:3d} wrong {p['wrong']:2d} lost {p['lost']:3d} | links {p['links']:4d} r/w/u {p['links_right']}/{p['links_wrong']}/{p['links_unknown']} impure {p['impure_row_share']:.4f}")


if __name__ == "__main__":
    allm, cand, c5, g250, am = {}, {}, {}, {}, {}
    for s in data.DEV_CLIPS:
        ident = fitlabels.piece_identity(s)
        P = data.pieces(s)
        by = {}
        for p in P.sort_values("first").piece:
            if int(p) in ident:
                by.setdefault(ident[int(p)][0], []).append(int(p))
        allm[s] = {"links": [[g[i], g[i + 1]] for g in by.values() for i in range(len(g) - 1)]}
        F = pair_features(s)
        same = [ident.get(int(a), (None,))[0] is not None and ident.get(int(a), (None,))[0] == ident.get(int(b), (-1,))[0] for a, b in zip(F.a, F.b)]
        G = F[same]
        cand[s] = {"links": G[["a", "b"]].to_numpy().tolist()}
        c5[s] = {"links": G[(G.clean_a >= 5) & (G.clean_b >= 5)][["a", "b"]].to_numpy().tolist()}
        g250[s] = {"links": G[G.gap <= 250][["a", "b"]].to_numpy().tolist()}
        am[s] = {"links": [[int(a), int(b)] for a, b, _ in data.appearance(s)["candidate_pairs"]
                           if int(a) in ident and int(b) in ident and ident[int(a)][0] == ident[int(b)][0]]}
    show("oracle: all same-identity pieces", allm)
    show("oracle: within m_appear candidates", cand)
    show("oracle: m_appear candidates, both >= 5 clean rows", c5)
    show("oracle: m_appear candidates, gap <= 250 frames", g250)
    show("oracle: appearance_match.py candidate pairs", am)
