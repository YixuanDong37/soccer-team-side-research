"""DIAGNOSIS (labels): list the wrong returns and the wrong links of a method.

    from diag_wrong import explain; explain({seq: {"links": [...]}})
"""
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from reid_lab import data, score  # noqa: E402

import fitlabels  # noqa: E402
from features import pair_features  # noqa: E402


def explain(method, show_right=False, base=None):
    """base: per-return outcomes of another method to compare with (only changed returns are printed)."""
    for seq in data.DEV_CLIPS:
        m = method.get(seq) or {}
        links = m.get("links") or []
        res = score.score_clip(seq, links, m.get("cuts") or [])
        c = score.clip(seq)
        ident = fitlabels.piece_identity(seq)
        F = pair_features(seq).set_index(["a", "b"])
        scored = [r for r in c.returns]
        for r, o in zip(scored, res["per_return"]):
            if o == "wrong" or (show_right and o == "right"):
                pa = int(c.piece[r["a"]])
                pb = int(c.piece[r["b"]])
                grp = _group(links, pb)
                print(f"  {seq} return ref {r['ref_id']} gap {r['gap']} A piece {pa} {ident.get(pa)} B piece {pb} {ident.get(pb)} -> {o}; "
                      f"B's group: {[(p, ident.get(p, (None,))[0]) for p in sorted(grp)]}")
        for a, b in links:
            a, b = int(a), int(b)
            if a in ident and b in ident and ident[a][0] != ident[b][0]:
                row = F.loc[(a, b)] if (a, b) in F.index else None
                extra = "" if row is None else (f" cos_tc {row.cos_tc:.3f} cos_raw {row.cos_raw:.3f} rival f/b {row.rival_fwd_tc:.2f}/{row.rival_bwd_tc:.2f} "
                                                 f"marg f/b {row.marg_fwd_tc:.2f}/{row.marg_bwd_tc:.2f} gap {row.gap} clean {row.clean_a}/{row.clean_b} same_plugin {row.same_plugin}")
                print(f"  {seq} WRONG LINK {a} {ident[a]} -> {b} {ident[b]}{extra}")


def _group(links, p):
    adj = {}
    for a, b in links:
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))
    seen, todo = {p}, [p]
    while todo:
        x = todo.pop()
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                todo.append(y)
    return seen


if __name__ == "__main__":
    import decide
    kind, mk, t_abs, margin = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    m = {s: {"links": decide.mutual(pair_features(s), kind, t_abs, margin, margin_kind=mk)} for s in data.DEV_CLIPS}
    print(score.evaluate(m)["pooled"])
    explain(m)
