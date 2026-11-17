"""DIAGNOSIS: prototype-level centring, appearance team, row selection and weights."""
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from reid_lab import data  # noqa: E402

from core import ctx  # noqa: E402
from diag2 import evaluate, unit  # noqa: E402

CL = {s: ctx(s) for s in data.DEV_CLIPS}
E0 = {s: np.where(CL[s].has[:, None], CL[s].E, 0.0) for s in CL}


def team_means(seq, sel):
    """Mean raw embedding of the selected rows of the pieces labelled player of each team."""
    c = CL[seq]
    P = c.P
    out = {}
    for tm in ("left", "right"):
        rows = [c.rows_of[int(p)] for p, r, t_ in zip(P.piece, P.role, P.team) if r == "player" and t_ == tm]
        rows = np.concatenate(rows or [np.zeros(0, int)])
        rows = rows[sel[rows] & c.has[rows]]
        out[tm] = E0[seq][rows].mean(axis=0) if len(rows) > 50 else None
    return out


def raw_means(seq, sel, min_rows=5, weight=None):
    c = CL[seq]
    out = {}
    for pid, rows in c.rows_of.items():
        r = rows[sel[rows] & c.has[rows]]
        if len(r) < min_rows:
            r = rows[c.has[rows]]
            if not len(r):
                continue
        if weight is None:
            out[pid] = E0[seq][r].mean(axis=0)
        else:
            w = weight[r]
            out[pid] = (E0[seq][r] * w[:, None]).sum(axis=0) / w.sum()
    return out


def centred(seq, means, how="label", sel=None):
    c = CL[seq]
    P = c.P
    tm = team_means(seq, c.clean if sel is None else sel)
    out = {}
    glob = np.mean([v for v in tm.values() if v is not None], axis=0)

    def nearest(m):
        cands = [(float(unit(m) @ unit(v)), k) for k, v in tm.items() if v is not None]
        return tm[max(cands)[1]] if cands else glob

    for pid, m in means.items():
        lab = P.team[c.index[pid]]
        if how == "label":
            mu = tm.get(lab) if lab in tm and tm[lab] is not None else glob
        elif how == "nearest":
            mu = nearest(m)
        elif how == "label_then_nearest":
            mu = tm[lab] if lab in tm and tm[lab] is not None else nearest(m)
        else:
            mu = glob
        out[pid] = unit(m - mu)
    return out


def run(tag, **kw):
    how = kw.pop("how", "label")
    return evaluate({s: centred(s, raw_means(s, **{k: (v(CL[s]) if callable(v) else v) for k, v in kw.items()}), how) for s in CL}, tag)


if __name__ == "__main__":
    run("proto-centred label; clean>=5", sel=lambda c: c.clean)
    run("proto-centred nearest; clean>=5", sel=lambda c: c.clean, how="nearest")
    run("proto-centred label_then_nearest; clean>=5", sel=lambda c: c.clean, how="label_then_nearest")
    run("proto-centred global; clean>=5", sel=lambda c: c.clean, how="global")
    agree = tot = 0
    for s in CL:
        c = CL[s]
        tm = team_means(s, c.clean)
        m = raw_means(s, c.clean)
        for pid, v in m.items():
            lab = c.P.team[c.index[pid]]
            role = c.P.role[c.index[pid]]
            if lab in ("left", "right") and role == "player" and all(x is not None for x in tm.values()):
                near = max((float(unit(v) @ unit(x)), k) for k, x in tm.items())[1]
                agree += near == lab
                tot += 1
    print("appearance team == label team on labelled player pieces:", agree, "/", tot)
    print("--- row selection (centring by label, fallback to all rows below the minimum)")
    run("clean>=1", sel=lambda c: c.clean, min_rows=1)
    run("clean>=10", sel=lambda c: c.clean, min_rows=10)
    run("all rows", sel=lambda c: np.ones(len(c.h), bool), min_rows=1)
    run("no-overlap only (edge ok, any height)", sel=lambda c: c.cover <= 0, min_rows=5)
    run("cover<=0.1 & edge>1 & h>=40", sel=lambda c: (c.cover <= 0.1) & (c.edge_dist > 1) & (c.h >= 40), min_rows=5)
    run("cover<=0.3 & edge>1 & h>=40", sel=lambda c: (c.cover <= 0.3) & (c.edge_dist > 1) & (c.h >= 40), min_rows=5)
    run("clean & edge_dist>10", sel=lambda c: c.clean & (c.edge_dist > 10), min_rows=5)
    run("clean & edge_dist>30", sel=lambda c: c.clean & (c.edge_dist > 30), min_rows=5)
    run("clean & h>=60", sel=lambda c: c.clean & (c.h >= 60), min_rows=5)
    run("no overlap, edge>1, h>=30", sel=lambda c: (c.cover <= 0) & (c.edge_dist > 1) & (c.h >= 30), min_rows=5)
    run("clean, weight=h", sel=lambda c: c.clean, weight=lambda c: c.h)
    run("clean, weight=h^2", sel=lambda c: c.clean, weight=lambda c: c.h ** 2)
    run("clean, weight=conf", sel=lambda c: c.clean, weight=lambda c: c.t.conf.to_numpy(float))
