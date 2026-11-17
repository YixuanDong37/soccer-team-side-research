"""(5a) Appearance, row level: how well ONE PRTReid embedding tells a person apart, and what
degrades it (diagnosis; the annotation gives every matched row its person and team).

Pairs of rows of one clip: same person (>= 25 frames apart, so not near-duplicates), different
persons of one annotated team, different teams. Cosine of the unit embeddings. AUC is the
probability that a same-person pair scores above a different-person same-team pair.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import common as C  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
data = C.data
rng = np.random.default_rng(0)
res: dict = {}
N_SAME, N_DIFF = 60000, 60000  # sampled pairs per clip and kind (fewer when the clip has fewer)


def row_table(seq):
    t = data.tracks(seq)
    m = C.matched(seq)
    emb = data.embeddings(seq)
    x1, y1, x2, y2 = (t[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
    area = (x2 - x1) * (y2 - y1)
    ov = np.zeros(len(t))
    for _, idx in t.groupby("frame").indices.items():
        if len(idx) < 2:
            continue
        ix = np.clip(np.minimum(x2[idx][:, None], x2[idx][None]) - np.maximum(x1[idx][:, None], x1[idx][None]), 0, None)
        iy = np.clip(np.minimum(y2[idx][:, None], y2[idx][None]) - np.maximum(y1[idx][:, None], y1[idx][None]), 0, None)
        inter = ix * iy
        np.fill_diagonal(inter, 0)
        ov[idx] = inter.max(axis=1) / np.maximum(area[idx], 1e-9)
    d = pd.DataFrame({"trow": m.trow.to_numpy(), "frame": m.frame.to_numpy(), "ref": m.ref_id.to_numpy(), "team": m.team_ref.to_numpy(), "role": m.role_ref.to_numpy()})
    i = d.trow.to_numpy()
    d["h"] = (y2 - y1)[i]
    d["cut"] = ((x1 <= 1) | (y1 <= 1) | (x2 >= C.W - 2) | (y2 >= C.H - 2))[i]          # box clipped by the image border (clean_rows' rule)
    d["near"] = ((x1 <= 10) | (y1 <= 10) | (x2 >= C.W - 11) | (y2 >= C.H - 11))[i]    # within 10 px of the border
    d["ov"] = ov[i]
    d["has"] = ~np.isnan(emb[i, 0])
    d["conf"] = t.conf.to_numpy()[i]
    d = d[d.has & d.role.isin(["player", "goalkeeper"])].reset_index(drop=True)
    return d, emb


def sample_pairs(d, kind, n):
    """Index pairs (i, j) into d for the pair kind."""
    ref, team, fr = d.ref.to_numpy(), d.team.to_numpy(object), d.frame.to_numpy()
    out_i, out_j = [], []
    tries = 0
    while sum(len(x) for x in out_i) < n and tries < 40:
        tries += 1
        i = rng.integers(0, len(d), 4 * n)
        if kind == "same":
            # partner: a random row of the same person
            order = np.argsort(ref, kind="stable")
            sref = ref[order]
            lo = np.searchsorted(sref, ref[i], "left")
            hi = np.searchsorted(sref, ref[i], "right")
            j = order[lo + (rng.random(len(i)) * (hi - lo)).astype(int)]
            ok = np.abs(fr[i] - fr[j]) >= 25
        else:
            j = rng.integers(0, len(d), len(i))
            ok = ref[i] != ref[j]
            ok &= (team[i] == team[j]) if kind == "diff_same_team" else (team[i] != team[j])
        out_i.append(i[ok])
        out_j.append(j[ok])
    i, j = np.concatenate(out_i)[:n], np.concatenate(out_j)[:n]
    return i, j


allp = []
for seq in C.CLIPS:
    d, emb = row_table(seq)
    E = emb[d.trow.to_numpy()]
    for kind, n in (("same", N_SAME), ("diff_same_team", N_DIFF), ("diff_team", N_DIFF // 3)):
        i, j = sample_pairs(d, kind, n)
        cos = np.einsum("ij,ij->i", E[i], E[j])
        p = pd.DataFrame({"seq": seq, "kind": kind, "cos": cos, "dt": np.abs(d.frame.to_numpy()[i] - d.frame.to_numpy()[j]) / C.FPS,
                          "h_min": np.minimum(d.h.to_numpy()[i], d.h.to_numpy()[j]), "h_max": np.maximum(d.h.to_numpy()[i], d.h.to_numpy()[j]),
                          "cut_any": d.cut.to_numpy()[i] | d.cut.to_numpy()[j], "cut_both": d.cut.to_numpy()[i] & d.cut.to_numpy()[j],
                          "near_any": d.near.to_numpy()[i] | d.near.to_numpy()[j],
                          "ov_max": np.maximum(d.ov.to_numpy()[i], d.ov.to_numpy()[j]), "conf_min": np.minimum(d.conf.to_numpy()[i], d.conf.to_numpy()[j])})  # fmt: skip
        allp.append(p)
PR = pd.concat(allp, ignore_index=True)
print("sampled row pairs:", PR.groupby("kind").size().to_dict())
same, dst, dt_ = PR[PR.kind == "same"], PR[PR.kind == "diff_same_team"], PR[PR.kind == "diff_team"]


def line(name, s, d1, d2=None):
    pr = C.prec_at_recall(s.cos, d1.cos)
    out = {"rows compared": name, "same pairs": len(s), "diff pairs": len(d1), "cos same p10": s.cos.quantile(0.1), "cos same p50": s.cos.median(),
           "cos diff p50": d1.cos.median(), "cos diff p99": d1.cos.quantile(0.99), "AUC": C.auc(s.cos, d1.cos),
           "TPR@FPR1%": float((s.cos > d1.cos.quantile(0.99)).mean()), "prec@rec0.5": pr[0.5][0], "prec@rec0.8": pr[0.8][0], "prec@rec0.9": pr[0.9][0]}  # fmt: skip
    return out


clean = lambda g: g[~g.cut_any & (g.ov_max == 0) & (g.h_min >= 40)]  # noqa: E731
tabs = [line("all rows; same person vs other person of the same team", same, dst),
        line("all rows; same person vs person of the other team", same, dt_),
        line("both rows clean (not clipped, no overlap, >= 40 px)", clean(same), clean(dst))]
print("\n=== overall (equal numbers of pairs per kind, so precision is at a 1:1 prior) ===")
T0 = pd.DataFrame(tabs)
print(T0.round(3).to_string())
T0.to_csv(C.OUT / "t5a_rows_overall.csv", index=False)
res["overall"] = T0.round(4).to_dict("records")

print("\n=== by the smaller box height of the two rows (720p px; rows not clipped, no overlap) ===")
ok = lambda g: g[~g.cut_any & (g.ov_max == 0)]  # noqa: E731
tabs = []
for lo, hi in ((0, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 110), (110, 2000)):
    f = lambda g: g[(g.h_min >= lo) & (g.h_min < hi)]  # noqa: E731
    tabs.append(line(f"min height {lo}-{hi}", f(ok(same)), f(ok(dst))))
T1 = pd.DataFrame(tabs)
print(T1.round(3).to_string())
T1.to_csv(C.OUT / "t5a_rows_by_height.csv", index=False)
res["by_height"] = T1.round(4).to_dict("records")

print("\n=== by clipping at the image border (no overlap, >= 40 px) ===")
base = lambda g: g[(g.ov_max == 0) & (g.h_min >= 40)]  # noqa: E731
tabs = [line("neither row clipped, neither within 10 px", base(same)[~base(same).near_any], base(dst)[~base(dst).near_any]),
        line("a row within 10 px of the border, none clipped", base(same)[base(same).near_any & ~base(same).cut_any], base(dst)[base(dst).near_any & ~base(dst).cut_any]),
        line("one row clipped by the border", base(same)[base(same).cut_any & ~base(same).cut_both], base(dst)[base(dst).cut_any & ~base(dst).cut_both]),
        line("both rows clipped", base(same)[base(same).cut_both], base(dst)[base(dst).cut_both])]  # fmt: skip
T2 = pd.DataFrame(tabs)
print(T2.round(3).to_string())
T2.to_csv(C.OUT / "t5a_rows_by_edge.csv", index=False)
res["by_edge"] = T2.round(4).to_dict("records")

print("\n=== by overlap with another box (largest share of a row's box covered; not clipped, >= 40 px) ===")
base = lambda g: g[~g.cut_any & (g.h_min >= 40)]  # noqa: E731
tabs = []
for lo, hi, nm in ((-1, 0, "0 (no overlap)"), (0, 0.1, "0-0.1"), (0.1, 0.3, "0.1-0.3"), (0.3, 0.6, "0.3-0.6"), (0.6, 1.01, "0.6-1")):
    f = lambda g: g[(g.ov_max > lo) & (g.ov_max <= hi)]  # noqa: E731
    tabs.append(line(f"overlap {nm}", f(base(same)), f(base(dst))))
T3 = pd.DataFrame(tabs)
print(T3.round(3).to_string())
T3.to_csv(C.OUT / "t5a_rows_by_overlap.csv", index=False)
res["by_overlap"] = T3.round(4).to_dict("records")

print("\n=== by time apart (same-person pairs; the different-person pairs are the clean pool) ===")
cs, cd = clean(same), clean(dst)
tabs = []
for lo, hi in ((1, 2), (2, 5), (5, 10), (10, 20), (20, 31)):
    tabs.append(line(f"{lo}-{hi} s apart", cs[(cs["dt"] >= lo) & (cs["dt"] < hi)], cd))
T4 = pd.DataFrame(tabs)
print(T4.round(3).to_string())
T4.to_csv(C.OUT / "t5a_rows_by_time.csv", index=False)
res["by_time"] = T4.round(4).to_dict("records")

print("\n=== by detection confidence (smaller of the two; clean rows) ===")
tabs = []
for lo, hi in ((0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)):
    f = lambda g: g[(g.conf_min >= lo) & (g.conf_min < hi)]  # noqa: E731
    tabs.append(line(f"min conf {lo}-{hi}", f(cs), f(cd)))
T5 = pd.DataFrame(tabs)
print(T5.round(3).to_string())
T5.to_csv(C.OUT / "t5a_rows_by_conf.csv", index=False)

print("\n=== per clip (clean rows): where a global cosine bar would sit ===")
tabs = []
for seq in C.CLIPS:
    s, d1 = cs[cs.seq == seq], cd[cd.seq == seq]
    tabs.append({"seq": seq, "cos same p05": s.cos.quantile(0.05), "cos same p50": s.cos.median(), "cos diff p50": d1.cos.median(), "cos diff p95": d1.cos.quantile(0.95),
                 "cos diff p99": d1.cos.quantile(0.99), "AUC": C.auc(s.cos, d1.cos), "TPR@FPR1%": float((s.cos > d1.cos.quantile(0.99)).mean())})  # fmt: skip
T6 = pd.DataFrame(tabs)
print(T6.round(3).to_string())
T6.to_csv(C.OUT / "t5a_rows_per_clip.csv", index=False)
res["per_clip"] = T6.round(4).to_dict("records")
(C.OUT / "t5a_appearance_rows.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
