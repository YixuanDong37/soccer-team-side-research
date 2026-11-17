"""critic / o5b_closed_set.py -- ORACLE (labels): ceiling of an IDENTITY-LEVEL, closed-set decision.

For each annotated return of an outfield player whose boxes after the return are matched:
  probe      mean PRTReid embedding of the person's matched tracked rows in the 3 s after the return
  gallery    one prototype per same-team outfield player seen before the return: mean embedding of
             ALL his matched rows before the return (perfect history = oracle grouping)
  candidates (a) every such team-mate, (b) only those absent from the picture in the frame before the
             return (perfect bookkeeping), (c) = (b) and last seen in the same image half
Reported: how often the true person is the most similar candidate, and by what lead; raw cosine and
team-centred cosine (the team mean embedding subtracted). Row filter: rows with an embedding; no
clean-row filter (so that short, edge-clipped returns stay in).
This is NOT a method: the gallery uses the annotation. It says what the embeddings could do if the
other identities were kept correctly, against the piece-to-piece decisions the methods make.
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/critic")
import common as C
from common import data, score
from gbf.evaluate import gsr

P = C.Tee(C.OUT / "o5b_closed_set.txt")
W = 75
res = []
for seq in data.DEV_CLIPS:
    d = json.loads((C.TRAIN / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    img_frame = {im["image_id"]: int(im["file_name"].split(".")[0]) - 1 for im in d["images"]}
    ann = defaultdict(dict)
    team = {}
    for a in d["annotations"]:
        at = a.get("attributes") or {}
        if at.get("role") != "player":
            continue
        rid = int(a["track_id"])
        team[rid] = at.get("team")
        ann[rid][img_frame[a["image_id"]]] = float(a["bbox_image"]["x_center"])
    # matched rows with their tracks row (same matching as the scorer)
    ref = gsr.load_reference(C.TRAIN / seq / "Labels-GameState.json")
    t = data.tracks(seq)
    pred = t[["frame", "track_id", "x1", "y1", "x2", "y2"]].copy()
    for col in ("x1", "y1", "x2", "y2"):
        pred[col] = pred[col] * score.SCALE
    pairs, _ = gsr.match(ref, pred)
    prow = pairs["pred_row"].to_numpy(np.int64)
    pfr = pairs["frame"].to_numpy(np.int64)
    pref = pairs["ref_id"].to_numpy(np.int64)
    emb = data.embeddings(seq)
    has = ~np.isnan(emb[prow, 0]) & (t.track_id.to_numpy()[prow] >= 0)
    tmean = {}
    for tm in set(team.values()):
        m = has & np.isin(pref, [q for q in team if team[q] == tm])
        tmean[tm] = emb[prow[m]].mean(axis=0) if m.any() else np.zeros(emb.shape[1])

    def proto(mask, centre=None):
        v = emb[prow[mask]].mean(axis=0)
        if centre is not None:
            v = v - centre
        return v / np.linalg.norm(v)

    c = score.clip(seq)
    for r in c.returns:
        p, f = r["ref_id"], r["first"]
        if p not in team:
            continue
        pm = has & (pref == p) & (pfr >= f) & (pfr < f + W)
        if pm.sum() < 5:
            continue
        mates = [q for q in team if team[q] == team[p]]
        gal = {}
        for q in mates:
            gm = has & (pref == q) & (pfr < f)
            if gm.sum() >= 5:
                gal[q] = gm
        if p not in gal:
            continue
        absent = [q for q in gal if (f - 1) not in ann[q]]
        side = ann[p][f] < 960
        same_side = [q for q in absent if (ann[q][max(x for x in ann[q] if x < f)] < 960) == side]
        out = {"seq": seq, "p": p, "gap": r["gap"], "n_probe": int(pm.sum())}
        for kind, centre in (("raw", None), ("tc", tmean[team[p]])):
            pv = proto(pm, centre)
            sims = {q: float(pv @ proto(gal[q], centre)) for q in gal}
            for name, cand in (("all", list(gal)), ("absent", absent), ("side", same_side if p in same_side else absent)):
                if p not in cand:
                    cand = cand + [p]
                others = [sims[q] for q in cand if q != p]
                out[f"{kind}_{name}_n"] = len(cand)
                out[f"{kind}_{name}_lead"] = sims[p] - max(others) if others else 9.0
        res.append(out)

n = len(res)
P(f"returns with a probe (>= 5 embedded rows in the 3 s after the return) and a gallery entry of the person: {n} of 167")
for kind in ("raw", "tc"):
    P(f"\n-- {'raw cosine' if kind == 'raw' else 'team-centred cosine'}")
    for name, txt in (("all", "every team-mate seen before"), ("absent", "only team-mates absent from the picture"), ("side", "absent and last seen in the same image half")):
        lead = np.array([r[f"{kind}_{name}_lead"] for r in res])
        k = np.array([r[f"{kind}_{name}_n"] for r in res])
        P(f"   {txt:44s}: candidates median {np.median(k):.0f}; true person is the most similar in {int((lead > 0).sum())} of {n} ({100 * np.mean(lead > 0):.1f}%); "
          f"lead >= 0.02: {int((lead >= 0.02).sum())}, >= 0.05: {int((lead >= 0.05).sum())}, >= 0.10: {int((lead >= 0.10).sum())}")
    for name in ("all", "absent"):
        for lab, sel in (("without SNGS-162", [r for r in res if r["seq"] != "SNGS-162"]), ("SNGS-162 only", [r for r in res if r["seq"] == "SNGS-162"]),
                         ("gap > 10 s", [r for r in res if r["gap"] > 250])):
            lead = np.array([r[f"{kind}_{name}_lead"] for r in sel])
            P(f"   {name:6s} {lab:18s}: top-1 {int((lead > 0).sum())} of {len(sel)}")
bad = [(r["seq"], r["p"], round(r["tc_absent_lead"], 3)) for r in res if r["tc_absent_lead"] <= 0]
P("\nreturns where the true person is NOT the most similar absent team-mate (team-centred):", bad)
