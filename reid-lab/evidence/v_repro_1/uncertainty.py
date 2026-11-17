"""fuse_r1 / uncertainty.py -- how sure the pooled counts are (LABELS: scoring only).

    python -I uncertainty.py [METHOD files ...]     -> out/uncertainty.md, out/uncertainty.json

For every METHOD file, from its per-clip counts (score.evaluate):
  - clip bootstrap (the 13 clips resampled 5000 times, seed 0): 5% / 50% / 95% of the right share and
    of the wrong share; the chance that the wrong share is within the point's limit (4, 7, 14 of 137);
  - paired with the baselines on the same resamples: chance that the wrong share is not above the
    baseline's and that the right share is above it (gbf now = split, plugin, mutual);
  - the pooled result without each single clip (range), and without SNGS-162;
  - how many of the wrong returns sit in the clip with the most.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402  LABELS: scoring

HERE = Path(__file__).resolve().parent
N_BOOT, SEED = 5000, 0
CAPS = {"strict": 4, "balanced": 7, "recall": 14}


def per_clip(method_clips: dict) -> np.ndarray:
    """(13, 3) right, wrong, scored per clip."""
    ev = score.evaluate(method_clips)["per_clip"]
    return np.array([[ev[s]["returns"]["right"], ev[s]["returns"]["wrong"],
                      ev[s]["returns"]["right"] + ev[s]["returns"]["wrong"] + ev[s]["returns"]["lost"]] for s in data.DEV_CLIPS], float)


def main(files):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(data.DEV_CLIPS), size=(N_BOOT, len(data.DEV_CLIPS)))
    base = {n: per_clip(score.baseline_links(n)) for n in ("split", "plugin", "mutual")}
    share = lambda M: (M[idx][:, :, 0].sum(axis=1) / M[idx][:, :, 2].sum(axis=1), M[idx][:, :, 1].sum(axis=1) / M[idx][:, :, 2].sum(axis=1))  # noqa: E731
    bshare = {n: share(M) for n, M in base.items()}
    rep = {}
    L = ["# fuse_r1: uncertainty of the pooled counts (clip bootstrap, 13 clips resampled 5000 times, seed 0)", "",
         "Baselines: " + "; ".join(f"{n} {int(M[:, 0].sum())} / {int(M[:, 1].sum())} / {int((M[:, 2] - M[:, 0] - M[:, 1]).sum())}" for n, M in base.items()), ""]
    q = lambda v: [round(float(x), 3) for x in np.quantile(v, [0.05, 0.5, 0.95])]  # noqa: E731
    for f in files:
        m = json.loads((HERE / f).read_text(encoding="utf-8"))
        M = per_clip(m["clips"])
        rs, ws = share(M)
        point = next((p for p in CAPS if p in f), None)
        r = {"right": int(M[:, 0].sum()), "wrong": int(M[:, 1].sum()), "scored": int(M[:, 2].sum()),
             "right_share_q05_q50_q95": q(rs), "wrong_share_q05_q50_q95": q(ws),
             "wrong_count_equiv_q05_q50_q95": [round(137 * x, 1) for x in q(ws)]}
        for cap in (4, 7, 14):
            r[f"P_wrong_share_le_{cap}_of_137"] = round(float((ws <= cap / 137 + 1e-12).mean()), 3)
        for n, (brs, bws) in bshare.items():
            r[f"P_wrong_not_above_{n}"] = round(float((ws <= bws + 1e-12).mean()), 3)
            r[f"P_wrong_below_{n}"] = round(float((ws < bws - 1e-12).mean()), 3)
            r[f"P_right_above_{n}"] = round(float((rs > brs + 1e-12).mean()), 3)
            d = rs - brs
            r[f"right_share_gain_over_{n}_q05_q95"] = [round(float(x), 3) for x in np.quantile(d, [0.05, 0.95])]
        loo = [(M[np.arange(13) != i, 0].sum() / M[np.arange(13) != i, 2].sum(), M[np.arange(13) != i, 1].sum() / M[np.arange(13) != i, 2].sum(),
                int(M[np.arange(13) != i, 1].sum())) for i in range(13)]
        r["without_one_clip_right_share_min_max"] = [round(float(min(x[0] for x in loo)), 3), round(float(max(x[0] for x in loo)), 3)]
        r["without_one_clip_wrong_count_min_max"] = [min(x[2] for x in loo), max(x[2] for x in loo)]
        i162 = data.DEV_CLIPS.index("SNGS-162")
        keep = np.arange(13) != i162
        r["without_SNGS-162"] = [int(M[keep, 0].sum()), int(M[keep, 1].sum()), int((M[keep, 2] - M[keep, 0] - M[keep, 1]).sum())]
        r["SNGS-162"] = [int(M[i162, 0]), int(M[i162, 1]), int(M[i162, 2] - M[i162, 0] - M[i162, 1])]
        r["wrong_in_worst_clip"] = int(M[:, 1].max())
        r["clips_with_a_wrong_return"] = int((M[:, 1] > 0).sum())
        r["clips_better_than_plugin_in_right"] = int((M[:, 0] > base["plugin"][:, 0]).sum())
        r["clips_worse_than_plugin_in_right"] = int((M[:, 0] < base["plugin"][:, 0]).sum())
        rep[f] = r
        cap = CAPS.get(point)
        L += [f"## {f}", "",
              f"- pooled: right {r['right']}, wrong {r['wrong']} of {r['scored']}; without SNGS-162: {r['without_SNGS-162'][0]} / {r['without_SNGS-162'][1]} / {r['without_SNGS-162'][2]}; "
              f"SNGS-162 alone: {r['SNGS-162'][0]} / {r['SNGS-162'][1]} / {r['SNGS-162'][2]}",
              f"- right share 5% / 50% / 95%: {r['right_share_q05_q50_q95']}; wrong share: {r['wrong_share_q05_q50_q95']} (as counts of 137: {r['wrong_count_equiv_q05_q50_q95']})",
              "- chance that the wrong share is within the limit: " + ", ".join(f"<= {c}/137: {r[f'P_wrong_share_le_{c}_of_137']}" for c in (4, 7, 14))
              + (f" (this point's limit is {cap})" if cap else ""),
              "- against the baselines (same resamples): " + "; ".join(
                  f"{n}: wrong not above {r[f'P_wrong_not_above_{n}']}, wrong below {r[f'P_wrong_below_{n}']}, right above {r[f'P_right_above_{n}']}, "
                  f"right-share gain 5%..95% {r[f'right_share_gain_over_{n}_q05_q95']}" for n in base),
              f"- with any one clip removed: right share {r['without_one_clip_right_share_min_max']}, wrong count {r['without_one_clip_wrong_count_min_max']}",
              f"- wrong returns: {r['wrong_in_worst_clip']} of {r['wrong']} in one clip; {r['clips_with_a_wrong_return']} clips have any",
              f"- right returns against the plugin per clip: better in {r['clips_better_than_plugin_in_right']}, worse in {r['clips_worse_than_plugin_in_right']} of 13", ""]
    text = "\n".join(L) + "\n"
    (HERE / "out" / "uncertainty.md").write_text(text, encoding="utf-8")
    (HERE / "out" / "uncertainty.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["METHOD_strict.json", "METHOD_balanced.json", "METHOD_recall.json",
                          "METHOD_fixed_strict.json", "METHOD_fixed_balanced.json", "METHOD_fixed_recall.json"])
