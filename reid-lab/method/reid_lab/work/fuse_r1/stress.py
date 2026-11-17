"""fuse_r1 / stress.py -- what happens to the method when the appearance embeddings are worse than on
the dev clips (LABELS: scoring only). A SYNTHETIC stress test, not a result.

Why: the re-id weights were very probably trained on the SoccerNet-GSR train split, which all dev
clips come from, so appearance may separate people less well on other footage (v_leak's finding).
That cannot be measured here. What can be shown is in which direction the method fails when the
embeddings are degraded and everything fitted stays as it is: more "lost" (safe) or more "wrong".

Every clip is processed exactly as in the held-out run (its fold's cut detector, position weights,
pair model and chosen settings from params_loco.json; nothing is refitted), except that
reid_lab.data.embeddings is replaced in this process by a degraded copy BEFORE anything reads it
(m_cuts' cuts, m_appear's prototypes and whitening, the self-check all see the degraded vectors):

  lowdim<k>   only the k strongest principal directions of the clip's embeddings are kept (PCA on the
              clip's own rows, label-free): fine detail is removed, so people look more alike
              (different-people cosines rise) while one person stays like himself.
  drift<s>    every stretch of 2 s of every gbf piece gets its own random shift inside the 16
              strongest directions in which same-team pieces differ, of size s x the typical distance
              between same-team pieces in those directions: one person looks different from one
              tracklet stretch to the next (same-person cosines fall, and by chance some other
              player gets closer).
  none        no change (must reproduce the held-out numbers of the METHOD files).

    python -I stress.py run <level>        # one level -> out/stress/<level>.json  (uses cache_stress/<level>/)
    python -I stress.py all [workers]      # every level, then out/stress.md
    python -I stress.py table              # out/stress.md from the existing files
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse_r1")
from reid_lab import data  # noqa: E402

HERE = Path(__file__).resolve().parent
LEVELS = ("none", "lowdim64", "lowdim32", "lowdim16", "lowdim8", "drift0.5", "drift1.0", "drift2.0")
BLOCK = 50          # frames per drift block (2 s)
K_ID = 16           # directions of the drift
SEED = 20261115


def degrade(E: np.ndarray, seq: str, level: str) -> np.ndarray:
    if level == "none":
        return E
    has = ~np.isnan(E[:, 0])
    X = E[has]
    out = E.copy()
    if level.startswith("lowdim"):
        k = int(level[6:])
        mu = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
        U = Vt[:k].T
        Y = mu + (X - mu) @ U @ U.T
    else:
        s = float(level[5:])
        t = data.tracks(seq)
        tid = t.track_id.to_numpy()[has]
        team = t.team.to_numpy()[has]
        frame = t.frame.to_numpy()[has]
        # same-team piece means, centred per team -> the directions in which same-team pieces differ
        means, cent = [], {}
        for tm in ("left", "right"):
            m = team == tm
            if m.sum() >= 50:
                cent[tm] = X[m].mean(axis=0)
        for p in np.unique(tid[tid >= 0]):
            m = tid == p
            tm = Counter(team[m].tolist()).most_common(1)[0][0]
            if m.sum() >= 5 and tm in cent:
                means.append(X[m].mean(axis=0) - cent[tm])
        M = np.array(means)
        _, _, Vt = np.linalg.svd(M - M.mean(axis=0), full_matrices=False)
        U = Vt[:K_ID].T
        scale = float(np.sqrt(((M @ U) ** 2).sum(axis=1).mean()))      # typical size of a piece's deviation in those directions
        rng = np.random.default_rng([SEED, int(seq.split("-")[1])])
        key = tid.astype(np.int64) * 100000 + frame // BLOCK
        uniq, inv = np.unique(key, return_inverse=True)
        G = rng.standard_normal((len(uniq), K_ID)) / np.sqrt(K_ID)
        Y = X + s * scale * (G[inv] @ U.T)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    out[has] = Y
    return out


def run_level(level: str) -> dict:
    orig = data.embeddings
    cache: dict = {}

    def patched(seq):
        if seq not in cache:
            cache[seq] = degrade(orig(seq), seq, level)
        return cache[seq]

    data.embeddings = patched                     # this process only; nothing has read the embeddings yet
    data.clean_rows.cache_clear()
    import prep
    prep.CACHE = HERE / "cache_stress" / level    # degraded tables never go into the method's cache
    prep.CACHE.mkdir(parents=True, exist_ok=True)
    prep._CUTS.clear()
    import method
    import selfcheck
    from reid_lab import score                    # LABELS: scoring only

    params = method.load_params()
    out = {"level": level, "points": {}, "selfcheck": {}, "cuts": 0}
    per = {p: {} for p in method.POINTS}
    pos_all, neg_all = [], []
    for seq in data.DEV_CLIPS:
        par = params[seq]
        for p in method.POINTS:
            ch = par["choice"][p]
            cuts, core, leaf = method.clip_links(seq, ch["margin"], ch["leaves"], par, seq)
            r = score.score_clip(seq, core + leaf, [list(x) for x in cuts])
            per[p][seq] = {**dict(Counter(r["per_return"])), "lk": r["links"], "impure": r["impure_rows"], "rows": r["tracked_rows"]}
        out["cuts"] += len(cuts)
        r0 = score.score_clip(seq, [], [list(x) for x in cuts])
        per.setdefault("cuts_only", {})[seq] = {**dict(Counter(r0["per_return"])), "lk": r0["links"], "impure": r0["impure_rows"], "rows": r0["tracked_rows"]}
        pos, neg = selfcheck.pairs(seq, cuts)
        pos_all.append(pos)
        neg_all.append(neg)
    for p, d in per.items():
        g = lambda k: int(sum(x.get(k, 0) for x in d.values()))  # noqa: E731
        lk = lambda k: int(sum(x["lk"][k] for x in d.values()))  # noqa: E731
        out["points"][p] = {"right": g("right"), "wrong": g("wrong"), "lost": g("lost"), "links_right": lk("right"), "links_wrong": lk("wrong"),
                            "links_unknown": lk("unknown"), "impure_row_share": round(g("impure") / g("rows"), 5),
                            "wrong_without_SNGS-162": int(sum(x.get("wrong", 0) for s, x in d.items() if s != "SNGS-162")),
                            "per_clip": d}
    out["selfcheck"] = selfcheck.summary(np.concatenate(pos_all), np.concatenate(neg_all))
    (HERE / "out" / "stress").mkdir(parents=True, exist_ok=True)
    (HERE / "out" / "stress" / f"{level}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def table() -> str:
    L = ["# fuse_r1: stress test with degraded embeddings (synthetic; held-out settings, nothing refitted)", "",
         "Cells: right / wrong / lost (wrong outside SNGS-162) | links right / wrong / unknown | impure rows. 137 scored returns.",
         "Self-check columns are label-free (selfcheck.py, pooled over the 13 clips, on the degraded embeddings).", "",
         "| embeddings | cuts | cuts only | strict | balanced | recall | self-check AUC | same 10% / 50% | different 95% / 99% | different >= 0.5 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for level in LEVELS:
        f = HERE / "out" / "stress" / f"{level}.json"
        if not f.exists():
            continue
        o = json.loads(f.read_text(encoding="utf-8"))
        cell = lambda p: (f"{p['right']} / {p['wrong']} / {p['lost']} ({p['wrong_without_SNGS-162']}) | {p['links_right']} / {p['links_wrong']} / "  # noqa: E731
                          f"{p['links_unknown']} | {100 * p['impure_row_share']:.2f}%")
        c0 = o["points"]["cuts_only"]
        s = o["selfcheck"]
        L.append(f"| {level} | {o['cuts']} | {c0['right']} / {c0['wrong']} / {c0['lost']}, {100 * c0['impure_row_share']:.2f}% | "
                 + " | ".join(cell(o["points"][p]) for p in ("strict", "balanced", "recall"))
                 + f" | {s['auc']} | {s['same_q10']} / {s['same_q50']} | {s['diff_q95']} / {s['diff_q99']} | {s['diff_share_ge_0.5']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "run":
        o = run_level(sys.argv[2])
        print(sys.argv[2], {p: (v["right"], v["wrong"], v["lost"]) for p, v in o["points"].items()}, o["selfcheck"], flush=True)
    else:
        if what == "all":
            workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
            todo, running = list(LEVELS), []
            while todo or running:                # one process per level; failed processes are reported, not hidden
                while todo and len(running) < workers:
                    lv = todo.pop(0)
                    running.append((lv, subprocess.Popen([sys.executable, "-I", "-B", str(HERE / "stress.py"), "run", lv])))
                lv, pr = running.pop(0)
                code = pr.wait()
                print(lv, "exit", code, flush=True)
                if code != 0:
                    raise SystemExit(f"level {lv} failed with exit code {code}")
        text = table()
        (HERE / "out" / "stress.md").write_text(text, encoding="utf-8")
        print(text)
