"""v_robust_1 / r1_reproduce.py -- (LABELS: scoring only) reproduce fuse_r1's claimed numbers.

1. Re-score fuse_r1's METHOD_*.json files with reid_lab.score.evaluate (the CLI's function).
2. Re-build every METHOD from fuse_r1/params_loco.json through fuse_r1's label-free apply path
   (method.build) in THIS process and compare links and cuts with the files (must be identical).
3. Baselines split / plugin / mutual.
Writes out/r1_reproduce.json (pooled + per clip + per-return outcomes) and out/r1_reproduce.md.
"""
import json
import sys
import time

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_1")
import vr_common as vc
from vr_common import data, method

from reid_lab import score  # LABELS: scoring

FILES = ["strict", "balanced", "recall", "fixed_strict", "fixed_balanced", "fixed_recall", "frozen_strict", "frozen_balanced", "frozen_recall", "cuts_only"]
CLAIMS = {  # from fuse_r1's report: right, wrong, lost, links right / wrong / unknown, impure (rounded as reported)
    "strict": (67, 3, 67, 312, 7, 25, 0.01611), "balanced": (85, 5, 47, 378, 16, 30, 0.02123), "recall": (106, 7, 24, 414, 17, 260, 0.02175),
    "fixed_strict": (70, 2, 65, 319, 6, 22, 0.01445), "fixed_balanced": (85, 4, 48, 361, 12, 30, 0.01813), "fixed_recall": (112, 7, 18, 415, 17, 311, 0.02177),
    "frozen_strict": (69, 2, 66, 315, 7, 23, 0.0152), "frozen_balanced": (86, 4, 47, 362, 12, 29, 0.01814), "frozen_recall": (112, 7, 18, 415, 17, 311, 0.02178),
    "cuts_only": (0, 0, 137, 0, 0, 0, 0.00999),
}
BASE_CLAIMS = {"split": (0, 4, 133), "plugin": (30, 14, 93), "mutual": (26, 6, 105)}


def full(method_clips):
    """score.evaluate + per-return outcomes per clip."""
    res = score.evaluate(method_clips)
    per = {}
    for s in data.DEV_CLIPS:
        m = method_clips.get(s) or {}
        r = score.score_clip(s, m.get("links") or [], m.get("cuts") or [])
        per[s] = {"per_return": r["per_return"], "links": r["links"], "impure_rows": r["impure_rows"], "tracked_rows": r["tracked_rows"]}
    return {"pooled": res["pooled"], "per_clip": per}


def key(p):
    return (p["right"], p["wrong"], p["lost"], p["links_right"], p["links_wrong"], p["links_unknown"], p["impure_row_share"])


if __name__ == "__main__":
    t0 = time.time()
    out = {"methods": {}, "baselines": {}, "rebuild": {}}
    L = ["# r1: reproduction of fuse_r1's numbers", "", "| file | score.evaluate here | claimed | same |", "|---|---|---|---|"]
    for f in FILES:
        M = vc.load_method(f)
        r = full(M)
        out["methods"][f] = r
        k = key(r["pooled"])
        same = k[:6] == CLAIMS[f][:6] and abs(k[6] - CLAIMS[f][6]) < 6e-5
        L.append(f"| METHOD_{f}.json | {vc.fmt_pool(r['pooled'])} ({r['pooled']['impure_row_share']}) | {CLAIMS[f]} | {same} |")
        print(f, vc.fmt_pool(r["pooled"]), r["pooled"]["impure_row_share"], same, flush=True)
    for b in ("split", "plugin", "mutual"):
        r = full(score.baseline_links(b))
        out["baselines"][b] = r
        p = r["pooled"]
        L.append(f"| baseline {b} | {vc.fmt_pool(p)} | {BASE_CLAIMS[b]} | {(p['right'], p['wrong'], p['lost']) == BASE_CLAIMS[b]} |")
        print(b, vc.fmt_pool(p), flush=True)
    out["baselines"]["cuts_only"] = out["methods"]["cuts_only"]
    # rebuild from params (label-free apply path)
    L += ["", "Rebuild through fuse_r1's apply path (method.build / method.cuts_only) from params_loco.json, compared with the files:", ""]
    for f in FILES:
        if f == "cuts_only":
            B = method.cuts_only()["clips"]
        elif f.startswith("frozen_"):
            B = method.build("frozen", f.split("_", 1)[1])["clips"]
        elif f.startswith("fixed_"):
            B = method.build("fixed", f.split("_", 1)[1])["clips"]
        else:
            B = method.build("heldout", f)["clips"]
        M = vc.load_method(f)
        diff = [s for s in data.DEV_CLIPS if sorted(map(tuple, B[s]["links"])) != sorted(map(tuple, M[s]["links"]))
                or sorted(map(tuple, B[s]["cuts"])) != sorted(map(tuple, M[s]["cuts"]))]
        out["rebuild"][f] = {"clips_that_differ": diff, "links": sum(len(B[s]["links"]) for s in B), "cuts": sum(len(B[s]["cuts"]) for s in B)}
        L.append(f"- METHOD_{f}.json: {out['rebuild'][f]['links']} links, {out['rebuild'][f]['cuts']} cuts rebuilt; clips that differ: {diff or 'none'}")
        print("rebuild", f, out["rebuild"][f], round(time.time() - t0, 1), flush=True)
    (vc.OUT / "r1_reproduce.json").write_text(json.dumps(out), encoding="utf-8")
    (vc.OUT / "r1_reproduce.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("done", round(time.time() - t0, 1), "s")
