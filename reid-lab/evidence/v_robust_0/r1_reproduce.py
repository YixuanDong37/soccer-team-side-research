"""v_robust_0 / r1_reproduce.py -- (LABELS: scoring only) reproduce fuse's claimed numbers.

1. Re-score fuse's METHOD_*.json files with reid_lab.score.evaluate (the CLI's function).
2. Re-build every METHOD from fuse/params_loco.json through fuse's label-free apply path
   (method.build) in THIS process and compare links and cuts with the files (must be identical).
3. Baselines split / plugin / mutual and "cuts only" (fuse's cuts, no links).
Writes out/r1_reproduce.json (pooled + per clip + per-return outcomes) and out/r1_reproduce.md.
"""
import json
import sys
import time

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_robust_0")
import vr_common as vc
from vr_common import data, method

from reid_lab import score  # LABELS: scoring

FILES = ["strict", "balanced", "recall", "margin_0.50", "frozen_strict", "frozen_balanced", "frozen_recall"]
CLAIMS = {  # from fuse's report: right, wrong, lost, links right / wrong / unknown, impure
    "strict": (33, 4, 100, 152, 4, 158, 0.01284), "balanced": (93, 7, 37, 290, 10, 376, 0.01887),
    "recall": (110, 7, 20, 314, 10, 437, 0.01909), "margin_0.50": (91, 4, 42, 272, 7, 345, 0.01511),
    "frozen_strict": (37, 0, 100, 152, 2, 152, 0.01052), "frozen_balanced": (109, 6, 22, 310, 10, 424, 0.01909),
    "frozen_recall": (112, 7, 18, 314, 11, 441, 0.02132),
}


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
    L = ["# r1: reproduction of fuse's numbers", "", "| file | score.evaluate here | claimed | same |", "|---|---|---|---|"]
    for f in FILES:
        M = vc.load_method(f)
        r = full(M)
        out["methods"][f] = r
        same = key(r["pooled"]) == CLAIMS[f]
        L.append(f"| METHOD_{f}.json | {vc.fmt_pool(r['pooled'])} | {CLAIMS[f]} | {same} |")
        print(f, vc.fmt_pool(r["pooled"]), same, flush=True)
    for b in ("split", "plugin", "mutual"):
        r = full(score.baseline_links(b))
        out["baselines"][b] = r
        L.append(f"| baseline {b} | {vc.fmt_pool(r['pooled'])} | | |")
        print(b, vc.fmt_pool(r["pooled"]), flush=True)
    # cuts only: the held-out cuts of every clip, no links
    for name, src in (("cuts_only_heldout", "recall"), ("cuts_only_frozen", "frozen_recall")):
        M = vc.load_method(src)
        r = full({s: {"links": [], "cuts": M[s]["cuts"]} for s in data.DEV_CLIPS})
        out["baselines"][name] = r
        L.append(f"| {name} (cuts of METHOD_{src}, no links) | {vc.fmt_pool(r['pooled'])} | | |")
        print(name, vc.fmt_pool(r["pooled"]), flush=True)
    # rebuild from params (label-free apply path)
    L += ["", "Rebuild through fuse's apply path (method.build) from params_loco.json, compared with the files:", ""]
    for f in FILES:
        if f.startswith("frozen_"):
            B = method.build("frozen", f.split("_", 1)[1])["clips"]
        elif f.startswith("margin_"):
            B = method.build("heldout", margin=float(f.split("_")[1]))["clips"]
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
