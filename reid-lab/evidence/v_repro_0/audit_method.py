"""v_repro / audit_method.py -- label hygiene of the apply path, checked at run time (no labels read here).

A Python audit hook records every file the process opens while (1) prep.py's tables of one clip are
built from nothing into a scratch cache (m_cuts' cuts, m_appear's and m_space's pair tables) and
(2) method.py builds the three held-out METHOD dicts and the three frozen ones from params_loco.json.
It then reports whether an annotation file, reid_lab/score.py or gbf.evaluate was touched, and
whether the METHOD dicts equal the METHOD_*.json files on disk.

    python -I -B audit_method.py
"""
import json
import sys
from pathlib import Path

opened = []


def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        opened.append(str(args[0]).replace("\\", "/"))


sys.addaudithook(hook)
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
import prep  # noqa: E402

HERE = Path(__file__).resolve().parent

# (1) tables of one clip from nothing, in a scratch cache, in this process
real_cache = prep.CACHE
scratch = HERE / "cache_audit"
scratch.mkdir(exist_ok=True)
for f in scratch.glob("*"):
    f.unlink()
prep.CACHE = scratch
prep._CUTS.clear()
seq = "SNGS-162"
for fold in (None, "SNGS-162", "SNGS-107"):
    cuts = prep.cuts_for(seq, fold)
    a = prep.app_table(seq, cuts)
    s, e = prep.space_tables(seq, cuts)
    import pandas as pd

    key = prep.cut_key(cuts)
    for name, df in (("app", a), ("space", s), ("ends", e)):
        ref = pd.read_pickle(real_cache / f"{name}_{seq}_{key}.pkl")
        pd.testing.assert_frame_equal(df, ref, check_exact=True)
    print("tables rebuilt in-process equal the cache:", seq, "fold", fold, len(cuts), "cuts", flush=True)
prep.CACHE = real_cache
prep._CUTS.clear()

# (2) the METHOD dicts
import method  # noqa: E402

bad = 0
for mode in ("heldout", "frozen"):
    for point in method.POINTS:
        m = method.build(mode, point)
        name = f"METHOD_{point}.json" if mode == "heldout" else f"METHOD_frozen_{point}.json"
        disk = json.loads((HERE / name).read_text(encoding="utf-8"))
        same = disk == m
        bad += not same
        print(name, "equal to the file on disk" if same else "DIFFERENT from the file on disk", flush=True)

labels = sorted({p for p in opened if "Labels-GameState" in p or "/external/" in p or "sn-gsr" in p.lower()})
scorer = sorted({p for p in opened if p.lower().endswith("reid_lab/score.py") or "/gbf/evaluate/" in p.lower()})
others = sorted({p for p in opened if "/gbf-data/" in p and "/gsr-runs/SNGS-" not in p})
mods = [k for k in sys.modules if k in ("reid_lab.score", "gbf.evaluate", "gbf.evaluate.gsr") or k.endswith("fitdata") or k.endswith("fitlabels")
        or k.endswith("fuse_labels") or k.endswith("__labels") or k.endswith("__truth") or k in ("labels", "truth")]
print("files opened:", len(opened), "distinct:", len(set(opened)))
print("annotation files opened:", labels)
print("scorer / gbf.evaluate files opened:", scorer)
print("gbf-data paths outside gsr-runs/SNGS-*:", others)
print("label-reading modules loaded:", mods)
roots = sorted({"/".join(p.split("/")[:6]) for p in set(opened) if p[1:3] == ":/"})
print("path roots opened:", roots)
print("AUDIT", "clean" if not (labels or scorer or others or mods or bad) else "PROBLEM")
