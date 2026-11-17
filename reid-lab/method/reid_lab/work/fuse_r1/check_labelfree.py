"""fuse_r1 / check_labelfree.py -- run-time proof that the apply path reads no annotation.

An audit hook records every file this process opens. Then the apply path is run from nothing for
two clips (cut detector, a from-scratch rebuild of all evidence tables in a throw-away cache, the
two-stage linker, held-out and frozen numbers, with and without leaves) plus the self-check.
Passes when: no opened path contains "Labels-GameState", "reentry-eval" or "sn-gsr"; the modules
reid_lab.score, gbf.evaluate and fuse_labels were never imported; and the links equal the ones in
the METHOD files.

    python -I check_labelfree.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
opened = []
sys.addaudithook(lambda ev, args: opened.append(str(args[0])) if ev == "open" and args and isinstance(args[0], (str, bytes, Path)) else None)

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse_r1")
import prep  # noqa: E402

HERE = Path(__file__).resolve().parent
tmp = HERE / "cache_check"
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir()
prep.CACHE = tmp                      # nothing cached is reused: cuts and tables are rebuilt
prep._CUTS.clear()
import method  # noqa: E402
import selfcheck  # noqa: E402

params = method.load_params()
norm = lambda ls: sorted(tuple(sorted(map(int, l))) for l in ls)  # noqa: E731
bad = 0
for seq in ("SNGS-162", "SNGS-066"):
    for mode, key, fold in (("heldout", seq, seq), ("frozen", method.ALL, None)):
        for point in method.POINTS:
            ch = params[key]["choice"][point]
            got = method.clip_method(seq, ch["margin"], ch["leaves"], params[key], fold)
            name = f"METHOD_{point}.json" if mode == "heldout" else f"METHOD_frozen_{point}.json"
            want = json.loads((HERE / name).read_text(encoding="utf-8"))["clips"][seq]
            same = norm(got["links"]) == norm(want["links"]) and norm(got["cuts"]) == norm(want["cuts"])
            bad += not same
            print(seq, mode, point, len(got["links"]), "links", len(got["cuts"]), "cuts", "= METHOD file" if same else "DIFFERENT from the METHOD file")
    selfcheck.pairs(seq, prep.cuts_for(seq, None))
hits = sorted({p for p in opened if any(k in p.replace("\\", "/") for k in ("Labels-GameState", "reentry-eval", "sn-gsr", "/external/"))})
mods = [m for m in ("reid_lab.score", "gbf.evaluate", "gbf.evaluate.gsr", "fuse_labels") if m in sys.modules]
print("files opened:", len(opened), "; annotation files among them:", hits or "none")
print("label modules imported:", mods or "none")
shutil.rmtree(tmp)
bad += bool(hits) + bool(mods)
print("APPLY PATH IS LABEL-FREE AND REPRODUCES THE METHOD FILES" if not bad else f"{bad} PROBLEMS")
sys.exit(1 if bad else 0)
