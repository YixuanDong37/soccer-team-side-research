"""v_leak round 1, stage 2 (NO labels: the annotation of EVERY clip is blindfolded in every worker).

Builds, from nothing, the label-free evidence tables that the blind folds (s3_blind_r1.py) read:
for every dev clip and every cut list a fold uses for it (round 0's blindfolded m_cuts re-fits,
v_leak_0/refit/cuts_<fold>.json), m_appear's pair table and m_space's pair tables / piece ends on all
pieces and on the solid pieces only (solid from 5, 3 and 10 boxes), with fuse_r1's prep.py, into this
directory's cache. fuse_r1's own cache is never read. Every worker logs the files it opened.

    python -I -B s2_tables.py [workers]     -> cache/tables/*.pkl, out/s2_tables.log
"""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = "C:/Research/plugins/tools/reid_lab/work/v_leak_1/"
V0 = "C:/Research/plugins/tools/reid_lab/work/v_leak_0/"
R1 = "C:/Research/plugins/tools/reid_lab/work/fuse_r1/"
N_SOLIDS = (0, 5, 3, 10)


def _setup():
    sys.path.insert(0, HERE)
    import blind

    if not getattr(blind, "_installed", False):
        blind.install("SNGS-")             # no clip's annotation can be opened in this process
        blind._installed = True
    sys.path.insert(0, "C:/Research/plugins/tools")
    sys.path.insert(0, R1)
    import prep

    prep.CACHE = Path(HERE) / "cache" / "tables"
    prep._CUTS.clear()
    return blind, prep


def build(job):
    blind, prep = _setup()
    seq, cuts = job
    t0 = time.time()
    a = prep.app_table(seq, cuts)
    n = [len(a)]
    for k in N_SOLIDS:
        s, e = prep.space_tables(seq, cuts, k)
        n += [len(s), len(e)]
    aud = blind.audit(list(blind.OPENED))
    bad = aud["label_files"] + aud["reentry_eval"] + aud["blocked"]
    foreign = [p for p in aud["work_derived"] if not p.lower().startswith(HERE.lower())]
    return seq, prep.cut_key(cuts), len(cuts), n, round(time.time() - t0, 1), bad, foreign


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    blind, prep = _setup()
    from reid_lab import data

    prep.CACHE.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for fold in data.DEV_CLIPS:
        cuts = json.load(open(V0 + f"refit/cuts_{fold}.json", encoding="utf-8"))["cuts"]
        for seq in data.DEV_CLIPS:
            jobs[(seq, prep.norm_cuts(cuts[seq]))] = None
    jobs = list(jobs)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    t0 = time.time()
    bad = foreign = 0
    with ProcessPoolExecutor(workers) as ex, open(HERE + "out/s2_tables.log", "w", encoding="utf-8") as log:
        for r in ex.map(build, jobs):
            line = f"{r[0]} cuts {r[1]} ({r[2]}) rows {r[3]} {r[4]} s | label files opened or blocked: {r[5]} | files of other agents' work read: {r[6]}"
            print(line, flush=True)
            log.write(line + "\n")
            bad += len(r[5])
            foreign += len(r[6])
        last = f"{len(jobs)} (clip, cut list) tables in {round(time.time() - t0, 1)} s; label files opened or blocked: {bad}; fitted / cached files of other agents read: {foreign}"
        print(last)
        log.write(last + "\n")
    sys.exit(1 if bad or foreign else 0)
