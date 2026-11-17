"""fuse_r1 / check_method.py -- label-free consistency check: the METHOD files written by method.py
(apply path, reading params_loco.json) hold exactly the links and cuts that fit_loco.py scored for
the chosen settings (cache/loco_raw.pkl), for the held-out, the frozen and the fixed files.
Also re-checks the structure of every file: no core link with a short piece, a short piece has at
most one link, no group with two pieces in one frame, mixed appearance teams or a referee."""
import json
import pickle
import sys
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/fuse_r1")
from reid_lab import data
import linker, prep

HERE = "C:/Research/plugins/tools/reid_lab/work/fuse_r1/"
raw = pickle.load(open(HERE + "cache/loco_raw.pkl", "rb"))["folds"]
params = json.load(open(HERE + "params_loco.json", encoding="utf-8"))
norm = lambda ls: sorted(tuple(sorted(map(int, l))) for l in ls)  # noqa: E731
bad = 0
for point in ("strict", "balanced", "recall"):
    for mode, name in (("heldout", f"METHOD_{point}.json"), ("frozen", f"METHOD_frozen_{point}.json"), ("fixed", f"METHOD_fixed_{point}.json")):
        m = json.load(open(HERE + name, encoding="utf-8"))["clips"]
        for s in data.DEV_CLIPS:
            fold = "all13" if mode == "frozen" else s
            ch = params["all13" if mode in ("frozen", "fixed") else s]["choice"][point]
            r = raw[fold]["res"][("r1", ch["margin"], s)]
            want = r["core"] + (r["leaf"] if ch["leaves"] else [])
            ok = norm(want) == norm(m[s]["links"]) and norm(raw[fold]["cuts"][s]) == norm(m[s]["cuts"])
            bad += not ok
            # structure
            cuts = prep.norm_cuts(m[s]["cuts"])
            af, core = linker._appear()
            c = core.ctx(s, cuts)
            pr = af.protos(s, cuts)
            solid = prep.solid_ids(s, cuts)
            deg = Counter(p for l in m[s]["links"] for p in l)
            problems = []
            if any(deg[p] > 1 for p in deg if p not in solid):
                problems.append("short piece with two links")
            if any(a not in solid and b not in solid for a, b in m[s]["links"]):
                problems.append("link between two short pieces")
            parent = {}

            def find(x):
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b in m[s]["links"]:
                parent[find(a)] = find(b)
            groups = {}
            for x in list(parent):
                groups.setdefault(find(x), []).append(x)
            for g in groups.values():
                if max(Counter(f for p in g for f in c.frames_of[p]).values()) > 1:
                    problems.append("two pieces of a group in one frame")
                if len({pr.ateam[p] for p in g}) > 1:
                    problems.append("mixed teams")
                if any(c.P.role[c.index[p]] == "referee" for p in g):
                    problems.append("referee")
            bad += bool(problems)
            if not ok or problems:
                print(name, s, "links/cuts equal" if ok else "LINKS OR CUTS DIFFER", problems)
        print(name, "checked")
print("method.py REPRODUCES the scored links; all groups consistent" if not bad else f"{bad} PROBLEMS")
