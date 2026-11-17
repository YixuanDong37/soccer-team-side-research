"""Label-free check of every METHOD file: within every linked group no two pieces share a frame
(a person is in one place at a time), no group mixes teams or holds a referee, every piece has
at most one earlier and one later partner, and every linked id exists after the cuts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
from reid_lab import data  # noqa: E402
import prep  # noqa: E402

af = prep._import("m_appear", "features")
core = sys.modules["m_appear__core"]
HERE = Path(__file__).resolve().parent
bad = 0
for path in sorted(HERE.glob("METHOD_*.json")):
    M = json.loads(path.read_text(encoding="utf-8"))["clips"]
    groups_n = links_n = 0
    for seq in data.DEV_CLIPS:
        cuts = prep.norm_cuts(M[seq]["cuts"])
        c = core.ctx(seq, cuts)
        pr = af.protos(seq, cuts)
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        nxt, prv = {}, {}
        for a, b in M[seq]["links"]:
            if a not in c.frames_of or b not in c.frames_of:
                bad += 1
                print("unknown piece", path.name, seq, a, b)
                continue
            if a in nxt or b in prv:
                bad += 1
                print("second partner", path.name, seq, a, b)
            nxt[a], prv[b] = b, a
            parent[find(a)] = find(b)
            links_n += 1
        members = {}
        for x in parent:
            members.setdefault(find(x), []).append(x)
        for g in members.values():
            groups_n += 1
            seen = set()
            for p in g:
                if seen & c.frames_of[p]:
                    bad += 1
                    print("shared frame", path.name, seq, p)
                seen |= c.frames_of[p]
            if len({pr.ateam[p] for p in g}) > 1 or any(c.P.role[c.index[p]] == "referee" for p in g):
                bad += 1
                print("mixed team or referee", path.name, seq, g)
    print(path.name, links_n, "links in", groups_n, "groups: ok" if not bad else "")
print("all groups consistent" if not bad else f"{bad} problems")
