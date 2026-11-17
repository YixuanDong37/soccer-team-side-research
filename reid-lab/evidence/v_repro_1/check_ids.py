"""v_repro_1 / check_ids.py -- label-free structural check of METHOD files, written independently of
fuse_r1's check_method.py: every link names two sub-pieces that exist in refine/tracks.parquet after the
cuts (score.py would silently count a link to a non-existent id as "unknown"), every cut splits a
piece into two non-empty parts, no identity holds two boxes in one frame, no link is repeated or
joins a piece to itself, and how many rows / pieces end up in multi-piece identities.
Reads tracks.parquet only (no annotation, no reid_lab).

    python -I -B check_ids.py METHOD_a.json [METHOD_b.json ...]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("C:/Research/gbf-data/gsr-runs")
SUB = 10 ** 7
bad = 0
for path in sys.argv[1:]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))["clips"]
    tot = Counter()
    for seq, c in m.items():
        t = pd.read_parquet(RUNS / seq / "gbf/refine/tracks.parquet")[["frame", "track_id", "role", "team"]]
        t = t[t.track_id >= 0]
        cuts = defaultdict(list)
        for p, f in c.get("cuts") or []:
            cuts[int(p)].append(int(f))
        piece, frame = t.track_id.to_numpy(), t.frame.to_numpy()
        sub = piece.copy()
        for p, fs in cuts.items():
            if not (piece == p).any():
                print(path, seq, "cut of a piece that does not exist:", p)
                bad += 1
            for j, f in enumerate(sorted(set(fs))):
                sel = (piece == p) & (frame >= f)
                sub[sel] = p + SUB * (j + 1)
            ids = [p] + [p + SUB * (j + 1) for j in range(len(set(fs)))]
            if any(not (sub == i).any() for i in ids):
                print(path, seq, "cut that leaves an empty part:", p, sorted(fs))
                bad += 1
            if len(set(fs)) != len(fs):
                print(path, seq, "repeated cut:", p, fs)
                bad += 1
        exist = set(np.unique(sub).tolist())
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        seen = set()
        for a, b in c.get("links") or []:
            a, b = int(a), int(b)
            if a not in exist or b not in exist:
                print(path, seq, "link to an id without rows:", a, b)
                bad += 1
            if a == b or (min(a, b), max(a, b)) in seen:
                print(path, seq, "self link or repeated link:", a, b)
                bad += 1
            seen.add((min(a, b), max(a, b)))
            if find(a) == find(b):
                print(path, seq, "link inside an identity that is already connected (cycle):", a, b)
                bad += 1
            parent[find(a)] = find(b)
        linked = set(parent)                      # the sub-pieces named in a link (before find() below adds the rest)
        g = np.array([find(int(x)) for x in sub])
        dup = pd.DataFrame({"g": g, "f": frame}).duplicated().sum()
        if dup:
            print(path, seq, "identity with two boxes in one frame:", int(dup), "rows")
            bad += 1
        role = t.role.to_numpy()
        major = pd.DataFrame({"s": sub, "r": role}).groupby("s")["r"].agg(lambda v: v.value_counts().index[0])
        refs = [x for x in linked if major.get(x) == "referee"]
        if refs:
            print(path, seq, "linked sub-pieces whose majority role is referee:", refs)
            bad += 1
        tot["referee_rows_in_linked"] += int(np.isin(sub, list(linked))[role == "referee"].sum()) if linked else 0
        tot["links"] += len(c.get("links") or [])
        tot["cuts"] += len(c.get("cuts") or [])
        tot["pieces"] += len(exist)
        tot["identities"] += len(set(g.tolist()))
        tot["linked_pieces"] += len(linked)
    print(f"{Path(path).name}: {tot['links']} links, {tot['cuts']} cuts, {tot['pieces']} sub-pieces -> {tot['identities']} identities "
          f"({tot['linked_pieces']} sub-pieces take part in a link; {tot['referee_rows_in_linked']} of their rows carry the role referee)")
print("ALL METHOD FILES STRUCTURALLY CONSISTENT" if not bad else f"{bad} PROBLEMS")
sys.exit(1 if bad else 0)
