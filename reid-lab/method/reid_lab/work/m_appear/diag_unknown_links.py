"""DIAGNOSIS (labels): the links score.py calls "unknown" (a piece with < 5 matched rows of one person).
Soft identity = the majority annotated person among the piece's matched rows, however few. Not a method result."""
import sys, os, json; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from reid_lab import data, score
import fitlabels
for path in sys.argv[1:]:
    m = json.load(open(path, encoding="utf-8"))["clips"]; tot = Counter(); rows = Counter()
    for s in data.DEV_CLIPS:
        c = score.clip(s); ident = fitlabels.piece_identity(s)
        soft = {}
        for pid in set(c.piece[c.piece >= 0].tolist()):
            cnt = Counter(c.refs[c.piece == pid].tolist()); soft[int(pid)] = (cnt.most_common(1)[0][0], sum(cnt.values()))
        nrows = data.pieces(s).set_index("piece").rows.to_dict()
        for a, b in (m.get(s) or {}).get("links", []):
            if a in ident and b in ident: continue
            small = a if a not in ident else b
            if a in soft and b in soft:
                k = "soft-right" if soft[a][0] == soft[b][0] else "soft-wrong"
            else:
                k = "a piece without any matched row"
            tot[k] += 1; rows[k] += min(nrows[a], nrows[b])
    print(os.path.basename(path), dict(tot), "| rows of the smaller piece:", dict(rows))
