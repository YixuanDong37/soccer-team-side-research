"""DIAGNOSIS: the links of the held-out recall point that score.py judges wrong, with the two
annotated persons; next to diag_failures' lists of label suspects (numbers only)."""
import json, sys
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_0")
from reid_lab import data
import fuse_labels as fitlabels, prep, evidence
HERE = "C:/Research/plugins/tools/reid_lab/work/v_repro_0/"
for p in ("strict", "recall"):
    M = json.load(open(HERE + f"METHOD_{p}.json", encoding="utf-8"))["clips"]
    print(p)
    for s in data.DEV_CLIPS:
        cuts = prep.norm_cuts(M[s]["cuts"])
        ident = fitlabels.identities(s, cuts)
        e = evidence.pieces(s, cuts).set_index("piece")
        for a, b in M[s]["links"]:
            ia, ib = ident.get(a), ident.get(b)
            if ia and ib and ia[1] >= 5 and ib[1] >= 5 and ia[0] != ib[0]:
                print(f"  {s} {a}>{b}: persons {ia[0]} ({ia[1]} rows) > {ib[0]} ({ib[1]} rows); frames {e['first'][a]}-{e['last'][a]} > {e['first'][b]}-{e['last'][b]}")
