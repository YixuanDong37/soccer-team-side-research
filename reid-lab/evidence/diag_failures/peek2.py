import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
seq = "SNGS-162"
r = data.refine(seq)
sa = r["stitch_audit"]
for k, v in sa.items():
    s = json.dumps(v)
    print(k, s[:1200])
    print()
ev = json.loads((data.RUNS/seq/"reentry-eval.json").read_text())
for k in ("links", "link_detail", "plugin_reentry_joins", "stitch_audit_counts", "abstentions_listed_all", "contaminated", "outcomes", "undetected_returns"):
    print(k, json.dumps(ev[k])[:1500]); print()
a = data.appearance(seq)
print(json.dumps(a["edge_board"])[:3000])
print(json.dumps(a["reentry_pairs"])[:1500])
