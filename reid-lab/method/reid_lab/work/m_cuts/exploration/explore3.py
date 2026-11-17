import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
for seq in data.DEV_CLIPS:
    a = data.appearance(seq)
    print(seq, "imp same team", {k: a["impostor_same_team"][k] for k in ("pairs","50","90","95","99")}, "diff team", a["different_team_covisible"], "genuine all", a["genuine_all_gaps"])
    print("    genuine halves", a["genuine_halves"])
print(a["params"])
print(a["reentry_pairs"][:3])
print(a["candidate_pairs"][:5], len(a["candidate_pairs"]))
