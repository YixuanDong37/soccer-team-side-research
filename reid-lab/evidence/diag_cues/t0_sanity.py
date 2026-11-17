"""Sanity: the return counts reproduce the lab (167 returns, 137 scored); matching summary."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import common as C  # noqa: E402

pd.set_option("display.width", 250)
tot = []
for seq in C.CLIPS:
    r = C.returns(seq)
    m = C.matched(seq)
    t = C.data.tracks(seq)
    pi = C.piece_ident(seq)
    tot.append({"seq": seq, "returns": len(r), "scored": int(r.scored.sum()), "tracks_rows": len(t), "matched": len(m),
                "ann_rows": len(C.ann(seq)), "pieces": len(pi), "pieces_ident": int(pi.ref_id.notna().sum()),
                "persons": C.ann(seq).ref_id.nunique(), "same_piece_ab": int(((r.piece_a == r.piece_b) & r.scored).sum()),
                "untracked_rows": int((t.track_id < 0).sum())})  # fmt: skip
df = pd.DataFrame(tot)
print(df.to_string())
print(df.sum(numeric_only=True).to_dict())
pa = pd.concat([C.person_attrs(s).assign(seq=s) for s in C.CLIPS])
print("persons with >1 team value:", int((pa.team_values > 1).sum()), "; >1 jersey value:", int((pa.jersey_values > 1).sum()),
      "; >1 role:", int((pa.role_values > 1).sum()))  # fmt: skip
print(pa.groupby("role").size().to_dict())
print(pa[(pa.team_values > 1) | (pa.jersey_values > 1) | (pa.role_values > 1)].to_string())
df.to_csv(C.OUT / "t0_sanity.csv", index=False)
