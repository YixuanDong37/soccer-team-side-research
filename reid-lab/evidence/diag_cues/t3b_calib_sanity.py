"""(3, extra) Degenerate calibrations: frames whose camera is not a plausible main camera."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
import common as C  # noqa: E402

pd.set_option("display.width", 250)
rows = []
for seq in C.CLIPS:
    cal = C.data.calib(seq).sort_values("frame_index")
    t = C.data.tracks(seq)
    n_rows = t.groupby("frame").size()
    for r in cal.itertuples():
        f = np.nan if r.K is None else float(np.asarray(r.K, float)[0])
        c = [np.nan] * 3 if r.C is None else np.asarray(r.C, float)
        rows.append({"seq": seq, "frame": r.frame_index, "status": r.calib_status, "usable": bool(r.pitch_usable), "refined": bool(r.refined), "focal": f,
                     "cx": c[0], "cy": c[1], "cz": c[2], "rep_err": r.rep_err, "age": r.calib_age, "track_rows": int(n_rows.get(r.frame_index, 0))})  # fmt: skip
D = pd.DataFrame(rows)
D["odd_focal"] = ~D.focal.between(700, 8000)          # 720p: a main camera is roughly 1100-3500 px
D["odd_centre"] = ~(D.cy.between(35, 110) & D.cz.between(-40, -3) & D.cx.between(-40, 40))
print("frames:", len(D), "; status:", D.status.value_counts().to_dict(), "; pitch_usable:", int(D.usable.sum()))
print("frames with an implausible focal length (outside 700-8000 px at 720p):", int((D.odd_focal & D.focal.notna()).sum()), "; of those pitch_usable:", int((D.odd_focal & D.focal.notna() & D.usable).sum()))
print("frames with an implausible camera centre:", int((D.odd_centre & D.cy.notna()).sum()), "; of those pitch_usable:", int((D.odd_centre & D.cy.notna() & D.usable).sum()))
print(D.groupby("seq").agg(usable=("usable", "sum"), no_camera=("focal", lambda s: int(s.isna().sum())), odd_focal=("odd_focal", lambda s: int((s & D.loc[s.index, "focal"].notna()).sum())),
                           odd_focal_usable=("odd_focal", lambda s: int((s & D.loc[s.index, "focal"].notna() & D.loc[s.index, "usable"]).sum())),
                           odd_centre_usable=("odd_centre", lambda s: int((s & D.loc[s.index, "cy"].notna() & D.loc[s.index, "usable"]).sum())),
                           focal_p50=("focal", "median"), cam_y=("cy", "median"), cam_h=("cz", lambda s: -s.median()), centre_moves=("cy", lambda s: float(s.max() - s.min()))).round(1).to_string())  # fmt: skip
print("usable frames by status:", D.groupby("status").usable.agg(["size", "sum"]).to_dict())
print("frame-to-frame focal change on usable frames, |log ratio| quantiles:", C.qtable(np.abs(np.log(D.focal[D.usable & ~D.odd_focal]).groupby(D.seq).diff()), (0.5, 0.9, 0.99, 0.999)))
D.to_csv(C.OUT / "t3b_calib_frames.csv", index=False)
