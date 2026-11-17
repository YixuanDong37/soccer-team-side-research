"""Position scores of piece pairs for the fusion step: the fitted position model applied to
features.pair_features(seq).

    import sys; sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
    import posscore
    df = posscore.position_scores("SNGS-107")                      # weights never saw SNGS-107
    df = posscore.position_scores("SNGS-107", exclude=("SNGS-107", "SNGS-163"))   # nor SNGS-163

Label hygiene. The weights are fitted with labels (linker.fit) on the dev clips NOT listed in
``exclude``; by default exclude = (seq,), so the scores of a clip never rest on its own labels
(out-of-fold scores). A fusion model that is itself fitted in a leave-one-clip-out loop should,
for its fit clips, pass exclude = (fit clip, held-out clip): then nothing fitted ever saw the
held-out clip. Fits are cached in params_cache/ (one JSON per excluded set).

Columns: a, b, pos_adm (the pair passes the hard rules: within reach, same role, teams not
known to differ; the other columns are NaN / 0 when it does not), pos_z (log evidence w . x),
pos_p_in = P(a | b) among the earlier candidates of b incl. "nobody", pos_p_out = P(b | a),
pos_sink (doubly normalised assignment score), pos_rank_in (1 = best earlier candidate of b by
pos_z), pos_rank_out (1 = best later candidate of a), pos_n_in / pos_n_out (number of
admissible candidates), pos_kind (2 = a ends at an image border and b starts at one, 1 = one of
them, 0 = neither).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import features  # noqa: E402
import linker  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "params_cache"
CONFIG = dict(resid="r_adv_cv", pair=("lu", "ls", "tu", "ee", "ex", "fa", "ba"), ctx=("one", "edge", "time"), edge_weight=3.0)
"""The delivered position model (the same as run_position_linker.CONFIG)."""


def params_excluding(exclude) -> dict:
    """Weights fitted on the dev clips not in ``exclude`` (cached on disk). LABELS of the
    fit clips are read here (linker.fit); the excluded clips are never touched."""
    exclude = tuple(sorted(set(exclude)))
    for s in exclude:
        if s not in data.DEV_CLIPS:
            raise ValueError(f"{s} is not a dev clip")
    CACHE.mkdir(exist_ok=True)
    path = CACHE / ("excl_" + "_".join(s.split("-")[1] for s in exclude) + ".json")
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        same = all(cached.get(k) == (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items())
        if same:
            return cached
    fit_clips = tuple(s for s in data.DEV_CLIPS if s not in exclude)
    params = linker.fit(fit_clips, **CONFIG)
    params["fit_clips"] = list(fit_clips)
    path.write_text(json.dumps(params, indent=1) + "\n", encoding="utf-8")
    return params


def position_scores(seq: str, exclude=None, params: dict | None = None) -> pd.DataFrame:
    if params is None:
        params = params_excluding((seq,) if exclude is None else exclude)
    p = features.pair_features(seq)
    q = linker.posteriors(p, params, "sinkhorn")
    q["pos_rank_in"] = q.groupby("b").z.rank(ascending=False, method="first")
    q["pos_rank_out"] = q.groupby("a").z.rank(ascending=False, method="first")
    q["pos_n_in"] = q.groupby("b").z.transform("size")
    q["pos_n_out"] = q.groupby("a").z.transform("size")
    q["pos_kind"] = linker.kind(q)
    q = q.rename(columns={"z": "pos_z", "p_in": "pos_p_in", "p_out": "pos_p_out", "score": "pos_sink"})
    cols = ["a", "b", "pos_z", "pos_p_in", "pos_p_out", "pos_sink", "pos_rank_in", "pos_rank_out", "pos_n_in", "pos_n_out", "pos_kind"]
    out = p[["a", "b"]].merge(q[cols], on=["a", "b"], how="left")
    out.insert(2, "pos_adm", out.pos_z.notna())
    for c in ("pos_p_in", "pos_p_out", "pos_sink"):
        out[c] = out[c].fillna(0.0)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for seq in data.DEV_CLIPS:
        s = position_scores(seq)
        print(seq, len(s), "pairs,", int(s.pos_adm.sum()), "admissible; pos_sink >= 0.5:", int((s.pos_sink >= 0.5).sum()))
    print(s[s.pos_adm].sort_values("pos_sink", ascending=False).head(8).round(3).to_string(index=False))
