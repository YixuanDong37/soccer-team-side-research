"""Sensitivity of the position linker to its hand-set constants (they were set once, a priori,
and not tuned). Each row changes ONE constant, refits the weights in leave-one-clip-out and
reports the held-out log-likelihood and the pooled held-out counts at two fixed thresholds.

    python -I sensitivity.py            (about 2 minutes per row)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data, score  # noqa: E402
import features  # noqa: E402
import fitdata  # noqa: E402
import geom  # noqa: E402
import linker  # noqa: E402
from run_position_linker import CONFIG, EMIT, FLOOR, MODE  # noqa: E402

HERE = Path(__file__).resolve().parent
SETTINGS = [
    ("default", None, None, None),
    ("MAX_SPEED 6", features, "MAX_SPEED", 6.0),
    ("MAX_SPEED 10", features, "MAX_SPEED", 10.0),
    ("VIEW_MARGIN_PX 0", features, "VIEW_MARGIN_PX", 0.0),
    ("VIEW_MARGIN_PX 50", features, "VIEW_MARGIN_PX", 50.0),
    ("ADV_KERNEL_M 8", features, "ADV_KERNEL_M", 8.0),
    ("ADV_KERNEL_M 30", features, "ADV_KERNEL_M", 30.0),
    ("FLOW_HALF 3", features, "FLOW_HALF", 3),
    ("FLOW_HALF 12", features, "FLOW_HALF", 12),
    ("END_WINDOW 12", geom, "END_WINDOW", 12),
    ("END_WINDOW 50", geom, "END_WINDOW", 50),
    ("CV_HORIZON_S 0.5", features, "CV_HORIZON_S", 0.5),
    ("CV_HORIZON_S 2", features, "CV_HORIZON_S", 2.0),
    ("START_RADIUS 5", features, "START_RADIUS", 5.0),
]


def clear():
    for fn in (geom.piece_ends, features.flow, features.view_steps, features.hidden_path_costs, features.carried_inview,
               features._player_velocities, features.advected, features.advected_inview, features.census,
               features._pair_features, fitdata.labelled_pairs):
        fn.cache_clear()


def run():
    lp = linker.loco_params(**CONFIG)
    tot = n = tot_e = n_e = 0.0
    for seq, pr in lp.items():
        a, b = linker.heldout_nll(pr, seq)
        tot, n = tot + a, n + b
        a, b = linker.heldout_nll(pr, seq, edge_only=True)
        tot_e, n_e = tot_e + a, n_e + b
    seqs = {s: linker.peel_sequence(features.pair_features(s), pr, MODE, FLOOR) for s, pr in lp.items()}
    res = {"nll": tot / n, "nll_border": tot_e / n_e}
    for t in (0.5, 0.4):
        res[f"t{t}"] = score.evaluate({s: {"links": linker.cut_sequence(q, t, EMIT)} for s, q in seqs.items()})["pooled"]
    return res


if __name__ == "__main__":
    only = sys.argv[1:]
    out = {}
    for name, module, attr, value in SETTINGS:
        if only and name.split()[0] not in only and name != "default":
            continue
        old = getattr(module, attr) if module else None
        if module:
            setattr(module, attr, value)
        clear()
        r = run()
        if module:
            setattr(module, attr, old)
        out[name] = r
        a, b = r["t0.5"], r["t0.4"]
        print(f"{name:20s} NLL {r['nll']:.4f} border {r['nll_border']:.4f} | theta 0.5: R{a['right']} W{a['wrong']} L{a['lost']} links r{a['links_right']} w{a['links_wrong']} u{a['links_unknown']}"
              f" | theta 0.4: R{b['right']} W{b['wrong']} L{b['lost']} links r{b['links_right']} w{b['links_wrong']} u{b['links_unknown']}", flush=True)
        (HERE / "sensitivity.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    clear()
