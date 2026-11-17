"""IN-SAMPLE exploration (all 13 clips, labels only through score.evaluate): mutual best + margin with the
new prototypes and the wider candidate set. Not a headline result."""
import os
import sys

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score  # noqa: E402

import decide  # noqa: E402
from features import pair_features  # noqa: E402


def show(tag, method):
    p = score.evaluate(method)["pooled"]
    print(f"{tag:64s} right {p['right']:3d} wrong {p['wrong']:2d} lost {p['lost']:3d} | links {p['links']:3d} r/w/u {p['links_right']:3d}/{p['links_wrong']:2d}/{p['links_unknown']:2d} "
          f"prec {p['link_precision'] if p['link_precision'] is not None else float('nan'):.3f} impure {p['impure_row_share']:.4f}")
    return p


if __name__ == "__main__":
    F = {s: pair_features(s) for s in data.DEV_CLIPS}
    show("baseline mutual (appearance_match candidates)", score.baseline_links("mutual"))
    for kind in ("raw", "tc", "w"):
        for mk in ("marg", "rival"):
            for t_abs in (-1.0, 0.3, 0.5, 0.6, 0.7):
                for margin in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
                    if kind == "raw" and t_abs not in (-1.0,):
                        continue
                    m = {s: {"links": decide.mutual(F[s], kind, t_abs, margin, margin_kind=mk)} for s in F}
                    show(f"mutual {kind} {mk} t_abs={t_abs} margin={margin}", m)
