"""IN-SAMPLE (fixed setting w / 0.5 / 0.25): effect of the candidate rules, one at a time."""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
from agglo import agglomerate
from core import ctx
from exp2_agglo import show
P = {s: ctx(s).P for s in data.DEV_CLIPS}
def run(tag, excl, **kw):
    show(tag, {s: {"links": agglomerate(s, exclude=excl(P[s]) if excl else None, **kw)} for s in data.DEV_CLIPS})
run("all non-referee pieces with a prototype (the method)", None)
run("only pieces with gbf's verified team", lambda p: set(p.piece[p.vteam.isna()].astype(int)))
run("only pieces with a team label (no nearest-mean team)", lambda p: set(p.piece[p.team.isna()].astype(int)))
run("only pieces with >= 5 clean rows", lambda p: set(p.piece[p.n_clean < 5].astype(int)))
run("only pieces labelled player (no goalkeepers)", lambda p: set(p.piece[p.role != "player"].astype(int)))
run("groups may share up to 3 frames (max_shared=3)", None, max_shared=3)
run("need a rival on both sides (need_rival)", None, need_rival=True)
run("no mutual-best requirement", None, mutual=False)
run("rival = shares >= 5 frames (covis=5)", None, covis=5)
run("linkage avg", None, linkage="avg")
run("linkage min", None, linkage="min")
run("kind tc at the same setting", None, kind="tc")
run("kind raw (no centring), t_abs 0.9, margin 0.05", None, kind="raw", t_abs=0.9, margin=0.05)
run("kind raw (no centring), t_abs 0.9, margin 0.1", None, kind="raw", t_abs=0.9, margin=0.1)
