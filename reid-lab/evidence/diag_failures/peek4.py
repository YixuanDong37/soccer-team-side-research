import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
def show(seq, piece, lo=None, hi=None):
    c = ctx(seq)
    rows = c.piece_rows[piece]
    t = c.t.iloc[rows][["frame","x1","y1","x2","y2","conf","team","role"]].copy()
    t["ref"] = c.ref_of_predrow[rows]; t["iou"] = c.iou_of_predrow[rows].round(2); t["clean"] = c.clean[rows]
    if lo is not None: t = t[(t.frame >= lo) & (t.frame <= hi)]
    print(seq, piece, {k: c.P[piece][k] for k in ("first","last","rows","matched","ident_any","ident_n","second","second_n","n_clean","team","vteam")})
    print(t.round(0).to_string())
def person(seq, rid, lo, hi):
    c = ctx(seq)
    g = c.people_by[rid]; g = g[(g.frame >= lo) & (g.frame <= hi)]
    m = c.pairs[(c.pairs.ref_id == rid) & (c.pairs.frame >= lo) & (c.pairs.frame <= hi)][["frame","track_id","iou"]]
    g = g.merge(m, on="frame", how="left")
    print(seq, "person", rid, c.att[rid]); print((g[["frame","x1","y1","x2","y2","track_id","iou"]]).round(2).to_string())
show("SNGS-162", 53, 620, 700)
person("SNGS-162", 1, 630, 660)
person("SNGS-162", 18, 630, 680)
person("SNGS-162", 10, 630, 680)
