"""DIAGNOSIS: threshold sweep on single features / simple scores, event level with NMS (diag_cand.csv)."""
import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 280); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 800)
df = pd.read_csv("C:/Research/plugins/tools/reid_lab/work/m_cuts/diag_cand.csv")
df["m_d"] = df.cos - df.alt_d
df["m_r"] = df.cos - df.alt_r
df["take"] = (df.d_cos - df.cos).where(df.d_iou >= 0.3)
NMS = 10


def nms(sel, score):
    """Greedy: best score first, drop candidates of the same piece within NMS rows of a kept one."""
    keep = []
    sel = sel.assign(_s=score.loc[sel.index]).sort_values("_s", ascending=False)
    taken = {}
    for r in sel.itertuples():
        key = (r.seq, r.piece)
        if any(abs(r.i - j) <= NMS for j in taken.get(key, [])):
            continue
        taken.setdefault(key, []).append(r.i)
        keep.append(r.Index)
    return df.loc[keep]


def report(name, score, ths):
    print(f"--- {name}")
    for th in ths:
        k = nms(df[score >= th], score)
        true = int((k.sw | (k["loc"] == True)).sum()); false = int((~k.sw & (k["loc"] == False)).sum()); unk = len(k) - true - false
        print(f"   th {th:6.3f}: cuts {len(k):4d} true {true:3d} (sw {int(k.sw.sum()):3d}) false {false:3d} unknown {unk:2d}  precision {true/max(1,true+false):.3f}  minor rows covered {int(k.minor.sum())}")


report("-cos", -df.cos, [-0.6, -0.65, -0.7, -0.75, -0.8, -0.825, -0.85, -0.875, -0.9, -0.925])
report("-cos_whole", -df.cos_whole, [-0.7, -0.75, -0.8, -0.85, -0.9])
report("-min(cos, cos_whole)", -np.minimum(df.cos, df.cos_whole), [-0.7, -0.75, -0.8, -0.85, -0.9])
report("-max(cos, cos_whole)", -np.maximum(df.cos, df.cos_whole), [-0.7, -0.75, -0.8, -0.85, -0.9])
report("-m_d", -df.m_d.fillna(1), [0.2, 0.15, 0.1, 0.05, 0.0])
report("-m_r", -df.m_r.fillna(1), [0.2, 0.15, 0.1, 0.05, 0.0])
report("-max(m_d,m_r)", -np.maximum(df.m_d.fillna(1), df.m_r.fillna(1)), [0.2, 0.15, 0.1, 0.05, 0.0])
# false cuts at cos < 0.85
k = nms(df[df.cos < 0.875], -df.cos)
bad = k[~k.sw & (k["loc"] == False)]
print(bad[["seq", "piece", "f0", "f1", "gap", "kind", "cos", "cos_whole", "nb", "na", "clean_b", "clean_a", "d_m", "dt", "alt_d", "alt_r", "m_d", "m_r", "role", "team", "box_h", "rows_before", "rows_after"]].to_string())
