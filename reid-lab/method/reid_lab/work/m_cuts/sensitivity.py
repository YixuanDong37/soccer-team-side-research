"""Sensitivity of the held-out result to the fixed settings of features.py (not tuning: the
defaults stay). Each variant re-runs the whole leave-one-clip-out evaluation in a fresh process.

    python -I sensitivity.py
"""
import json
import os
import subprocess
import sys

HERE = "C:/Research/plugins/tools/reid_lab/work/m_cuts/"
PY = sys.executable
CODE = r'''
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import loco, truth as T
from reid_lab import data
tab = loco.table()
n_sw = sum(len(T.switches(s)) for s in data.DEV_CLIPS)
res, pars, par, ll = loco.loco(loco.MODELS["lr(cos,mm,lgap,ld)"], tab, (0.7,))
held, ins = res[0.7]
r = loco.evaluate_cuts(held, n_sw)
print("RESULT " + json.dumps({"logloss": ll, **{k: r[k] for k in ("cuts", "T", "F", "t", "f", "U", "cut_precision", "switches_hit")},
      "split": r["split+cuts"], "mutual": r["mutual+cuts(appearance)"]}))
'''
variants = [{}, {"K_WIN": 10}, {"K_WIN": 40}, {"NMS_ROWS": 5}, {"NMS_ROWS": 20}, {"GAP_MIN": 1}, {"GAP_MIN": 3}, {"ALT_FRAMES": 125}, {"ALT_FRAMES": 500}]
out = []
for v in variants:
    env = dict(os.environ, **{"M_CUTS_" + k: str(x) for k, x in v.items()})
    p = subprocess.run([PY, "-I", "-c", CODE], env=env, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        print(v, "FAILED", p.stderr[-500:])
        continue
    r = json.loads(line[0][7:])
    out.append({"variant": v or "default", **r})
    s, m = r["split"], r["mutual"]
    print(f"{str(v or 'default'):22s} log-loss {r['logloss']:.4f} cuts {r['cuts']} T {r['T']} F {r['F']} t {r['t']} f {r['f']} U {r['U']} prec {r['cut_precision']:.3f} sw {r['switches_hit']} | "
          f"split+cuts {s['right']}/{s['wrong']}/{s['lost']} impure {100*s['impure_row_share']:.2f}% | mutual+cuts {m['right']}/{m['wrong']}/{m['lost']} impure {100*m['impure_row_share']:.2f}%", flush=True)
json.dump(out, open(HERE + "sensitivity.json", "w"), indent=1)
