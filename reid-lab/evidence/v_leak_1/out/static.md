# v_leak round 1: static check of fuse_r1's apply path

## 1. Imports of the apply path (ast)

| file | imports | label import | where | annotation path in a string literal |
|---|---|---|---|---|
| fuse_r1/prep.py | __future__, concurrent, hashlib, importlib, json, numpy, pandas, pathlib, pickle, reid_lab, reid_lab.data, sys, time | no | - | - |
| fuse_r1/evidence.py | __future__, pandas, prep, sys | no | - | - |
| fuse_r1/linker.py | __future__, numpy, prep, sys | no | - | - |
| fuse_r1/linker2.py | __future__, linker, numpy, prep, sys | no | - | - |
| fuse_r1/method.py | __future__, evidence, json, linker2, numpy, pairmodel, pathlib, prep, reid_lab, reid_lab.data, sys | no | - | - |
| fuse_r1/pairmodel.py | __future__, numpy, pandas, scipy | no | - | - |
| fuse_r1/selfcheck.py | __future__, linker, numpy, pathlib, prep, reid_lab, reid_lab.data, sys | no | - | - |
| m_cuts/features.py | __future__, functools, json, numpy, os, pandas, reid_lab, reid_lab.data, sys | no | - | - |
| m_appear/features.py | agglo, core, functools, numpy, os, pandas, reid_lab, reid_lab.data, sys, time, warnings | no | - | - |
| m_appear/core.py | functools, numpy, pandas, reid_lab, reid_lab.data, sys | no | - | - |
| m_appear/agglo.py | core, features, numpy, os, sys | no | - | - |
| m_space/features.py | __future__, functools, geom, numpy, pandas, reid_lab, reid_lab.data, scipy, sys, time | no | - | - |
| m_space/geom.py | __future__, functools, numpy, pandas, reid_lab, reid_lab.data, sys | no | - | - |
| m_space/linker.py | __future__, features, fitdata, geom, numpy, pandas, reid_lab, reid_lab.data, scipy, sys | yes | fitdata line 336 in def fit; fitdata line 383 in def heldout_nll | - |

## 2. Fitted files read per fold against round 0's blindfolded re-fits

| fold | m_cuts weights: max abs diff | m_space weights: max abs diff | fold in m_space fit_clips | cuts in fuse_r1/cache/cuts.json equal to the blind re-fit's (13 clips) |
|---|---|---|---|---|
| SNGS-107 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-163 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-106 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-097 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-067 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-100 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-066 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-161 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-169 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-162 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-151 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-102 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |
| SNGS-112 | 0.0e+00 | 0.0e+00 | False | 13 of 13 |

Largest difference of any weight: 0.0e+00; all cut lists equal: True.

## 3. Modification times (local time, 2026-11-15)

| file | modified |
|---|---|
| m_cuts/features.py | 04:18:06 |
| m_cuts/loco_params.json | 04:15:43 |
| m_cuts/final_params.json | 04:15:43 |
| m_appear/features.py | 04:08:15 |
| m_appear/core.py | 04:07:13 |
| m_space/features.py | 04:18:19 |
| m_space/geom.py | 03:32:21 |
| m_space/linker.py | 04:32:04 |
| m_space/params_all13.json | 04:31:03 |
| m_space/params_cache/excl_162.json | 04:28:05 |
| v_leak_0/refit/cuts_SNGS-162.json | 06:43:52 |
| v_leak_0/refit/space_SNGS-162.json | 06:46:59 |
| fuse_r1/PREREG.txt | 07:28:49 |
| fuse_r1/prep.py | 07:29:10 |
| fuse_r1/evidence.py | 07:32:43 |
| fuse_r1/linker.py | 07:26:24 |
| fuse_r1/linker2.py | 07:32:13 |
| fuse_r1/pairmodel.py | 07:26:24 |
| fuse_r1/rules.py | 07:34:38 |
| fuse_r1/fit_loco.py | 07:35:37 |
| fuse_r1/method.py | 07:40:49 |
| fuse_r1/logs/prep.log | 07:29:35 |
| fuse_r1/logs/check_linker2.log | 07:33:56 |
| fuse_r1/logs/fit_loco.log | 07:45:15 |
| fuse_r1/cache/loco_raw.pkl | 08:27:08 |
| fuse_r1/out/run1/tradeoff.md | 08:19:08 |
| fuse_r1/METHOD_strict.json | 07:46:25 |
| fuse_r1/out/cli_scores.txt | 07:49:07 |
| fuse_r1/params_loco.json | 08:27:09 |
| fuse_r1/logs/fit_loco_run2.log | 08:27:09 |
| fuse_r1/README.md | 08:27:57 |

PREREG.txt modified 07:28:49; rules.py 07:34:38; fit_loco.py 07:35:37; oldest log of a label-using fuse_r1 script: 07:45:15.

