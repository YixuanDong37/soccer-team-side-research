# reid-lab: the re-identification lab (research version)

This folder is the research version of the method that recognises a football player who leaves
the picture and comes back. The full documentation is in Chinese: start at [README.md](README.md).

## What it is

- The final method is `fuse_r1`: cut tracks where the tracker moved onto another person, then
  join the pieces of one person by appearance and position. It has three operating points:
  strict, balanced and recall.
- This folder holds what the production code does not: fitting, the leave-one-clip-out
  evaluation, the pre-registered rule for the operating points, diagnosis, and the reports of
  three verification rounds.
- The production version is `gbf.relink` in the gbf repository (branch `relink`, commits 9a70f90
  and 22754d8). It applies the parameters that were fitted here on all 13 clips and fits nothing.

## Headline numbers

13 clips of the SN-GSR-2025 train split, from 3 matches. Each cell is right / wrong / lost.

| | earlier tracker settings (137 scored returns) | lead's tracker settings (156) |
|---|---|---|
| gbf now (split, no re-linking) | 0 / 4 / 133 | 0 / 3 / 153 |
| the plugin's own track ids | 30 / 14 / 93 | 7 / 7 / 142 |
| strict | 67 / 3 / 67 | earlier fits unchanged 78 / 10 / 68; refitted 25 / 7 / 124 |
| balanced | 85 / 5 / 47 | earlier fits unchanged 104 / 11 / 41; refitted 59 / 12 / 85 |
| recall | 106 / 7 / 24 | earlier fits unchanged 127 / 14 / 15; refitted 116 / 13 / 27 |

These rows are fully held out: every clip was processed with weights and a setting fitted without
it. Most wrong returns sit in one clip, SNGS-162, whose annotation is suspect; nobody has looked at
its frames. Every number with its source file: [docs/结果汇总.md](docs/结果汇总.md) and
[results/headline-numbers.csv](results/headline-numbers.csv).

## Layout

| path | content |
|---|---|
| `method/reid_lab/` | the lab: `data.py` (label-free loaders), `score.py` (the scorer, reads labels), `check_score.py`, and `work/{m_cuts,m_appear,m_space,fuse_r1}` |
| `pipeline/` | the scripts that produce the lab's inputs from SN-GSR train clips, for both tracker settings; the patches; the clip selection |
| `evidence/` | history and verification: round 0 (`fuse`), an unused component (`m_keep`), two diagnosis folders, three verifiers in two rounds, the critic |
| `results/` | the lead-settings results (`out_T` earlier fits unchanged, `out_R` refitted), the refitted parameters, logs, the number summary |
| `docs/` | the Chinese documentation and the list of what was left out |
| `relocate.py` | rewrites the hard-coded absolute paths; run it first |

## Quick start

```bash
python relocate.py --runs D:/data/gsr-runs --runs-leadcfg D:/data/gsr-runs-leadcfg \
    --gsr-train D:/data/sn-gsr-2025/train --gbf-repo D:/code/gbf --sngs D:/code/sn-gamestate
python relocate.py --check
PY=D:/code/gbf/.venv/Scripts/python.exe
(cd method/reid_lab && $PY -I check_score.py)          # last line: problems: none
cd method/reid_lab/work/fuse_r1 && $PY -I -B prep.py 5 && $PY -I -B method.py
```

The paths above are examples. `python relocate.py --help` and the text at the top of
`relocate.py` describe every option, including `--dest` (relocate a copy) and `make-lab` (build
the lab copy that reads the lead-settings runs). Step by step: [docs/运行手册.md](docs/运行手册.md).

## Not included

Run data (videos, images, parquet, npz, pklz, caches), the SN-GSR data and its labels, per-item
tables and listings that quote annotation ids (SN-GSR: numbers only), every HTML page including the
run viewer page, and the lead's private clip. The list with reasons: [docs/left-out.csv](docs/left-out.csv).

## Licences and data

Lab code and docs follow the gbf repository (MIT). The pipeline scripts run inside third-party
environments (sn-gamestate GPL-3.0, ultralytics AGPL-3.0, prtreid and bpbreid Hippocratic License
3.0, NBJW GPL-2.0-only); none of that code is shipped, only small patches. Every fitted parameter
was fitted on SN-GSR clips: research use, not for commercial builds without review. Details:
[docs/许可与数据.md](docs/许可与数据.md). Engineering judgement, not legal advice.

## Limits in one paragraph

13 clips from 3 matches; the appearance model (PRTReid) was probably trained on SoccerNet data;
the wrong returns concentrate in one clip with suspect labels; the operating points depend on the
selection rule; the lead-settings runs used 720p video and PyTorch, not the lead's runtime; shot
cuts are not handled in the lab. See [docs/局限.md](docs/局限.md).
