# reid-lab：认人实验台（研究版）

这个文件夹是“认人”方法的研究版完整交付。认人指的是：球员走出画面再回来时，把他认回原来那个人。

- 最终方法叫 `fuse_r1`，分三档：稳妥档、均衡档、多认档。
- 这里是**研究版**：能重新拟合参数、做“留一段不看”的评估、做各种诊断。
- **产品版**已经另外交付：gbf 仓库里的 `gbf.relink`（fork 的 `relink` 分支，提交 9a70f90 和 22754d8）。它用的是这里在全部 13 段上拟合好的冻结参数，只管“应用”，不管拟合。两者的关系见 [docs/方法说明.md](docs/方法说明.md) 第 8 节。

English summary: [README_en.md](README_en.md)。

## 一眼看结果

数据是 SN-GSR-2025 训练集里的 13 段片段（来自 3 场比赛），每段 30 秒。下面每格是“认对 / 认成别人 / 当成新人”。

| 做法 | 原来的跟踪设置（137 次能打分的回来） | 负责人的跟踪设置（156 次） |
|---|---|---|
| gbf 现在（出画面处切开，不拼） | 0 / 4 / 133 | 0 / 3 / 153 |
| 插件原样（跟踪器自己的编号） | 30 / 14 / 93 | 7 / 7 / 142 |
| 稳妥档 | 67 / 3 / 67 | 原参数照搬 78 / 10 / 68；重新拟合 25 / 7 / 124 |
| 均衡档 | 85 / 5 / 47 | 原参数照搬 104 / 11 / 41；重新拟合 59 / 12 / 85 |
| 多认档 | 106 / 7 / 24 | 原参数照搬 127 / 14 / 15；重新拟合 116 / 13 / 27 |

- 这些都是“完全留出”的成绩：每一段用到的参数和门槛，都是用另外 12 段定出来的。
- 认错集中在一段：SNGS-162。去掉它，原设置下稳妥档是 64 / 1 / 54。这一段的标注本身可疑，还没有人看过画面。
- 全部数字、每个数字出自哪个文件，见 [docs/结果汇总.md](docs/结果汇总.md)。要打的折扣见 [docs/局限.md](docs/局限.md)。

## 文件夹里有什么

| 位置 | 内容 |
|---|---|
| [method/](method/) | 实验台本体。`reid_lab/` 里是读数据的 `data.py`（不读标注）、打分的 `score.py`（读标注）、核对打分的 `check_score.py`，以及最终方法和它的三个部件：`work/m_cuts`（切开）、`work/m_appear`（外观）、`work/m_space`（位置）、`work/fuse_r1`（合成）。 |
| [pipeline/](pipeline/) | 从 SN-GSR 训练片段生成实验台输入的全套脚本，含两套跟踪设置、补丁和片段选择表。 |
| [evidence/](evidence/) | 历史和核对材料：第一轮合成 `fuse`、没用上的部件 `m_keep`、两组诊断、三位检查员各两轮（`v_leak`、`v_repro`、`v_robust`）、批评者 `critic`。 |
| [results/](results/) | 负责人设置下的两种结果（`out_T` 原参数照搬、`out_R` 重新拟合）、重新拟合得到的参数、日志，以及汇总表 `headline-numbers.csv`。 |
| [docs/](docs/) | 中文文档（见下）。 |
| [relocate.py](relocate.py) | 改路径的工具。代码里写死了原机器的绝对路径，用之前先跑它。 |
| `manifest-sha256.txt` | 每个文件的 sha256。 |

## 文档

1. [docs/方法说明.md](docs/方法说明.md)：方法是什么、每一步、拟合了哪些参数、事先写下的规则 B、留一段不看、和 `gbf.relink` 的关系。
2. [docs/运行手册.md](docs/运行手册.md)：一步一步的命令和预计时间。
3. [docs/结果汇总.md](docs/结果汇总.md)：每个主要数字和它的来源文件。
4. [docs/局限.md](docs/局限.md)：哪些地方要打折扣。
5. [docs/文件清单.md](docs/文件清单.md)：每个文件夹装了什么，哪些没带上、为什么。
6. [docs/许可与数据.md](docs/许可与数据.md)：第三方许可和数据使用规则。

## 最快的上手方式

在 Git Bash 里，进入本文件夹：

```bash
# 1. 把路径改成你机器上的位置（只改本文件夹里的文件）
python relocate.py --runs D:/data/gsr-runs --runs-leadcfg D:/data/gsr-runs-leadcfg \
    --gsr-train D:/data/sn-gsr-2025/train --gbf-repo D:/code/gbf --sngs D:/code/sn-gamestate
python relocate.py --check        # 应该输出 "no absolute path outside the configured locations"

# 2. 核对打分工具（约 30 秒到 1 分钟；需要 13 段的运行结果和标注）
PY=D:/code/gbf/.venv/Scripts/python.exe
(cd method/reid_lab && $PY -I check_score.py)     # 最后一行应为 "problems: none"

# 3. 用带来的参数重新生成结果文件，并打分（共约 8 分钟）
cd method/reid_lab/work/fuse_r1
$PY -I -B prep.py 5 && $PY -I -B method.py
$PY -I ../../score.py METHOD_strict.json OUT_check.json   # 应为 67 / 3 / 67
```

上面的 `D:/...` 只是例子。运行结果（`gsr-runs`）和 SN-GSR 数据不在这个文件夹里，要按 [docs/运行手册.md](docs/运行手册.md) 第 2 节自己生成。

## 没有带上的东西

- 运行数据：视频、图片、parquet、npz、pklz、缓存（pkl）。可以用 `pipeline/` 重新生成。
- SN-GSR 数据和标注文件。
- “一行一个回来 / 一个片段 / 一对拼接”的明细表，以及逐条列出标注人物编号的日志。原因：SN-GSR 的使用规则是只发数字、不发画面，明细表等于标注的摘录。汇总数字都在。清单见 [docs/left-out.csv](docs/left-out.csv)。
- 所有 HTML 页面，包括看片页 `viewer.html`（生成看片数据的脚本带了）。
- 负责人的私有片段：不在这次交付里，也没有任何由它得出的数据。

## 许可与数据权利（简述）

- 实验台代码和文档随 gbf 仓库的许可（MIT）。
- `pipeline/` 的脚本要在 sn-gamestate（GPL-3.0）等第三方环境里运行；`pipeline/patches/` 是对第三方代码的小补丁。
- 所有拟合参数都只在 SN-GSR 片段上拟合，属于研究用途；用于商业构建前需要复核。
- 详情见 [docs/许可与数据.md](docs/许可与数据.md)。这些是工程判断，不是法律意见。
