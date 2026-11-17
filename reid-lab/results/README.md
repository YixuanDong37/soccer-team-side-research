# results：结果文件

结果的中文汇总和每个数字的出处在 [../docs/结果汇总.md](../docs/结果汇总.md)。这个文件夹里是：

| 位置 | 内容 |
|---|---|
| [headline-numbers.csv](headline-numbers.csv)、[headline-numbers.json](headline-numbers.json) | 主要数字的汇总表，每一行写明来源文件。打包时由结果文件生成，不读标注 |
| [inventory.csv](inventory.csv) | 包里每个文件的路径、字节数、sha256 |
| [leadcfg-settings/out_T/](leadcfg-settings/out_T/) | 负责人的跟踪设置，原参数照搬（T）：方法文件、打分文件、打分日志 |
| [leadcfg-settings/out_R/](leadcfg-settings/out_R/) | 负责人的跟踪设置，重新拟合（R）：同上 |
| [leadcfg-settings/refit_R/](leadcfg-settings/refit_R/) | R 拟合出的参数和汇总表 |
| [leadcfg-settings/eval-logs/](leadcfg-settings/eval-logs/) | 两次评估的日志；`baselines.json` 是对照做法的成绩 |
| [leadcfg-settings/validation/](leadcfg-settings/validation/) | 副本可信度的核对 |

原来的跟踪设置下的结果不在这里重复存放，它们在 [../method/reid_lab/work/fuse_r1/](../method/reid_lab/work/fuse_r1/)（`METHOD_*.json`、`OUT_*.json`、`out/`）。

English: the number summary (each row names its source file), the file inventory, and the results on the runs made with the lead's tracker settings: `out_T` (earlier fits applied unchanged), `out_R` (refitted), `refit_R` (the refitted parameters). The results on the earlier-settings runs live next to the method, in `method/reid_lab/work/fuse_r1/`. The Chinese summary is `docs/结果汇总.md`.
