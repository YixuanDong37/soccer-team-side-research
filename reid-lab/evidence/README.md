# evidence：历史和核对材料

这里的 11 个文件夹是得到最终方法的过程记录，不是方法的一部分。每个文件夹对应原来 `reid_lab/work/` 下的同名文件夹，内部结构不变；各自的 `README.md` 是当时的英文报告。

| 文件夹 | 是什么 |
|---|---|
| [fuse/](fuse/) | 第一轮合成（没有分长短），被 `fuse_r1` 取代 |
| [m_keep/](m_keep/) | 试过但最终没用的部件：插件自己的拼接该保留还是断开 |
| [diag_cues/](diag_cues/) | 诊断：各种线索（队别、位置、相机、球衣号、外观）有多少信息量 |
| [diag_failures/](diag_failures/) | 诊断：167 次回来为什么认不出，理想上限是多少 |
| [critic/](critic/) | 批评者：方法还缺什么 |
| [v_leak_0/](v_leak_0/)、[v_leak_1/](v_leak_1/) | 检查员“偷看检查”，两轮 |
| [v_repro_0/](v_repro_0/)、[v_repro_1/](v_repro_1/) | 检查员“复现检查”，两轮 |
| [v_robust_0/](v_robust_0/)、[v_robust_1/](v_robust_1/) | 检查员“稳健性检查”，两轮 |

第 0 轮检查的是 `fuse`，第 1 轮检查的是最终方法 `fuse_r1`。

要知道的：

- 这里的材料主要用来读。缓存没带，逐条的明细表和逐条列出标注人物的日志也没带（原因见 [../docs/文件清单.md](../docs/文件清单.md) 第 6 节），所以报告里提到的一部分输出文件在这里找不到，要重跑脚本才有。
- 报告里提到的 `report-*.html` 没带。
- 改完路径后有三个脚本不能直接运行，见 [../docs/运行手册.md](../docs/运行手册.md) 第 10 节。

English: history and verification material, one folder per former `reid_lab/work/<name>`. Round 0 checked `fuse`, round 1 the final method `fuse_r1`. Caches, per-item tables and listings that quote annotation ids are not included, so some output files named in the reports are missing here; the scripts that write them are included.
