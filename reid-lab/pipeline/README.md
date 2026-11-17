# pipeline：生成实验台输入的脚本

这里是从 SN-GSR-2025 训练片段生成实验台输入（每段的 `gbf/refine/*.parquet`、`reid-embeddings.npz` 等）的全部脚本，有两套跟踪设置：原来的设置（`run_gsr_*.sh`、`track_clip.sh`、`run_tracklab.py`）和负责人的设置（文件名带 `_leadcfg` 的）。

- 怎么运行：[../docs/运行手册.md](../docs/运行手册.md) 第 2 节。先运行 `../relocate.py`，脚本里写死的路径才会指向你的机器。
- 每个文件是什么：[../docs/文件清单.md](../docs/文件清单.md) 第 3 节。
- 第三方代码的版本、补丁和模型权重的校验值：[gsr/provenance-gsr.json](gsr/provenance-gsr.json)、[gsr/provenance-gsr-leadcfg.json](gsr/provenance-gsr-leadcfg.json)。
- 补丁的用法：[patches/README.md](patches/README.md)。

要知道的三点：

1. **只在 Windows 的 Git Bash 下用过。** 脚本用到 `.venv/Scripts/python.exe` 和 `powershell.exe`。
2. **看片页面没带。** `gsr/viewer_data.py` 生成看片数据，`gsr/serve_runs.py` 在本机提供文件；读这些数据的页面 `viewer.html` 是 HTML，不在这次交付里。
3. **许可。** 这些脚本是自己写的，但要在第三方环境里运行，会导入 tracklab（MIT 和 GPL-3.0）、sn-gamestate（GPL-3.0）、ultralytics（AGPL-3.0）等。它们作为独立进程运行，只通过文件和 gbf 交换结果，不要从 gbf 核心代码里导入它们。详见 [../docs/许可与数据.md](../docs/许可与数据.md)。

English: the scripts that turn SN-GSR-2025 train clips into the lab's inputs, for the earlier tracker settings and for the lead's. Run `../relocate.py` first. Git Bash on Windows only. The viewer page itself (HTML) is not included; only its data generator and the local file server are. The scripts run inside third-party environments (GPL-3.0, AGPL-3.0); keep them out of the import path of the MIT core.
