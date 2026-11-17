# method：实验台本体

这个文件夹就是代码里 `sys.path.insert(...)` 指向的地方：它下面的 `reid_lab/` 是一个 Python 包。

- `reid_lab/data.py`：读输入，不读标注。
- `reid_lab/score.py`：读标注并打分。
- `reid_lab/check_score.py`：核对打分工具。
- `reid_lab/work/fuse_r1/`：最终方法。它按路径导入三个部件：`work/m_cuts`（切开）、`work/m_appear`（外观）、`work/m_space`（位置）。

使用前先在上一级运行 `python relocate.py ...`，否则代码里的路径还指向原机器。

- 方法是什么：[../docs/方法说明.md](../docs/方法说明.md)
- 怎么运行：[../docs/运行手册.md](../docs/运行手册.md)
- 各部件的英文原始说明：[reid_lab/work/fuse_r1/README.md](reid_lab/work/fuse_r1/README.md)、[m_cuts](reid_lab/work/m_cuts/README.md)、[m_appear](reid_lab/work/m_appear/README.md)、[m_space](reid_lab/work/m_space/README.md)

English: the lab itself. This folder is what the code puts on `sys.path`; `reid_lab/` is the package. Run `python relocate.py ...` one level up before using it.
