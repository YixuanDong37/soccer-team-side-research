# patches：对第三方代码的本地补丁

四个补丁文件都是原样保存的，**不要改动它们**：来源记录（`../gsr/provenance-gsr*.json`）里登记了它们的 sha256，`relocate.py` 也不会改它们。

| 文件 | 打在哪里 | sha256 | 改了什么 |
|---|---|---|---|
| `sn-gamestate-local-v2.patch` | sn-gamestate，提交 `1c958345067218297d221e45e1a6405f975f83e0` | `ff2236e0bc2d6f744547faf22019ad4be16672e63d3090596d7b01fcefb0fa75` | `pyproject.toml`：去掉 torch 1.13.1 的固定版本和 MMOCR 一组依赖，prtreid 和 bpbreid 改用本地副本；标定里按位置取关键点；EasyOCR 的读取器只建一次；队别聚类改成按检测数加权的 KMeans |
| `sn-gamestate-local.patch` | 同上 | `7e922c34c3ca5c4b927befc2acaa5cff9887360b827fa876774d994355f6f927` | 上面那个补丁的第一版，只留作记录，不要用 |
| `prtreid-local.patch` | prtreid，提交 `30617a75967e84d5d516959c4b84cbeea6f56493` | `a18c422000b63015b79f002f565f095ff507b91095798f8d42baf79a417a5e1a` | 不编译只在评估时用的 Cython 扩展 |
| `bpbreid-local.patch` | bpbreid，提交 `a2dc4304284784d8b2c061764512f0698aa38c21` | `1fe808fa176717df5c062fc6c36095a4f0238a1ca1970fd380a0e66bb6b6baff` | 同上 |

用法：

```bash
cd <sn-gamestate> && git apply <本文件夹>/sn-gamestate-local-v2.patch
cd <prtreid>      && git apply <本文件夹>/prtreid-local.patch
cd <bpbreid>      && git apply <本文件夹>/bpbreid-local.patch
```

打完 sn-gamestate 的补丁后，手工改一处：`pyproject.toml` 里 `"prtreid @ file:///..."` 和 `"torchreid @ file:///..."` 两行写的是原机器上的位置，改成你本地 prtreid 和 bpbreid 的位置。

许可：sn-gamestate 的两个补丁是对 GPL-3.0 代码的修改，按 GPL-3.0；prtreid 和 bpbreid 的补丁是对 Hippocratic License 3.0 代码的修改。这里只有补丁，没有这些项目的源码。

English: four patches kept verbatim (their sha256 is recorded in the provenance files; `relocate.py` never rewrites them). Apply `sn-gamestate-local-v2.patch` to sn-gamestate at commit 1c958345, and the two small patches to prtreid (30617a75) and bpbreid (a2dc4304). After applying, edit the two `file:///` lines of sn-gamestate's `pyproject.toml` so that they point at your clones. The sn-gamestate patches are derived from GPL-3.0 code; the other two from Hippocratic License 3.0 code.
