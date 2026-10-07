# MPM-Lite 最新改进研究版本

本分支基于 `fy-123456/mpm-lite` 的 `main`，加入本地验证的完整改进源码。原始项目许可证保留。研究发布为 `material-coupling-next/20261007T155407Z-material-coupling-next`，release SHA256 **57b08528621258129026ce26e11b3a9f70738ae7e25e817227ca00b203bd08c8**。源目录没有Git元数据，因此该研究身份与本次Git提交身份分开记录。

## 本次改进与范围

- 全部1397项冻结Python源码：此前1377项 + 最新20项，包含各向异性、局部富集、完整惯性、材料积分压缩和压力功兼容研究。
- 最新入口增加分区应力/能量/力校验、共享材料与几何预算、真实初态材料拒绝后的q7整步重算，以及常量SPD流阻的统一方程。
- 修复α=0对照的多步时间/边界匹配。7项合同测试通过；实际4步封闭场景、排水与干固体对照、半步长检查和检查点恢复通过限定验收。
- 短窗稳定不等于精度认证。半步长冲量变化约2.91%，超过本次排水/封闭反力差约0.74%；独立空间参考、一般强非线性收敛、生产APIC/GPU和完整论文证据仍未认证。

详细状态：[实施记录](docs/MPM_LITE_MATERIAL_RULE_AND_MECHANICAL_COUPLING_PROGRESS_20261007_ZH.md)、[能力矩阵](docs/results/material-coupling-next/20261007T155407Z-material-coupling-next/capability-matrix.json)。旧文档中的“当前默认”和历史结论按其版本理解，不自动代表本分支已完成生产集成。

## 下载与依赖

GitHub禁止向此公开Fork新增LFS对象，因此本分支使用普通Git中的32MiB数据分片保存大缓存。克隆后执行一次还原；分片、完整文件均校验SHA256，保持原始字节，没有降低精度或重新生成基函数。仓库下载包含约662MB缓存分片，需预留至少2GiB本地空间。

```bash
python tools/restore_space_cache.py
uv sync
uv pip install matplotlib
```

`matplotlib`仅用于重新绘图，浏览已生成的页面不需要它。`pyproject.toml`/`uv.lock`保持本地版本原样。环境安装未在本次上传中重新执行。

## 便携校验与轻量测试

```bash
python tools/verify_research_upload.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/verify_research_upload.py --smoke
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest tests.research_material_coupling_next.test_contracts -v
```

校验器核查本次发布文件、1397项数值源码及完整最新95项封存产物；`--smoke`额外从本分支加载完整质量/空间缓存和q/v/p检查点，对照保存的场查询结果，只读不改封存目录。SHA不符或未还原缓存时直接失败。

## 最新可视化

```bash
python -m http.server 8765 --bind 127.0.0.1   --directory docs/results/material-coupling-next/20261007T155407Z-material-coupling-next
```

浏览 `http://127.0.0.1:8765/`；远程IDE转发8765。页面展示真实4步数据、两区压力、位移、J、反力与能量，支持显示倍率切换。165个探针不等于全场极值。

## 历史数据与审计说明

此分支包含全部改进源码、普通文档、最新发布和直接父发布的全部产物、32个祖先发布的元数据/索引，以及最新模型运行必须的7个空间包/数组文件。所有必要数据目录已实体化，不包含指向`/root/autodl-tmp`的断裂符号链接。

未上传虚拟环境、凭据、编译缓存及数GB以上的历史实验轨迹；历史基准若依赖这些旧结果，需要另取原始归档。旧元数据中保留的绝对路径用于来源记录，不代表远程机器应具有这些目录。

原`publication`命令是针对完整本地32代历史归档的审计，不能在该精简分发包上直接替代便携校验。本次源端完整审计已通过：32个祖先、1377项旧源码、20项新源码、47829个唯一文件、95项最新产物。本分支不声称包含全部历史大数据；`github-upload-manifest.json`明确定义上传和校验范围。

不要向已封存结果目录重跑会写文件的benchmark。新的物理实验应使用新输出目录及单独协议。
