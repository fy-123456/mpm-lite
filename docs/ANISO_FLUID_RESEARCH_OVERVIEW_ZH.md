# MPM Lite 各向异性与流体耦合：研究概况及阅读建议

整理日期：2026-09-27。

本文依据本地源码、已有实验报告及相关论文整理。“当前版本”指本地工作区：已查看的源码包含 9 月 27 日新增内容，最新完整实验报告为 9 月 26 日。本地目录没有 `.git`，无法确认对应的远端提交；实验结论引用已有记录，本次整理未重新运行实验。

## 1. 拟改进的开源代码、论文及选择理由

- **开源代码：**[MPM Lite 官方仓库](https://github.com/f1shel/mpm-lite)。
- **基础论文：**[MPM Lite: Linear Kernels and Integration without Particles，2026](https://arxiv.org/abs/2602.07853)。
- **项目主页：**[MPM Lite](https://mpmlite.github.io/)。
- **选择理由：**通过固定积分点减少隐式求解中对粒子的重复访问，具有计算效率优势；代码规模适中，便于修改材料模型、传递方法和求解器，适合作为各向异性及后续流固耦合研究的基础。

## 2. 前人工作的问题与局限

- 传统隐式 MPM 的材料积分和切线计算通常随粒子数量增加，计算成本较高。
- MPM Lite 原有应力—伸长重建利用了各向同性性质，推广到各向异性时，需要额外处理材料方向和变形历史。参见[原方法说明](https://mpmlite.github.io/)。
- 针对本项目的扩展需求，简单平均方向或变形会损失局部相关信息，影响能量、应力和刚度；这一点已有本地[联合采样实验](ANISO_JOINT_SAMPLING_ZH.md)支持。这是本地扩展中的诊断结论，不应直接作为对原论文所有算例的评价。
- 流体—织物耦合已有研究，例如 [Liquid-Fabric Interactions](https://www.cs.columbia.edu/cg/wetcloth/)，但如何与 Lite 的高效积分架构结合，仍需研究，不能直接套用现有方法。

## 3. 想改进的地方与主要解决的问题

- **近期：**降低粒子跨网格时的应力振荡、网格位置敏感性和速度投影耗散。
- **核心方法：**使材料积分、粒子—网格传递、变形历史更新保持一致，并用少量积分状态保留方向—变形关联。
- **后续：**加入流体与各向异性固体的双向作用；若研究多孔介质，再引入方向相关的渗透、阻力和饱和度。
- **潜在创新：**在保持 Lite 效率优势的同时，实现可靠的各向异性状态压缩和耦合离散；目前仍需文献对照、精度、稳定性及整体性能实验支撑。

需要区分：**弹性各向异性描述不同方向的变形难易，渗透各向异性描述液体沿不同方向通过的难易。当前代码主要处理前者。** 表面流固耦合与介质内部渗流涉及不同状态和方程，应根据目标场景选择。

## 4. 推荐阅读的论文

按与当前工作的关联顺序阅读；兼顾近期研究和必要的基础论文。

| 论文 | 主要阅读目的 |
| --- | --- |
| [MPM Lite: Linear Kernels and Integration without Particles，2026](https://arxiv.org/abs/2602.07853) | 理解当前代码架构、固定积分点及各向同性重建假设 |
| [An Angular Momentum Conserving Affine-Particle-In-Cell Method，JCP 2017](https://arxiv.org/abs/1603.06188) | 理解粒子—网格传递、动量守恒和数值耗散；链接为 2016 年预印本 |
| [Anisotropic Elastoplasticity for Cloth, Knit and Hair Frictional Contact，2017](https://mass.math.ucdavis.edu/~jteran/papers/JGT17.pdf) | 学习材料方向、各向异性本构与变形更新；其薄结构及摩擦模型不同于当前体积弹性模型 |
| [A Multi-Scale Model for Simulating Liquid-Fabric Interactions，2018](https://www.cs.columbia.edu/cg/wetcloth/) | 学习各向异性阻力、毛细作用、饱和度及液体—织物耦合 |
| [Solid-Fluid Interaction on Particle Flow Maps，2024](https://arxiv.org/abs/2409.09225) | 了解近期弹性固体与流体双向耦合思路；并非专门的各向异性渗流模型 |
| [A Semi-Implicit Double-Point Material Point Method for Both Free-Surface Flow and Seepage in Deformable Porous Media，2026](https://arxiv.org/abs/2608.00578) | 了解自由液面、渗流与可变形多孔介质的统一处理；迁移到纤维介质需额外处理方向性 |

**优先精读三篇：MPM Lite、APIC、Liquid-Fabric Interactions。** 分别对应当前架构、传递误差和后续耦合物理。

若主要负责压力求解和渗流稳定性，可补读 [Stabilized Two-Phase Material Point Method for Hydromechanical Coupling Problems in Solid–Fluid Porous Media，2025](https://gmd.copernicus.org/articles/18/4743/2025/)，重点关注压力振荡、体积锁定和验证算例。

## 5. 目前已经完成的工作

依据本地源码和已有实验记录：

- 实现了纤维增强各向异性弹性模型、应力及切线，以及隐式求解和历史提交／回滚。当前基础二次纤维能量在拉伸和压缩时都生效，尚不是完整的真实织物或湿润材料模型。
- 开展了四阶方向矩、分组材料积分、二次空间重建及稳定化研究。四阶方向矩的精确压缩结论限于共同变形梯度和当前二次纤维模型。
- 建立了完整质量矩阵、统一插值和粒子历史更新的一致性参考实现。
- 完成实际跨网格实验：**传递与历史一致性通过，但时间精度及网格敏感性仍未达标**。详见[最新跨网格验证报告](ANISO_XZ_MPM_VALIDATION_ZH.md)。
- 9 月 27 日源码新增二次 B 样条、弱支撑节点处理和速度残差增强选项；本次查看未找到这些新增选项的完整验收报告，不能认定相关问题已经解决。
- **流体耦合尚未完成；小规模一致性参考实现也尚不能等同于生产 Lite 路径已修复或已获得整体加速。**

相关材料：[各向异性扩展说明](MPM_LITE_ANISOTROPIC_EXTENSION_ZH.md)、[联合采样研究](ANISO_JOINT_SAMPLING_ZH.md)、[材料积分生产接入检查](ANISO_QUADRATURE_PRODUCTION_GATE_ZH.md)。

## 6. 可以从哪部分入手学习和参与

| 切入点 | 可承担的具体工作 |
| --- | --- |
| 本构与基础力学 | 阅读 [constitutive.py](../engine/aniso_phase1/constitutive.py)，复现不同纤维角度下的拉伸、剪切和旋转测试 |
| 数值方法，当前最推荐 | 围绕 [spatial_mpm.py](../engine/aniso_phase1/spatial_mpm.py) 和 [spatial_basis.py](../engine/aniso_phase1/spatial_basis.py)，比较插值核、边界支撑及传递方式，分析误差和耗散来源 |
| 材料积分与压缩 | 比较平均方向、四阶方向矩和分组采样的精度与成本 |
| 流体耦合预研 | 从固定各向异性多孔块的渗流基准开始，再加入骨架变形和双向反馈 |

第一项适合熟悉代码；第二项最能直接推进当前工作；第四项适合为后续流体方向准备模型和验证算例。

**建议第一个具体任务：**围绕现有 432 粒子场景，固定材料和初态，逐项比较插值核、边界支撑处理及速度传递方式。记录跨单元前后的应力变化、质量矩阵条件数、时间／空间加密误差，以及时间积分和传递投影各自的能量损失。基准入口为 [benchmarks/aniso_spatial_mpm.py](../benchmarks/aniso_spatial_mpm.py)。
