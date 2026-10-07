# E 第二阶段实现

入口：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m benchmarks.research_e.stage2.run`。

仅新增 E stage2 文件，旧父包及 E 第一轮资产只读。入口先核验最新共同父包和原 E 清单，再从共同源码 zip 恢复独立副本，补入 E 四项额外源码资产，解引用复制输入数组，并用该副本导入严格冻结加载器。协议、源码摘要先于正式实验封存；成果在数据盘 E 私有目录，仓库保留同路径链接。

`evaluation.py` 对真实 Q2 梯度积分，以相同点/权重比较制造解。平面应变采用完整 3×3 张量，zz 不省略；压力常值重建并从对角线上减去一次。center、cell_mean、full_field 是三个不同输出。数学控制采用可解析 Q2 交叉多项式，保证中心与平均不再混淆；全域、边界带、内部均为必检区域。

`transaction.py` 产生拥有数据副本的纯值候选，由父 `StateTransaction` 统一提交。E 不持有独立 committed 历史，不提前调用外部对象的 commit。候选携带二维空间、参数、单位、时间、通量和账本；验证器重算真实混合方程残差。恢复同时检验内容和物理配置身份。

`handoff.py` 输出 A/G/C/B/H、非对称非 SPD 混合矩阵、作用和参考解，严格校验预期空间、边界、dt、单位、协议和源码。D 新增二维 (u,p) MixedBlockContract 后，E 通过精确 RT0 消元使用真实 D 合同校验，重建通量并复核原三块残差/作用。D 源码作为独立只读输入绑定摘要，E 不修改 D 公共入口；C 实际动态推进和 D 求解后端仍需后续接入。未认证 GPU、三维、移动、有限变形或生产 Lite。
