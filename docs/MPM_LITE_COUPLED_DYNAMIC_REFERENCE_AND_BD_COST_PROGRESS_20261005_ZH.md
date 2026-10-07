# 运动骨架耦合参考与 BD 成本实施记录

执行 [MPM_LITE_NEXT_COUPLED_DYNAMIC_REFERENCE_AND_BD_COST_PLAN_20261005_ZH.md](MPM_LITE_NEXT_COUPLED_DYNAMIC_REFERENCE_AND_BD_COST_PLAN_20261005_ZH.md)。

最新基线 `20261005T101215Z-transverse-reference`，SHA `e948656e54181fd2cb3bfc35db518ae071c54898d638453cd4bd3922e46b83b1`。结果 `/root/workspace/mpm-lite/docs/results/coupled-reference/20261005T105825Z-coupled-reference`。

S0：23源码/23快照/623产物及祖先全审计通过。原144空间、M7/q7、128格与200微秒身份保持；新数据/缓存独立，系统盘7.51 GiB，不触发迁移。

S0接线问题：首次构造在物理身份检查处拒绝，尚未推进。原因是内存元组与JSON列表直接比较，已统一JSON表示，未放松字段检查或改动方程；失败进程19.85秒计入预算。

S1：已构造含完整质量投影、压力几何项、Darcy几何项和非零初始驱动的r8/r12诊断参考，压力保持128格。4方向导数检查 通过；正式空间未变，完整空间参考不作认证。

S2桥接：BD第8→9步、同位置场、含量/能量账本及错身份拒绝 通过。同步诊断推进3.175s；局部raw转置0.218s，896次。

S1参考自检：{'rank': 12, 'initial_error': 0.0, 'max_cell_mass_defect_m3': 2.7828692854047223e-22, 'min_pressure_Pa': 0.003540102891758712, 'nonzero_affine_drift': True, 'new_GPU_steps': 0, 'independent_method': 'scaled BDF, independent of augmented expm', 'independent_pressure_error_Pa': 7.301812793292228e-09, 'independent_rhs_calls': 644, 'independent_times_s': [2.5e-05, 3.7500000000000003e-05], 'fixed_skeleton_degenerate_error_Pa': 7.573108806724349e-13, 'passed': True}。r8→r12压力差5.21e-10Pa，最大全空间加速度投影残差比3.19e-08；仍标记diagnostic_only。S2扩窗/h2不触发，不把投影模型当真实非线性真值。

S1 CPU自检首次因浮点时间节点略超字面量终点而拒绝，已改用冻结节点作为t_span终点；不改变步长或容差。S3：登记唯一候选FBD，共用x向转置的代数分配律，保留完整P、原稀疏转置和原方程。

S3唯一候选FBD：4状态算子、完整P压力功/体积链、返回所有权、元数据失效与局部失败重建 通过。

S3公平单步配对：[(8, 0.00664075642022488, 55.6073615550284), (16, 0.10232966970802848, 0.7299372927408645)]；未满足采用条件，保留BD。

S1追加单一预登记留出状态：未参与参考构造的第16步/50微秒工程检查 通过；最大压力差0.00087Pa。新增动态步0。新进程实际加载BatchDownloadGeometry第9步。


## 本轮结果与限制

- 已完成真实耦合的局部诊断参考：原完整质量投影、128格压力、非零初始驱动、压力几何切线及Darcy几何导数均保留；4方向检查通过。
- 独立BDF与指数参考压力差7.3e-09 Pa；r8/r12压力差5.21e-10 Pa。三个时刻全空间加速度投影残差比最大3.19e-08。这些是诊断证据，不是连续空间或完整非线性精度证明。
- 参考构造使用的快照也用于局部余项检查，明确不是独立隐藏集。实体位移约80纳米，低于工程绝对下限；不宣称机械相对精度。
- 不扩窗、不新增物理帧、不更换144空间；纯固体252步结果仅只读继承。
- FBD共用两支局部梯度的x向转置，减少重复计算；两组整步相对BD降幅0.66%/10.23%。采用决定：BD，理由：speed/setup/isolation gate not met; retain certified BD。仅代表实际短窗整步，不外推完整周期。
- 6项CPU合同测试通过，5次实际动态尝试、5次成功；GPU相关完整进程暂计178.2秒，最终总数见S5/attempt-accounting.json。
- 新进程实际加载BatchDownloadGeometry第9步；守恒账本与逐步真实残差通过。系统盘7.51GiB，未触发迁移，结果与缓存位于数据盘。

## 下一步

已补第16步/50微秒留出状态工程检查，最大压力差约0.000870 Pa。下一轮先分析原始启动流量及时间相位是否有可分辨误差，再决定最小h/2短窗；获得独立空间参考后才调整144函数预算。完整耦合周期、生产C/E、耦合q5和真实驱动丢失恢复仍未认证。

## 最新可视化CLI

```bash
cd /root/workspace/mpm-lite
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 \
  --directory docs/results/coupled-reference/20261005T105825Z-coupled-reference
```

IDE转发8765后打开 http://127.0.0.1:8765/ 。参考曲线为本轮新计算，实体场为继承75微秒数据，新增物理帧0。浏览器交互未验证。


S5：实际查看5张图；11个HTTP页面资源/内容及PNG解码检查通过，测试服务已关闭。保留BD；第一组配对在未改动的cofactor/download段出现额外耗时，根因未确证，不追加重复实验或宣称稳定整步提速。
