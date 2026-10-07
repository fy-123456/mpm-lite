# 本轮三维固体与少量压力单元接口

2026-10-01。本接口基于本轮选中的 `nonlinear-modes-swap6` 固体空间，仍为 144 个局部标量函数、648 个独立自由向量分量，使用完整 M7、原 Ks 和充分材料 q7。纯固体日常入口保持独立。来源与验收见 [实施记录](MPM_LITE_SPATIAL_PHASE_PROGRESS_20261001_ZH.md)、[N4 成果](results/spatial-phase/20261001T043604Z-spatial-phase/N4)。

## 空间、参数与边界

- 压力在参考 x 方向分成 2 个常值单元；4 单元仅用于已登记的敏感性检查。横向流量封闭，不是任意三维压力网格。
- 面未知量是按全局 +x 统一朝向的体积流率，单位 m³/s。内部面只保存一个流率，相邻单元符号相反。
- 固体夹持提升固定为 0；初始压力 0.01 Pa。封闭场景没有外部流量；排水场景右侧储库压力为 0.002 Pa，左侧封闭。左半域源项为每秒 0.001 倍参考单元体积。
- 参数为数值验证用途：α=0.8，S=0.2 Pa⁻¹，k=1e-4 m²，μ=1e-3 Pa·s。S=0.0002 只作小储存秩诊断，不代表材料标定。
- 每例 Δt=0.01 s、4 步。几何积分按固体单元与压力单元的交集分割，切口不对齐时增加真实积分分段，不能只按旧 Gauss 点坐标分类。

## 同一体积定义对应压力力与含量

每个压力单元 k 定义

\[
V_k(q)=\int_{\Omega_{0,k}}\det F(q)\,dV,
\qquad C_k=S|\Omega_{0,k}|,
\]
\[
m_k=\alpha(V_k-V_{0,k})+C_kp_k,
\qquad E_f=\frac12\sum_kC_kp_k^2.
\]

m 是积分体积含量，单位 m³；没有乘流体密度，不能把它标成 kg。固定 m 求储能对 q 的导数得到

\[
f_p=-\alpha\sum_kp_k\nabla_q V_k.
\]

沿系数直线段用两点 Gauss 计算离散梯度 \(\bar g_k\)。det(F) 沿该路径为三次式，其梯度为二次式，因此

\[
\bar g_k^T(q_1-q_0)=V_k(q_1)-V_k(q_0).
\]

中点压力 \(\bar p=(p_0+p_1)/2\) 同时进入固体压力力与流体储存方程。逐单元交换功为

\[
W_{s,k}=-\alpha\bar p_k\bar g_k^T\Delta q,
\qquad W_{f,k}=\alpha\bar p_k\Delta V_k,
\qquad W_{s,k}+W_{f,k}=0.
\]

## 混合流量与耗散

B 的正号为该单元流出。每个内部面在 B 中恰有一个 +1 和一个 −1：

\[
\dot{\boldsymbol m}+B\boldsymbol z=\boldsymbol s,
\qquad H(q)\boldsymbol z=B^T\boldsymbol p-\boldsymbol g_\partial.
\]

以参考 RT0 面基函数构造

\[
A(F)=\frac{k}{\mu}J F^{-1}F^{-T},\qquad
H_{ij}=\int\psi_i^TA^{-1}\psi_j\,dV.
\]

本实现活跃流量为 x 面通量，但保留真实张量关系：\((A^{-1})_{xx}=(F^TF)_{xx}/[(k/\mu)J]\)，不能替换成 \(1/A_{xx}\)。独立的六面 RT0 解析检查覆盖常非对角迁移率及方向／单位；主场景仍明确约束横向流量为零。

一单元右排水 RT0 电阻为 \(L/(3A_f k/\mu)\)，历史单单元的体积平均两点电阻为 \(L/(2A_f k/\mu)\)。两者空间试函数和边界处理不同，历史结果保持原样，以独立解析参照检查新离散。

## 真实残差、能量与求解

固体使用原 AVF。以 \(W=(q_1-q_0)/h\)、\(v_1=2W-v_0\) 为例，解耦合残差

\[
2M(W-v_0)+h\left(\bar f_s-\alpha\sum_k\bar p_k\bar g_k-f_{ext}\right)=0,
\]
\[
\alpha\Delta V+C\Delta p+hB\bar z-hs=0,
\qquad H(q_{mid})\bar z-B^T\bar p+g_\partial=0.
\]

采用一般稠密直接求解、弦截迭代矩阵和真实残差线搜索。迭代中的固体块使用静止 K₀，是明确的近似矩阵；没有把非线性完整 Jacobian 假定为 SPD，没有使用 CG 或压力罚项。正常与小储存的 2／4 单元混合矩阵均满秩，这只是该有限问题的可解性证据，不是一般 inf-sup 认证。

每步分别报告固体材料能、Ks、动能、流体储能、外力功、源功、储库功、Darcy 耗散、AVF 路径误差和求解误差。流体外部供能与符号约定一致：

\[
W_{fluid,ext}=h(\bar p^Ts-g_\partial^T\bar z),
\qquad D=h\bar z^TH\bar z\ge0.
\]

验收逐单元 \(\Delta m+hB\bar z-hs\)，并核对累计边界／源历史与全域含量。展示总应力

\[
P_{total}=P_s-\alpha pJF^{-T}.
\]

文件中 `solid_PK1` 与 `total_PK1` 分开；`pressure_Pa` 是单元常值压力，接口处采用明确的单元归属，不作平滑。场比较采用共同物理位置、位移 u=x−X、分区体积加权 RMS；压力在同一参考单元均值上比较。

## 提交、恢复与适用边界

一次外层事务拥有 q、v、预测器、每单元压力／含量、面流量、累计边界流量和源历史。局部数组被破坏后抛错必须恢复整步；检查点指针决定是否已经提交，发布后观察错误不得重复推进。两单元排水场景在第 2 步退出并由新进程恢复，与不中断结果一致；最终数值源码下再验证归档载入后的恢复。

已检查：能量方向导数、压力力切线、离散体积梯度、q7/q8 几何作用、张量通量解析对照、混合秩、α=0 固体极限、固定固体中点流动极限、封闭／排水短程及局部事务。仍未认证连续压力精度、一般三维流量网格、长期孔弹性响应或生产 C–E 接口。纯固体 q5 资格不用于本耦合计算。

已实现的可视化命令只读取提交帧，不重新计算场景：

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_spatial_phase_next.coupling_study render \
  --run docs/results/spatial-phase/20261001T043604Z-spatial-phase \
  --output /root/autodl-tmp/mpm-lite-spatial-phase/preview-coupled-latest
```

## 图像复核发现的瞬态限制

排水初期的内部流量短暂反向，固定固体极限也出现小幅压力过冲。已按用户要求作有限根因分析，见 [解析诊断](results/spatial-phase/20261001T043604Z-spatial-phase/N4/flux-transient-diagnostic.json)。本两单元常迁移率布局的活跃通量矩阵为

\[
H=\begin{pmatrix}40&10\\10&20\end{pmatrix},\qquad
B=\begin{pmatrix}1&0\\-1&1\end{pmatrix}.
\]

H 的非对角项将内部流量与右侧排水耦合。初始两单元均为 0.01 Pa、右储库为 0.002 Pa 时，即使没有体积源，离散瞬时内部流量也是 −1.142857e-4 m³/s，上游单元的离散压力导数为 +0.024381 Pa/s。以该半离散线性系统的矩阵指数作精确时间演化，0.01 s 上游压力仍为 0.010186 Pa，高于初始上限；所以不能把现象仅归因于中点时间误差、固体耦合或回滚实现。登记的固定固体第一步也被解析中点矩阵复现至约 1.7e-18。

根因是粗 P0 压力与一致 RT0 通量在突然施加排水边界时不保持单调性。守恒、非负耗散和正确弱形式不自动保证无过冲。因此本接口保留为稳定的有限混合算子／事务验证案例，**不声明压力单调性、连续压力精度或生产可用性**。没有将 H 偷换成独立面阻力来掩盖现象；下一轮需单独研究压力分辨率、边界施加方式与相容的单调性处理。
