# MPM Lite 算法框架、代码流程与现状评估

> 公式采用 Markdown 数学语法：独立公式使用 `$$...$$`，行内符号使用 `$...$`，可在 GitHub、VS Code Markdown 预览和支持 MathJax/KaTeX 的阅读器中渲染。

> 本文对应仓库当前快照（2026-09-19），分析对象是 engine/solver3d.py、engine/kernel/d3/kernel_lite.py、engine/kernel/d3/kernel_lite_implicit.py、engine/kernel/d3/kernel_lite_misc.py 和 mpmlite2d.py。文中把“论文/项目声称的性质”和“当前代码确实实现的性质”分开描述。

## 1. 先给结论

这个仓库不是普通的“把传统 MPM 搬到 Warp/GPU 上”的实现，而是 [MPM Lite: Linear Kernels and Integration without Particles](https://arxiv.org/abs/2602.07853) 的参考代码。它的核心改变是：粒子继续保存位置、速度、变形梯度和塑性历史，但隐式求解时不再以粒子作为积分点；粒子状态先被重采样到固定的 cell-center quadrature，之后的内力、能量、切线和时间积分全部在中心点/网格节点上完成。[论文的摘要和引言](https://arxiv.org/html/2602.07853v3)把这一点概括为 “integration without particles”。

截至 2026-09-19，可以比较准确地说：

- 它是 2026 年公开的、面向图形学大变形材料、把隐式积分从粒子循环中解耦出来的代表性开源实现。
- 不能严谨地说它是“全领域最新的隐式 MPM 开源代码”。“最新”取决于问题类别、发布时间、是否有公开代码以及是工程 MPM 还是图形学 MPM。2026 年 4 月已经出现新的 [implicit CK-MPM 论文](https://arxiv.org/abs/2604.18917)，其重点是紧支撑核和接触精度；[Newton 的 SolverImplicitMPM 文档](https://newton-physics.github.io/newton/1.3.0/api/_generated/newton.solvers.SolverImplicitMPM.html)也公开了另一条 GPU 隐式 MPM 路线。它们和 MPM Lite 的目标不同，因此更合适的表述是“MPM Lite 是当前较新的、方法上有明显独创性的隐式 MPM 参考实现之一”，而不是唯一最新。
- 当前仓库更像研究参考实现，而不是已经覆盖所有工程边界条件、接触、各向异性材料和自适应网格的生产求解器。隐式代码尤其需要在更复杂场景中做收敛、能量、网格和时间步验证。

## 2. 代码结构和离散对象

主要文件的职责如下。

| 文件 | 作用 |
| --- | --- |
| engine/solver3d.py | MPMSolver、材料注册、粒子/稀疏块管理、一步显式/隐式求解 |
| engine/kernel/d3/kernel_lite.py | 粒子到中心、中心到节点、节点到中心、中心到粒子的线性核传输和塑性更新 |
| engine/kernel/d3/kernel_lite_implicit.py | 隐式残差、∇δv、应力增量、矩阵自由 Ap、局部对角块预条件 |
| engine/kernel/d3/kernel_lite_misc.py | 隐式迭代初始化、活动节点/中心压缩、quadrature scratch 更新 |
| engine/math/conjugate_gradient.py | 预条件矩阵自由共轭梯度（PCG） |
| engine/kernel/d3/helper_constitutive_model.py | Hencky-StVK、压力模型以及 von Mises、Drucker–Prager、NACC、snow、foam 返回映射 |
| engine/boundary_utils.py | sticky/slippy 节点和高频移动平面边界的投影 |
| engine/sp_grid.py | 32^3 block 的稀疏网格索引 |
| mpmlite2d.py | 独立的 2D 演示；它不是 MPMSolver 的 2D 版本 |

记号如下：

- 粒子：$p$，保存 $x_p,v_p,F_p,G_p$、质量 $m_p$、参考体积 $V_p$ 和材料历史。
- 固定中心 quadrature：$c$，位于每个 voxel 的中心，保存 $m_c,v_c,G_c,V_{c,k},\tau_{c,k}$。
- 网格节点：$i$，保存 $m_i,v_i$。
- $w_{cp}=N_c(x_p)$ 是粒子到中心的 Q1 多线性权重；$w_{ic}=N_i(x_c)=2^{-d}$ 是中心到节点的角点权重。
- 3D 中中心到节点的一个中心只连接 8 个角点；粒子到中心也最多连接 8 个中心。因此核心传输是线性核，而不是传统二次 B-spline 的 27 节点支撑。

## 3. 一步时间推进的总流程

### 3.1 Unload：粒子 → 中心

lite_p2c_kernel_1 对每个粒子计算应力和线性核权重，然后累加：

$$
m_c^n=\sum_p w_{cp}m_p,
$$

$$
\begin{aligned}
m_c v_c^n
&=\sum_p w_{cp}m_p
  \left[v_p^n+G_p^n(x_c-x_p^n)\right].
\end{aligned}
$$

$$
m_cG_c^n=\sum_p w_{cp}m_pG_p.
$$

G_p 是粒子携带的速度梯度/仿射信息。代码中的 enable_apic=True 时，G_p(x_c-x_p) 被用于局部线性速度外推；关闭时退化为 PIC 风格的常速度传递。

应力不直接平均 F_p。先由本构模型得到 Kirchhoff 应力

$$
\tau_p=P(F_p)F_p^\mathsf{T},
$$

再以“体积乘应力”这个可加的广延量进行重采样：

$$
\begin{aligned}
V_{c,k}^n &= \sum_{p\in k}w_{cp}V_p,\qquad
\tau_{c,k}^n &= \frac{\sum_{p\in k}w_{cp}V_p\tau_p^n}{V_{c,k}^n}.
\end{aligned}
$$

因此严格满足

$$
\sum_c V_{c,k}^n\tau_{c,k}^n 
=\sum_{p\in k}V_p\tau_p^n.
$$

这是本项目最重要的物理选择之一：F 既不是强度量也不是广延量，直接平均不同粒子的 F 会把不同旋转和拉伸混在一起；Vτ 才适合用作力积分的重采样量。

lite_p2c_kernel_2 再把 v_c,G_c,τ_c 归一化。如果用于隐式积分，还会由 τ_c 重建一个无旋转的基准拉伸 F_c^base。

### 3.2 Center → Grid：中心 → 节点

lite_c2g_kernel_1 对每个活动节点执行 gather：

$$
m_i^n=\sum_{c\in{\rm Corners}(i)}w_{ic}m_c^n,
$$

$$
\begin{aligned}
(mv)_i^n &= \sum_c w_{ic}m_c^n
  \left[v_c^n+G_c^n(x_i-x_c)\right],\qquad
v_i^n &= \frac{(mv)_i^n}{m_i^n}.
\end{aligned}
$$

中心到节点的内部力使用中心的应力：

$$
f_i^{\rm int,n} 
=-\sum_{c,k}V_{c,k}^n\,\tau_{c,k}^n\nabla w_{ic}.
$$

Q1 权函数在 3D cell center 的梯度为

$$
\nabla w_{ic} 
=\frac{1}{4\Delta x} 
(s_x,s_y,s_z),\qquad s_a\in\{-1,+1\},
$$

这正对应代码里的 inv_4dx = 0.25 / dx。

显式时间推进为

$$
v_i^{n+1} 
=v_i^n+\Delta t\, 
\frac{f_i^{\rm int,n}+f_i^{\rm ext,n}}{m_i^n}.
$$

代码中外力目前主要是重力，边界节点随后经过 proj_boundary_vel 投影到 sticky 或 slippy 约束。

### 3.3 Implicit：在网格上解增量速度

隐式版本先将

$$
v_i^{(0)}=v_i^n+\Delta t\,g
$$

作为初始迭代值（再进行边界投影），然后在活动节点上求增量 δv。中心速度梯度是节点增量的线性函数：

$$
G_c(\delta v) 
=\sum_{i\in{\rm Corners}(c)} 
\delta v_i\otimes\nabla w_{ic}.
$$

代码 lite_implicit_grad_v_dv_kernel 只做这一步，不访问粒子。

在当前迭代速度 v 下，中心试探变形为

$$
F_c(v)=\left(I+\Delta t\,G_c(v)\right)F_c^{\rm base}.
$$

离散增量势（省略常数外力项）为

$$
\begin{aligned}
\Phi(v)={}&\sum_i\frac12m_i\|v_i-v_i^n\|^2\\
&+\Delta t\sum_i m_i\,(-g)\cdot v_i\\
&+\sum_{c,k}V_{c,k}^n\,
\psi_k\!\left(F_{c,k}(v)\right).
\end{aligned}
$$

其一阶条件是节点残差

$$
\begin{aligned}
r_i(v)={}&m_i(v_i-v_i^n)-\Delta t\,m_i g\\
&+\Delta t\sum_{c,k}V_{c,k}^n
\left[\tau_{c,k}(v)\nabla w_{ic}\right].
\end{aligned}
$$

lite_implicit_residual_from_scratch_kernel_fast 实现的正是这个结构，并同时构造每个节点的局部 Hessian 对角块 H_ii。

代码没有组装全局稀疏矩阵。给定 CG 搜索方向 p，矩阵自由算子按以下链条计算：

$$
p_i 
\longrightarrow 
\nabla p_c 
\longrightarrow 
\delta F_c=\Delta t(\nabla p_c)F_c^{\rm base} 
\longrightarrow 
\delta P_c 
\longrightarrow 
\delta\tau_c 
\longrightarrow 
(Ap)_i.
$$

其中

$$
(Ap)_i= 
m_i p_i 
+\Delta t\sum_{c,k}V_{c,k}\, 
\delta\tau_{c,k}\nabla w_{ic} 
+\epsilon_i p_i .
$$

lite_implicit_precond_kernel_Hii 使用 H_ii^{-1} 做块对角预条件，matrix_free_cg 解

$$
A\,\delta v=-r.
$$

每个 Newton/fixed-point 外迭代结束后，代码把 δv 加到 grid_v_it。当搜索增量的无穷范数小于 v_tol 或达到 max_iters 时，才把迭代速度写入 grid_v_new。

当前实现有一个必须注意的事实：solver3d.py 中 alpha=1.0，且注释明确写着 line-search is ommitted for current repo。也就是线搜索接口保留了，但实际没有做回溯或信赖域控制；大时间步、强非线性和接触切换下不能把论文中的优化形式等同于“无条件收敛”。

### 3.4 Load：节点 → 中心 → 粒子

先在中心恢复

$$
v_c^{n+1}=\sum_iw_{ic}v_i^{n+1},
$$

$$
G_c^{n+1} 
=\sum_i v_i^{n+1}\otimes\nabla w_{ic}.
$$

再由中心插值回粒子：

$$
v_p^{n+1}=\sum_cw_{cp}v_c^{n+1},\qquad 
G_p^{n+1}=\sum_cw_{cp}G_c^{n+1}.
$$

速度使用 PIC/FLIP 混合：

$$
v_p^{n+1} 
=\eta\left(v_p^n+\sum_cw_{cp}\Delta v_c\right) 
+(1-\eta)\sum_cw_{cp}v_c^{n+1},
$$

其中 η=flip_ratio，默认构造函数值为 0.9。

位置和变形梯度更新为

$$
x_p^{n+1}=x_p^n+\Delta t\,v_p^{n+1},
$$

$$
F_p^{n+1} 
=\left(I+\Delta t\,G_p^{n+1}\right)F_p^n.
$$

对于水，代码只需要 J_p=det F_p；对于固体，随后按材料类型执行 von Mises、Drucker–Prager、NACC、snow 或 foam 返回映射。

## 4. 隐式本构和应力到拉伸的重建

### 4.1 为什么不能直接把应力当作普通显式应力率

如果采用 Jaumann 型应力率，

$$
\tau^{n+1} 
=\tau^n+\Delta t\left( 
\mathbb C:D(v)+W(v)\tau^n-\tau^nW(v) 
\right),
$$

其中 D=sym G、W=skw G，旋转项会使力对速度的 Jacobian 通常非对称，难以写成一个标量势的 Hessian。这样不适合直接使用依赖对称正定结构的优化型 Newton/PCG。

MPM Lite 的办法是只重建各向同性本构所需的 stretch。令

$$
\tau=P(F)F^\mathsf T,\qquad 
F=U\Sigma V^\mathsf T,
$$

对于谱能量 ψ(F)=ψhat(σ_1,...,σ_d)，

$$
\tau_i=\sigma_i 
\frac{\partial\widehat\psi}{\partial\sigma_i}, 
\qquad 
\tau=U\,{\rm diag}(\tau_i)\,U^\mathsf T.
$$

因此先对 τ_c 做特征分解得到 U，再解标量方程得到 σ_i，最后取

$$
F_c^{\rm base} 
=U\,{\rm diag}(\sigma_i)\,U^\mathsf T.
$$

它保留当前应力对应的拉伸状态，但舍弃无法从混合应力唯一恢复的旋转。对于各向同性材料，论文证明这种旋转自由处理每步相对于保留旋转的速度差为 O(Δt^2)。

### 4.2 Hencky-StVK

代码采用的 Hencky-StVK 能量是

$$
\psi(F) 
=\mu\sum_i(\log\sigma_i)^2 
+\frac{\lambda}{2} 
\left(\sum_i\log\sigma_i\right)^2.
$$

第一 Piola 应力为

$$
P(F) 
=U\left[ 
2\mu\Sigma^{-1}\log\Sigma 
+\lambda\,{\rm tr}(\log\Sigma)\Sigma^{-1} 
\right]V^\mathsf T.
$$

代码中 StVK_Hencky_PK1_3D、StVK_Hencky_dPK1_3D 和 dPdF_StVK_Hencky_3D_analytic_v2 分别负责应力、方向导数以及矩阵自由切线收缩；奇异值会用 s_min=1e-6 截断，并可对切线作正定投影/谱平移，以降低压缩状态下的病态性。

当前中心重建采用应力的偏/球分解。对 3D Hencky 模型，代码先由

$$
E= 
\frac{1}{2\mu}{\rm dev}(\tau) 
+\frac{{\rm tr}(\tau)} 
{3(2\mu+3\lambda)}I
$$

求对数应变，再通过谱指数得到 F_c^base=exp(E)。这正是 lite_p2c_kernel_2 中的 sym_exp3 路径。

### 4.3 水/压力模型

水不需要剪切变形，只追踪 J。当前实现使用

$$
\psi(J)=\frac{\kappa}{2}(J-1)^2,\qquad 
\pi(J)=J\psi'(J)=\kappa J(J-1).
$$

由中心压力反解

$$
J_c^{\rm base} 
=\frac{1+\sqrt{1+4\pi_c/\kappa}}{2}.
$$

隐式试探值为

$$
J_c(v)=\det(I+\Delta t\,G_c(v))J_c^{\rm base}.
$$

在代码中，压力本构的隐式切线使用 dJ=Δt J_base tr(G_c) 的方向导数。

## 5. 显式和隐式的执行顺序

### 显式 solver_type="lite_explicit"

1. 激活粒子影响到的 32^3 稀疏块。
2. lite_p2c：粒子质量、动量、G、体积和 Vτ 到中心。
3. lite_c2g(explicit_force=True)：中心 gather 到节点并计算内部力。
4. 节点施加重力和边界投影。
5. lite_g2c：节点速度和增量速度到中心。
6. lite_c2p：FLIP/PIC 回传、位置推进、F 更新和塑性返回映射。

### 隐式 solver_type="lite_implicit"

1. 执行同样的 lite_p2c。
2. lite_c2g(explicit_force=False)，只生成 m_i,v_i^n，不在初始阶段显式加内力。
3. 压缩活动节点和活动中心，建立 node2dof、center2dof。
4. 初始化 grid_v_it。
5. 每个外迭代：
   - 从当前网格迭代速度计算 G_c；
   - 更新 quadrature_scratch（SVD、τ、Hencky 切线，按 lagged_interval 可滞后更新）；
   - 计算残差和局部 H_ii^{-1}；
   - 用矩阵自由 PCG 解 δv；
   - 将增量提交到 grid_v_it。
6. 把收敛的网格速度投影为 grid_v_new。
7. lite_g2c、lite_c2p 完成粒子回传和本构历史更新。

## 6. 相对于传统 MPM 的主要改进

### 6.1 隐式求解复杂度不再随 PPC 成比例增长

传统隐式 MPM 的每次残差/Hessian-vector product 通常要走一遍 G2P2G，代价与粒子数、PPC 同时增长。MPM Lite 只在时间步的 unload/load 阶段访问粒子；Newton/PCG 的内部循环只访问活动中心和节点。因此在固定网格下，隐式求解部分主要由网格规模决定。

论文在其基准中报告：相对传统显式 MPM 最高约 1.88x，相对传统隐式 MPM 最高约 15.9x；在 24 PPC 的测试中给出最高 15.9x 的总运行时间加速。这个数字来自论文特定硬件、场景、材料和参数，不能直接当作本仓库在任意 GPU/CPU 上的保证。

### 6.2 用线性核换取紧支撑和明确边界语义

传统二次 B-spline 每个粒子在 3D 影响 27 个节点。这里拆成“粒子—中心”和“中心—节点”两跳，每跳 8 个邻居，并在节点上使用 Q1 形函数。这样有三个工程收益：

- 支持更小，核函数更易并行；
- 节点是 Q1 网格节点，边界条件具有 Kronecker-delta 风格的明确语义；
- 中心到节点是 gather，代码中不需要对节点动量做大规模原子 scatter。

论文在局部 C^(2,1) 光滑速度假设下证明，二跳传输相对于二次 B-spline APIC 的速度和速度梯度差异是 O(Δx^2)，并且误差常数与粒子占用数无关。这是“线性核不一定等于一阶质量”的关键论据。

### 6.3 应力守恒重采样和旋转自由重建

传统做法若直接平均 F， 会得到不客观的混合状态。这里保留

$$
\sum V\tau
$$

的广延守恒，并用应力反解中心 stretch，使中心 quadrature 既有可积的应力状态，又能放进标准 FEM 风格的增量势。这使得矩阵自由切线、PCG 和局部预条件器可以直接作用在网格自由度上。

### 6.4 更容易接入 FEM 风格求解器

固定中心 quadrature、Q1 梯度和节点自由度形成一个更新拉格朗日 hexahedral FEM 结构。理论上可以替换 PCG、接入 VBD、Newton、子空间或其他网格线性求解器；当前仓库实际实现的是矩阵自由 PCG 和块对角预条件器，不能把“可接入”误认为“所有求解器已经实现”。

### 6.5 粒子数和隐式工作区内存解耦

隐式临时数组主要按活动节点/中心分配，而不是按每次迭代的粒子 quadrature 分配。代价是中心上需要额外存储应力、体积、重建 F 和切线 scratch。代码还使用 32^3 block 的容量数组（例如 MAX_PTS、MAX_DOF、MAX_BLOCKS），所以理论上的解耦不等于内存无限扩展。

### 6.6 材料覆盖面较广，但要区分“接口支持”和“隐式切线支持”

Material 枚举包含 water、elastic、von Mises、Drucker–Prager、NACC、snow13 和 foam；lite_c2p_kernel 中有相应返回映射。论文还展示了雪、沙、水、金属、面团/奶油等场景。

但隐式算子 kernel_lite_implicit.py 的核心分支主要是 Constitutive.stvk 与 Constitutive.pressure。新材料要真正用于稳定的隐式求解，还必须提供正确的应力、∂P/∂F 或矩阵自由方向导数、状态更新以及正定/非正定处理，不能只在显式回传阶段加入一个返回映射函数。

## 7. 当前限制、风险和适用场景

### 7.1 各向同性限制

旋转自由的 F_c^base 依赖各向同性：能量和切线只看奇异值/伸长。如果是纤维、木材、层合材料、正交各向异性或带方向硬化的模型，旋转携带材料方向，不能直接丢弃。需要额外传输方向、结构张量或完整的变形状态。

### 7.2 应力到 stretch 的反演可能病态

在极端压缩、极端拉伸、近不可压缩 λ/κ 很大或应力接近奇异时，τ→σ 的反演可能不唯一、病态或数值不稳定。当前代码使用奇异值下限、正定投影和局部块逆来缓解，但没有一个对任意本构自动成立的全局鲁棒性保证。

### 7.3 单元中心一点积分的精度边界

每个 voxel 主要使用一个固定 cell-center quadrature。论文说明传输会过滤经典 hourglass 模式，但这不等于一点积分没有误差。以下场景要谨慎：

- 梁、薄壳、薄片和弯曲主导问题；
- 单元内应力梯度很大或局部几何很薄；
- 需要高阶应力/位移精度的工程验证；
- 自由表面或界面附近只有很少粒子的区域。

这些情况下可能需要多个中心 quadrature、更高阶重采样、局部稳定化或网格细化。

### 7.4 隐式“可用”不等于“任意大步长稳定”

隐式方法通常允许比显式更大的 Δt，但当前实现仍受以下因素影响：

- alpha=1.0，线搜索被明确省略；
- PCG 假设矩阵适合其迭代结构，塑性、接触和非凸本构可能使系统非正定；
- node_Hii_inv 只是局部对角块预条件，不是全局 Hessian；
- maxiter=5000 是固定上限，失败时需要查看 error、残差和迭代日志；
- lagged_interval 会滞后更新本构切线，减少成本但可能降低 Newton 收敛速度；
- tikhonov 在当前 lite_implicit_iterations 调用中传入 0，极端状态下没有额外正则；
- 固定点塑性必须与隐式循环一致，否则会出现沙堆塌陷等非物理解。

因此应同时做时间步收敛、残差收敛、能量耗散和网格收敛检查，而不是只观察动画是否“看起来稳定”。

### 7.5 线性核和低 PPC 的噪声问题

隐式积分部分不随 PPC 增长，但粒子仍然负责：

- P2C/C2P 重采样；
- 位置推进；
- 本构和塑性历史更新；
- 自由表面/材料界面状态的采样。

所以总运行时间仍包含 O(N_p) 部分。PPC 过低时中心的 Vτ、质量和速度梯度会变得噪声更大；高频界面、细小物体和稀疏自由表面尤其明显。

### 7.6 接触、摩擦和混合物模型还比较简化

boundary_utils.py 提供 sticky/slippy 投影和移动半空间边界，demo 使用 voxelized mesh 或平面边界。当前实现不是完整的摩擦接触互补求解器，也没有一般刚体两向耦合、接触冲量、摩擦锥求解或碰撞检测管线。

多个材料可以共享一个节点速度场，但这是单速度混合物。相分离、多速度孔隙流、复杂流固耦合、非笛卡尔/自适应网格仍属于扩展方向。

### 7.7 代码工程化边界

- MPMSolver 当前显式/隐式接口断言只支持 3D；mpmlite2d.py 是独立示例，使用自己的一套 2D kernel。
- Demo 中关闭了 Warp backward；仓库没有现成的端到端伴随/自动微分验证。
- 固定容量 MAX_PTS、MAX_DOF、MAX_BLOCKS 需要按场景调整；超过容量需要额外扩容逻辑。
- numpy() 读取计数、最大速度和残差会造成设备同步，频繁调试打印会掩盖 GPU kernel 的真实吞吐。
- 当前仓库以 demo 为主，缺少系统的单元测试、解析解回归测试、跨 GPU/CPU 基准和自动收敛报告。

## 8. 和其他“最新”路线的关系

| 路线 | 主要优点 | 与本仓库的差别 |
| --- | --- | --- |
| 传统二次 B-spline/APIC 隐式 MPM | 成熟、平滑、图形学资料多 | 每次隐式迭代通常访问粒子，PPC 高时成本明显 |
| CK-MPM / implicit CK-MPM | 紧支撑、接触局部性好；2026 年已有隐式 CK-MPM 研究 | 仍以粒子 quadrature 为中心，重点是核设计和接触，不是把积分完全移到固定中心 |
| 工程隐式 MPM（如 Newton SolverImplicitMPM） | 面向刚性材料/完全塑性极限，GPU 友好 | 其文档强调无条件时间步稳定和 GPU rheology；算法架构、材料接口和目标场景不同 |
| MPM Lite | 固定中心 quadrature、线性核、应力守恒重采样、旋转自由 stretch、网格端矩阵自由积分 | 依赖各向同性本构和一点 quadrature，接触/各向异性/自适应网格覆盖较少 |
| 固定 Gauss/更新拉格朗日 FEM 混合 MPM | 网格积分和边界条件成熟 | 通常需要额外处理大变形拓扑变化；MPM Lite 保留粒子作为拓扑/历史载体 |

因此，“最新”应该按指标拆开：

- **方法新颖性**：MPM Lite 的“粒子只做状态载体、网格完成积分”很新颖。
- **隐式 MPM 论文发布时间**：它不是 2026 年唯一或最后出现的工作，implicit CK-MPM 等工作更晚。
- **公开可运行代码**：本仓库是明确的公开参考实现；其他项目可能更偏工程、材料力学或平台集成。
- **某一应用的最好选择**：取决于接触、各向异性、颗粒流、孔压、可微分和许可证要求，不能由发布时间单独决定。

## 9. 适用场景建议

比较适合：

- 大变形软体、塑性金属、雪、沙、泡沫和高 PPC 的图形学场景；
- 希望使用 GPU、固定笛卡尔网格和隐式大步长的动态模拟；
- 需要在多个材料之间共享网格速度，同时允许粒子发生拓扑变化；
- 研究线性核、固定 quadrature 和矩阵自由 FEM/MPM 耦合。

需要谨慎或优先考虑其他方法：

- 纤维/层合/正交各向异性材料；
- 薄壳、细梁、强弯曲和高精度接触压力；
- 多速度孔隙流、复杂摩擦接触和刚体双向耦合；
- 需要严格工程认证、强误差控制或成熟后处理的计算力学项目；
- 需要端到端自动微分的逆问题（当前仓库没有现成的隐式伴随实现）。

实践上，建议先从 wheel.py（刚性+塑性）、noodles.py（隐式挤压）和 snow.py（显式雪）分别做三组验证：减小 Δt 的时间步收敛、增加网格分辨率的空间收敛、改变 PPC 的噪声/性能曲线。记录 Newton/CG 迭代数、最大残差、动能/弹性能、线动量和角动量，而不要只比较渲染结果。

## 10. 参考资料

1. [本仓库 README 与代码](../README.md)
2. [MPM Lite 论文（arXiv v3，2026-02-20）](https://arxiv.org/html/2602.07853v3)
3. [MPM Lite 项目主页](https://mpmlite.github.io/)
4. [MPM Lite GitHub 仓库](https://github.com/f1shel/mpm-lite)
5. [An Implicit Compact-Kernel Material Point Method（2026-04-20）](https://arxiv.org/abs/2604.18917)
6. [Newton SolverImplicitMPM 文档](https://newton-physics.github.io/newton/1.3.0/api/_generated/newton.solvers.SolverImplicitMPM.html)
7. [Matter：开源工程 MPM 实现](https://github.com/larsblatny/matter)
