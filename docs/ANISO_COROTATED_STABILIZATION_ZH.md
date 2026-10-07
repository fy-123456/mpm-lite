# 生产共旋稳定化：实现与验证

> 后续更新：[二次空间重建与材料转动模板](ANISO_QUADRATIC_RECONSTRUCTION_ZH.md)。已针对本文发现的 Q1 平方项缺失接入 `quadratic` / `material_quadratic`，显著改善测试梁朝向误差。本页保留旧 `corotated` 的公式和基线数据。

本轮新增可选 `--stabilization corotated`。保留旧模式与默认值，不把新模式未经验证地设为默认。代码见 [corotated.py](../engine/aniso_phase1/corotated.py)、[enhancements.py](../engine/aniso_phase1/enhancements.py)，实验入口见 [aniso_corotated.py](../benchmarks/aniso_corotated.py)。

## 实施记录

- 已接入 GPU/CPU 生产稀疏残差、精确解析 matvec 和 Gauss–Newton 正定近似；包括极分解旋转的一阶/二阶影响，内层没有粒子遍历、有限差分或自动微分 Tape。
- 在步间重建节点材料参考坐标，计算并冻结中心与 8 个 Gauss 点的材料梯度；不把参考单元公式直接套在旧空间梯度上。
- 原始共旋势能进入现有 Armijo 线搜索；PCG 曲率/残差检查与修改切线回退保留。
- 初始材料参考映射折叠/奇异时拒绝推进，试探 Gauss/中心变形非正 J 时拒绝该试探，避免静默退回旧稳定化。
- 账本新增 `stabilization_start_energy`、`stabilization_solve_delta`、`stabilization_rebuild_delta`。这三个字段用于拆分附加能的冻结步变化与重建变化，**不重复加入原有账本求和**。
- 生产算子的导数、对称性、旋转力协变和短程回归通过。绕 y 轴的粗网格总弹性能朝向变化从约 23.7% 降至 0.12%；绕 z 轴仍有约 23.9% 的固定网格重建/表示误差，详见下文。**不能将本轮描述为已经消除所有大转动能量误差。**

## 材料参考映射

沿用粒子携带的 $X_p,F_p$ 重建节点材料位置：

$$
\widehat X_i=\frac{\sum_p m_pw_{pi}[X_p+F_p^{-1}(x_i-x_p)]}{\sum_p m_pw_{pi}},\qquad u_i=x_i-\widehat X_i.
$$

记 $g_{qi}=\nabla_xN_i(q)$，在中心（q=0）和八个 Gauss 点分别计算

$$
J_{X,q}=\sum_i\widehat X_i\otimes g_{qi}=I-\sum_i u_i\otimes g_{qi},\quad
b_{qi}=J_{X,q}^{-T}g_{qi},\quad
\widetilde F_q^n=J_{X,q}^{-1}.
$$

随后在整个 Newton 过程中冻结 $b_{qi}$、$\widetilde F_q^n$、参考体积和方向矩，只更新

$$
\widetilde F_q(v)=\widetilde F_q^n+\Delta t\sum_i v_i\otimes b_{qi}.
$$

这里 $\widetilde F$ 专门来自节点参考映射；中心主材料能使用的 $F_c$ 仍来自粒子历史重采样。非均匀场景中二者一般不同，不能假称完全相等。均匀仿射变形下二者一致，各 Gauss 点的变形也相同。

## 势能、完整旋转导数与两种切线

用 $\widetilde F_c=RU$、$D_q=R^T(\widetilde F_q-\widetilde F_c)$ 构造

$$
E_{cor}=\frac12\sum_q w_qD_q:H_0:D_q,\qquad w_q=\eta V_c^0/8.
$$

$H_0$ 使用相同参考材料切线；四阶矩模式和粒子基线使用同一方向分布平均。为保证后述半正定近似，corotated 要求参考体积模量 $\lambda+2\mu/3\geq0$（$\mu,k_f\geq0$ 已由材料参数检查）。叠加刚体旋转后，固定材料梯度下 $\widetilde F_q'=Q\widetilde F_q,R'=QR$，所以 $D_q'=D_q$。这证明冻结映射中的客观性；它不证明笛卡尔网格重建后积分能完全不变。

以下每式省略 q 下标。设 $S=H_0[D]$，求解反对称矩阵 Z：

$$
UZ+ZU=DS^T-SD^T.
$$

完整势能的一阶导数为

$$
P_q=RS,\qquad P_c=R(Z-S),\qquad
r_i^{cor}=\Delta t\sum_qw_q(P_qb_{qi}+P_cb_{ci}).
$$

$RZ$ 即不能漏掉的极分解旋转导数项。代码用三维反对称矩阵的轴向量，将 Sylvester 方程化为 $(\operatorname{tr}U I-U)^{-1}$ 的 3×3 运算，避免重复奇异值附近的奇异值差分母。

对方向 $p$，令 $A=R^T\delta\widetilde F_c$，求

$$
U\Omega+\Omega U=A-A^T,\quad
\delta U=A-\Omega U,\quad
\delta D=R^T(\delta\widetilde F_q-\delta\widetilde F_c)-\Omega D,\quad
\delta S=H_0[\delta D].
$$

精确切线继续微分 Z 方程：

$$
U\delta Z+\delta ZU=\delta D S^T+D\delta S^T-\delta S D^T-S\delta D^T-\delta UZ-Z\delta U,
$$

$$
\delta P_q=R(\Omega S+\delta S),\qquad
\delta P_c=R[\Omega(Z-S)+\delta Z-\delta S].
$$

这是真正的精确 Hessian 作用，对称但不保证正定。回退时改为 $L^TH_0L$（$L=\partial D/\partial v$），仍包含完整的一阶旋转导数，但省略势能的剩余曲率项。该 Gauss–Newton 贡献半正定，与质量和正定近似材料项组合后供 PCG 使用；**它不是原始精确 Jacobian**。两种切线都使用同一个原始势能与精确残差。

## 生产算子与空间刚度验证

实际稀疏网格跨 8 个块，含固定和滑动边界；自由子空间 45 维。统计见 [operator.json](results/corotated/operator.json)，混合方向的同类检查还覆盖在单元测试中。

| 检查 | 结果 |
|---|---:|
| 原势能方向导数 vs 残差 | 相对差 $8.16\times10^{-10}$ |
| 残差有限差分 vs 精确解析 matvec | 相对差 $6.96\times10^{-11}$ |
| 精确/修改切线对称性误差 | 约 $1.6\times10^{-16}$ |
| 当前试探状态精确切线最小特征值 | $5.54\times10^{-7}$ |
| 同状态修改切线最小特征值 | $1.13\times10^{-6}$ |
| 固定材料映射下旋转能量相对误差 | 小于 $6.1\times10^{-14}$ |
| 固定材料映射下完整力旋转相对误差 | 小于 $9.1\times10^{-14}$ |

上述正定性数字只证明该状态的完整自由子空间谱，不代表任意变形下的精确切线都正定。边界处理继续是两侧投影 $QJQ$。新增测试还直接调用 `pcg_projected`，确认使用修改切线时仍对原势能做 Armijo 检查。

在未变形参考状态，$D=0$ 且精确/修改附加切线都退化为原来的补充积分刚度。两个网格的生产 matvec 与同一本构小应变全积分 Q1 参考一致：

| 网格 | 无稳定化零模态 | corotated 零模态 | 参考端位移 | 生产 matvec 相对差 |
|---|---:|---:|---:|---:|
| 17³ | 32 | 0 | −0.000569794 | $5.91\times10^{-16}$ |
| 33³ | 64 | 0 | −0.000606623 | $4.65\times10^{-16}$ |

端位移是该独立冻结刚度的静力解，不是有限变形动态梁的静态验收。均匀拉伸、剪切各 4 步，附加能约 $10^{-32}$；应力对照通过，新模式没有给均匀仿射场增加额外刚度。

## 精确旋转：修复的部分与剩余问题

使用上一轮完全相同的预弯曲梁（幅度 0.01）、精确旋转增量、物理尺寸和质量；每单元每轴 2 个粒子，17/33 网格分别 256/2048 粒子。所有节点速度规定，因此比较弹性能而不是要求含约束外功的总机械能守恒。

以下是采样全过程总弹性能相对初值的最大下降，包含中心材料能和附加能：

| 网格 / dt / 轴 | 旧 supplemental | 新 corotated |
|---|---:|---:|
| 17 / 0.01 / y | 23.696% | **0.1194%** |
| 33 / 0.01 / y | 上轮未测 | **0.0312%** |
| 17 / 0.01 / z | 23.888% | **23.904%** |
| 33 / 0.01 / z | 6.049% | **6.054%** |
| 17 / 0.005 / z | 23.888% | **23.904%** |

对 corotated，每一个冻结步的刚体旋转附加能变化 $|E_{stab,end}-E_{stab,start}|/E_{stab,start}<3.1\times10^{-14}$；按试探节点位置 $y_i=x_i+\Delta t v_i$ 计算的附加力矩约 $10^{-19}$。历史 F、纤维旋转仍正确。**剩余大变化发生在步间重新构造参考映射与积分模板时，而不是漏掉了旋转导数。**

仍沿用步初采样角度画能量曲线，避免把冻结采样相位误认为动态滞后。新账本将 `stabilization_solve_delta` 与 `stabilization_rebuild_delta` 分开，可以直接复核上述结论。

### 对 z 轴异常的进一步定位

为判断是否主要是粒子到节点历史插值不准，对 0° 和 45° 另外用预弯曲映射的**解析逆映射**直接生成节点材料坐标，绕过粒子历史平均，再执行相同 Q1 映射/积分。解析逆变换的回代误差约 $1.1\times10^{-16}$。

| 网格 / 节点参考坐标来源 | 0° 附加能 | 45° 附加能 |
|---|---:|---:|
| 17 / 粒子重建 | $5.876\times10^{-6}$ | $1.462\times10^{-7}$ |
| 17 / 解析逆映射 | $6.888\times10^{-6}$ | $1.241\times10^{-7}$ |
| 33 / 粒子重建 | $1.656\times10^{-6}$ | $2.977\times10^{-8}$ |
| 33 / 解析逆映射 | $1.722\times10^{-6}$ | $3.099\times10^{-8}$ |

即使给出解析节点参考位置，朝向依赖仍然很大。这排除了“单纯提高粒子历史插值精度就能解决它”的解释。

对应的空间表示原因可以从二维 Q1 的基底理解：它包含 $1,x,y,xy$，不包含 $x^2,y^2$。旋转坐标后，一个可表示的混合二次项可能变成

$$
xy=\frac12(\xi^2-\zeta^2)
$$

（符号随旋转约定变化）。纯平方项在每个 Q1 单元内被插值为沿该方向的线性函数，其梯度波动会丢失。因此，同一弯曲场相对固定网格转动后，中心外的可表示波动不同。共旋解决的是同一材料映射上叠加刚体转动的不一致，**无法补回重建空间本身没有的二次项**。粒子体积权重/中心材料平均也会随重分箱变化；本对照不宣称所有剩余总误差都只由单一项造成。

该异常根因已定位到固定网格表示/重建环节，未采用调整能量数值或人为旋转力的方式掩盖。下一步适合比较能再现二次弯曲的重建与随材料旋转的局部积分模板，保持原势能—残差—切线一致性，再检查静态零模态。当前模式仍为可选实验功能，不能推广成任意大变形的完整验收。

## 小变形梁释放和混合纤维回归

梁释放保持幅度 0.002、17 网格、FLIP=0.9、总时间 0.008 s：

| dt | corotated 总能量变化 | 旧 supplemental | 最小粒子 J |
|---|---:|---:|---:|
| 0.001 | −7.66865% | −7.668% | 0.999197 |
| 0.0005 | −4.44739% | −4.447% | 0.999183 |

两组均正常收敛，无几何爆炸。能量账本闭合误差小于 $2.5\times10^{-22}$。这说明本改动没有明显改变小变形梁响应，**也没有消除原有时间积分/传递耗散**。

交叉纤维拉伸继续采用 9 网格、dt=0.005、最大位移 0.002、单程加载 0.16 s、两个平滑循环（每个方案 128 步）：

- 中心四阶矩与粒子积分总反力曲线差 **0.9434%**。
- 两次加载功的中心—粒子差 **0.9281% / 0.9018%**。
- 对各自旧 supplemental，反力曲线变化约 $1.1$～$1.3\times10^{-6}$，即约 **0.0001%**。
- 两循环反力曲线重复差仍约 8.8%；没有据此宣称准静态、无耗散。

参数、CSV、原始状态、对照统计和图见 [结果目录](results/corotated/)。

![共旋模式的改善与剩余网格误差](results/corotated/summary.png)

## 求解器建议与代价

默认仍用 `pcg`：先尝试精确对称切线，检查曲率、真实线性残差与下降方向；必要时回退到正定近似材料切线＋共旋 Gauss–Newton，再对原势能线搜索。也新增 CLI `--linear-solver pcg_projected`，用于从第一轮就选择修改切线的对照。没有恢复使用面向旧非对称残差的 BiCGSTAB/GMRES 默认策略。

新增主要缓存为每中心 9 个 3×3 映射和 9×8 个参考梯度，双精度约 **2376 字节/中心**，另有共用小数组。已测梁场景全部 enhancement 数组约 697056 字节；这不是求解器总显存。诊断开启时新模式每步中位约 22～25 ms，包含同步及主机能量统计，不能作为公平无诊断性能结论。此轮没有为了速度省略旋转导数或缩减积分点。

## 回归与运行命令

CPU 全套 **79 项：77 通过，2 项 CUDA 检查跳过**；随后 GPU **7 项全部通过**，包含新增 5 项和被跳过的 2 项。材料参数半正定条件检查最后另行通过。实验包括 5 组旋转、2 组释放、2 组循环拉伸、2 组仿射场、两网格静态参考和一次解析重建定位。原始日志保存在结果目录。

Viser 对照回放通过 HTTP 200、自动播放和正常退出检查；未做浏览器截图级验收。CLI 已验证默认 PCG 共旋梁、修改切线共旋梁和粒子共旋拉伸。使用自动/低显存占用设备；磁盘守卫继续每步执行，末次系统约 18 GiB、数据盘约 36 GiB，未触发迁移或暂停。

直接运行共旋梁：

```bash
.venv/bin/python -m demos.aniso --scene beam --grid 17 --device auto \
  --dt .0005 --stabilization corotated --direction-model fourth_moment --play
```

查看 y 轴改善，以及仍存在的 z 轴误差（两条命令分别运行）：

```bash
.venv/bin/python -m demos.aniso_rotation --modes supplemental corotated --axis y --port 8081
.venv/bin/python -m demos.aniso_rotation --modes supplemental corotated --axis z --port 8081
```

打开 `http://127.0.0.1:8081`；远程运行需转发端口。回放是规定运动，形状相同为预期结果，应查看侧栏能量曲线。

共旋慢速混合纤维拉伸：

```bash
.venv/bin/python -m demos.aniso --scene tensile --grid 9 --device auto \
  --dt .005 --loading-time .16 --loading-speed .0125 --loading-cycles 2 \
  --smooth-loading --fiber-field crossed --direction-model fourth_moment \
  --stabilization corotated --play
```

粒子积分对照将 `--direction-model fourth_moment` 换为 `--quadrature particle`；主动使用修改切线可加 `--linear-solver pcg_projected`。

复现实验：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.aniso_corotated --device auto --part all
MPLCONFIGDIR=/tmp/mpm-lite-matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.aniso_corotated_report
```
