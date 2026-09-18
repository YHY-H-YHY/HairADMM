# 算法定位：面向三维发丝图的几何与时序感知 ADMM

本文档明确 HairADMM 的算法主张，并说明它与传统面向 QP 的 ADMM 求解器有何不同。

## 1. 被优化的对象是三维几何图

发型表示为发丝图

$$
G=\left(V,E_{\mathrm{strand}},E_{\mathrm{knn}}\right),
\qquad
X\in\mathbb{R}^{N\times 3}.
$$

- $V$ 是三维发丝采样点，所有点的位置组成矩阵 $X$；
- $E_{\mathrm{strand}}$ 连接同一根折线发丝上的相邻点；
- $E_{\mathrm{knn}}$ 连接不同发丝上的空间邻居。

优化目标是在保持 Init、跨发丝 KNN 结构和源发丝方向的同时，使发丝与目标人体保持
安全距离，并抑制逐帧独立修复产生的时序尖峰。本项目中的“三维网络”指这个嵌入三维
空间的几何图，不是神经网络。

## 2. 传统 QP/ADMM 基线

历史方法先固定非线性量和附近的人体切平面，再构造带逐顶点线性不等式的凸 QP。
CVXPY 把这个 QP 交给 OSQP，而 OSQP 本身已经是一种基于 ADMM 的求解器。

简化后的 OSQP 问题为

$$
\begin{aligned}
\min_x\quad & \frac{1}{2}x^\top P x+q^\top x,\\
\text{subject to}\quad & l\le Ax\le u.
\end{aligned}
$$

OSQP 引入标量约束副本，在通用 KKT 求解、投影到区间 $[l,u]$ 和对偶更新之间迭代。
历史代码对每个新的 QP 都会重新构造 CVXPY 问题和求解器设置。

这个基线在本任务中有两个局限：

1. 一条约束只处理一个发丝顶点，无法直接约束相邻顶点之间的线段内部；
2. 把 xyz 打包进通用 KKT 系统，没有显式利用三个坐标共享的标量图算子，也没有利用
   Guide 和 Normal 不同的耦合结构。

## 3. HairADMM 改变了哪些部分？

HairADMM 保留增广拉格朗日与 x/z/u 交替更新的基本思想，但改变了分裂变量、近端映射、
约束算子和线性代数实现。

### 3.1 图结构 x-update

outer 二次化之后，发型能量具有标量图 Hessian $H$。碰撞采样由标量仿射采样矩阵
$C$ 给出。x-update 为

$$
(H+\rho C^\top C)X=B+\rho C^\top(Z-U).
$$

同一个 $N\times N$ 稀疏矩阵用于 x、y、z 三个坐标列，而不是把所有坐标视为一个
不透明的 $3N$ 向量。

- 全局耦合的 Guide 图使用 warm start 的预条件 CG；
- 结构更简单的 Normal 图在每个 outer 中只做一次直接分解，并在所有 inner ADMM
  迭代和三个坐标右端项之间复用。

这是预期降低重复求解器成本的主要机制。

### 3.2 几何感知 z-update

分裂条件为

$$
CX=Z,
\qquad
Z\in\mathrm{exterior}(\mathrm{body},1\text{ mm}).
$$

$Z$ 的每一行都是一个三维位置，而不是标量约束值。z-update 查询人体有符号距离，并将
不安全的样本投影到人体外部。它用几何近端算子替代了对固定半空间标量值的区间裁剪。

### 3.3 感知线段的采样算子

$C$ 的行可以计算：

1. 有效非发根顶点；
2. $E_{\mathrm{strand}}$ 中的线段内部仿射点，确定性采样间距不超过 2 mm；
3. 残余精确线段—三角形交点本身及其邻域位置。

线段行的形式为

$$
(CX)_r=(1-\alpha)X_i+\alpha X_j.
$$

因此，即使 $X_i$ 和 $X_j$ 都在人体外，它仍然可以观察两者之间的线段穿模。

### 3.4 自适应约束更新

传统面向 QP 的 ADMM 通常在固定 $A$ 上求解。HairADMM 使用嵌套流程：

~~~text
outer：二次代理和约束生成
  inner：结构化 x/z/u ADMM
  使用 Embree 查询残余精确交点
  必要时向 C 增加交点邻域行
~~~

Embree 是几何检测器，不是求解后再独立执行的第二次修复。新增采样点仍回到同一个
人体外部 z-update 中处理。

### 3.5 Active 时序门禁

完整序列模式使用前后参考帧构造离散时间二阶差分

$$
A_i^t=\widehat X_i^{t-1}-2X_i^t+\widehat X_i^{t+1}.
$$

只有 $\|A_i^t\|_2$ 超过由 Init 序列运动幅度确定的阈值时，该点才加入时序 active set。
对应 z-update 将二阶差分投影到三维球 $\|Z_{T,i}\|_2\le r_i^t$。因此该机制针对的是
异常的帧间抖动，而不是无条件地平滑所有真实运动。单帧输入没有前后参考帧，自动退化为
纯几何 HairADMM。

## 4. 哪些内容不是创新？

以下技术本身都是已有方法，不应单独声称为本文发明：

- ADMM 和增广拉格朗日；
- 使用 OSQP 求解 QP；
- 有符号距离投影；
- 均匀线段采样；
- Embree 相交查询；
- 稀疏 CG 和稀疏矩阵分解。

本项目的算法主张是它们针对三维发丝图的耦合专门化：

> 在一个嵌套 HairADMM 中，联合使用坐标共享的图结构 x-update、三维人体外部集合
> z-update、由精确交点驱动的自适应采样算子，以及只处理异常二阶位移的 active 时序
> 门禁。

已有工作使用 ADMM 处理发丝物理仿真和接触。本文处理的是带 Init、KNN、源方向目标和
可选时序门禁的发型迁移后处理序列。因此文献表述应是“面向该任务的新 ADMM 构造”，
而不是“首次将 ADMM 用于发丝”。

## 5. 建议的论文贡献表述

较严谨的贡献陈述可以写成：

> 我们提出 HairADMM，一种用于三维发丝图碰撞优化的几何感知 ADMM。不同于传统的
> 逐顶点 QP/OSQP 流程，HairADMM 求解坐标共享的稀疏图系统，将三维顶点和线段样本
> 投影到人体外部集合，并根据残余精确交点动态扩充采样算子。该设计同时针对通用求解器
> 的重复开销、逐顶点约束遗漏的线段穿模，以及逐帧独立碰撞修复产生的时序尖峰。

只有在 [复现实验规范](reproducibility.md)中的受控实验完成后，才能进一步写入“更快”、
“减少线段穿模”和“减少帧间抖动”等定量结论。

## 6. 最相关参考文献

- Lu Yu, Zhong Ren, Youyi Zheng, Xiang Chen, and Kun Zhou. “Shape Adaptation
  for 3D Hairstyle Retargeting.” IEEE TVCG.
  https://doi.org/10.1109/TVCG.2024.3521003
- Bartolomeo Stellato et al. “OSQP: an Operator Splitting Solver for Quadratic
  Programs.” Mathematical Programming Computation, 2020.
  https://doi.org/10.1007/s12532-020-00179-2
- Gilles Daviet. “Interactive Hair Simulation on the GPU using ADMM.”
  SIGGRAPH 2023.
  https://research.nvidia.com/labs/prl/admm_hair/
- Gilles Daviet. “Simple and Scalable Frictional Contacts for Thin Nodal
  Objects.” ACM TOG, 2020.
  https://doi.org/10.1145/3386569.3392439
