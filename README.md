# HairADMM

**面向三维发丝图的快速碰撞后处理**

HairADMM 用于处理发型迁移后的穿模问题。输入是已经迁移到目标人物上的发丝和目标人体
网格，输出是在尽量保持原发型的同时减少人体穿透的发丝。

它不负责生成或迁移发型，只负责迁移完成后的这一步优化。

## 1. 泛用 ADMM 在做什么？

对一个带约束的优化问题，ADMM 通常引入辅助变量 $Z$：

$$
\min_{X,Z} f(X)+g(Z),
\qquad AX=Z.
$$

然后反复执行：

$$
X^{k+1}
=\arg\min_X f(X)
+\frac{\rho}{2}\|AX-Z^k+U^k\|^2,
$$

$$
Z^{k+1}
=\mathrm{prox}_g(AX^{k+1}+U^k),
$$

$$
U^{k+1}
=U^k+AX^{k+1}-Z^{k+1}.
$$

泛用求解器需要面对各种不同问题，因此通常会：

- 把所有未知量打包成一个通用向量；
- 求解通用稀疏线性系统或 KKT 系统；
- 使用固定约束矩阵；
- 对标量区间或其他通用集合做投影。

OSQP 就是一个基于 ADMM 的通用 QP 求解器。因此，我们的方法并不是“用 ADMM
替代 QP”，而是针对发丝问题重新设计 ADMM 的变量、线性系统和碰撞约束。

历史 QP 对发生碰撞的顶点使用局部切平面约束：

$$
\min_X E_{\mathrm{hair}}(X),
\qquad
\mathbf n_i^\top(X_i-\mathbf q_i)\ge\varepsilon.
$$

其中 $\mathbf q_i$ 是人体表面点，$\mathbf n_i$ 是该处外法向。这个表达式只直接约束
顶点 $X_i$，并且依赖当前固定的局部平面。

## 2. 发丝问题有哪些可以利用的性质？

发型可以看成一个三维几何图：

- 同一根发丝上的相邻点组成发丝边；
- 不同发丝之间的 KNN 关系组成跨发丝边；
- 每个点都有 x、y、z 三个坐标，但三个坐标的图连接关系完全相同；
- 碰撞不仅可能发生在顶点，也可能发生在两个顶点之间的线段内部。

这些性质意味着我们没有必要把它当成一个完全泛用的 QP。

需要保持的发型目标可以概括为

$$
\begin{aligned}
E_{\mathrm{hair}}(X)
= {}&
\frac{w_{\mathrm{fid}}}{2}
\sum_i\omega_i\|X_i-X_i^{\mathrm{init}}\|_2^2\\
&+\frac{w_{\mathrm{knn}}}{2}
\|LX-\Delta^{\mathrm{src}}\|_W^2\\
&+\frac{w_{\mathrm{dir}}}{2}
\sum_{(i,j)}
\left\|
\frac{X_j-X_i}{\|X_j-X_i\|_2}
-d_{ij}^{\mathrm{src}}
\right\|_2^2 .
\end{aligned}
$$

三项分别保持 Init 位置、跨发丝 KNN 结构和源发丝方向。碰撞处理是在这个共同目标之外
增加的约束，而不是用逐点投影替代整个发型目标。

### 性质一：三个坐标共享同一个图结构

经过当前 outer 迭代的二次化后，x-update 可以写成

$$
(H+\rho C^\top C)X
=B+\rho C^\top(Z-U).
$$

其中 $X\in\mathbb R^{N\times3}$。

同一个 $N\times N$ 稀疏矩阵同时用于 x、y、z 三个坐标，只是右端项不同。这样不需要
把问题扩成一个不透明的 $3N$ 维通用系统。

### 性质二：Guide 和 Normal 的耦合方式不同

Guide 发丝决定整体形变，需要求解带 KNN 耦合的图系统；Normal 发丝主要跟随已经求出
的 Guide，其系统结构更简单。

Guide 的跨发丝关系写成

$$
(LX^G)_i
=\sum_{j\in\mathcal N(i)}a_{ij}(X_i^G-X_j^G)
\approx\Delta_i^{\mathrm{src}}.
$$

Guide 求出后，Normal 点获得逐点目标

$$
T_i
=\sum_{j\in\mathcal N(i)}a_{ij}\widehat X_j^G
+\Delta_i^{\mathrm{src}},
\qquad X_i^N\approx T_i.
$$

因此我们采用：

- Guide：使用带预条件和 warm start 的 CG；
- Normal：每个 outer 只分解一次矩阵，在所有 inner ADMM 迭代和三个坐标间复用。

### 性质三：碰撞约束本质上是三维几何约束

传统 QP 对顶点建立局部切平面约束，再把标量结果限制在某个区间内。HairADMM 中，
本项目中的 $Z$ 直接保存发丝采样点的三维位置：

$$
CX=Z,\qquad Z\in\Omega_{1\mathrm{mm}}.
$$

其中 $\Omega_{1\mathrm{mm}}$ 表示距离人体表面至少 1 mm 的外部区域。

令

$$
Y=CX^{k+1}+U^k,\qquad m=1\text{ mm}.
$$

若 $\phi(Y_r)$ 是人体外为正的有符号距离，$\mathbf n(Y_r)$ 是外法向，则 z-update
可以概括为

$$
Z_r^{k+1}
=
\begin{cases}
Y_r, & \phi(Y_r)\ge m,\\
Y_r+\bigl(m-\phi(Y_r)\bigr)\mathbf n(Y_r),
& \phi(Y_r)<m.
\end{cases}
$$

也就是说，安全点保持不变，不安全点被推到人体表面外 1 mm。

### 性质四：只约束顶点不能发现所有线段穿模

两个端点都在人体外时，它们之间的线段仍可能穿过人体。为此，$C$ 不只选择发丝顶点，
还计算线段内部的仿射采样点：

$$
(CX)_r=(1-\alpha)X_i+\alpha X_j.
$$

对长度为 $\ell_{ij}$ 的线段，均匀采样数由

$$
n_{ij}
=\left\lceil\frac{\ell_{ij}}{2\text{ mm}}\right\rceil,
\qquad
\alpha_q=\frac{q}{n_{ij}},
\quad q=1,\ldots,n_{ij}-1.
$$

确定，所以相邻检查位置最多相隔 2 mm。

若 Embree 在 $\alpha_{\mathrm{hit}}$ 处检测到残余交点，下一轮额外加入

$$
\mathrm{clip}
\left(
\alpha_{\mathrm{hit}}+\delta,\ 0.02,\ 0.98
\right),
\qquad
\delta\in\{-0.10,0,0.10\}.
$$

这些比例对应交点及其左右邻域，生成新的 $C$ 行后继续求解。

## 3. 我们对泛用 ADMM 做了什么优化？

总体上，我们把泛用 ADMM 改成了一个专门服务于三维发丝图的求解器：

完整分裂问题为

$$
\min_{X,Z}
E_{\mathrm{hair}}(X)+I_{\Omega_{1\mathrm{mm}}}(Z),
\qquad CX=Z.
$$

$E_{\mathrm{hair}}$ 由 x-update 处理，人体外部约束由 z-update 处理，$U$ 负责累积
$CX$ 与 $Z$ 的不一致。

| 泛用 ADMM / OSQP | HairADMM |
| --- | --- |
| 将 xyz 打包成通用变量 | 三个坐标共享同一个图稀疏矩阵 |
| 求解通用 KKT 或 QP 系统 | Guide 使用 PCG，Normal 复用稀疏分解 |
| 对标量约束区间做投影 | 对三维发丝采样点做人体外部投影 |
| 约束矩阵通常固定 | 根据残余交点动态扩充采样矩阵 $C$ |
| 历史实现主要约束顶点 | 同时约束顶点和线段内部位置 |

算法流程为：

~~~text
输入迁移后的发丝 Init
        ↓
求解 Guide 发丝
  x-update：图结构稀疏线性系统
  z-update：投影到人体外 1 mm
  u-update：累计 x 与 z 的差异
        ↓
用 Guide 生成 Normal 的目标位置
        ↓
求解 Normal 发丝
        ↓
Embree 检测残余线段交点
        ↓
如有必要，在交点附近增加约束并继续求解
        ↓
输出碰撞减少后的发丝
~~~

## 4. 理论时间与空间复杂度

用下面的符号描述单帧复杂度：

- $N_G,N_N$：Guide 和 Normal 的自由发丝点数；
- $M_G,M_N$：Guide 和 Normal 的碰撞采样行数；
- $F$：人体三角面数量；
- $E_G,E_N$：参与 Embree 检测的发丝线段数；
- $R$：outer 次数，$I$：每个 outer 的 inner ADMM 次数；
- $J_G$：Guide 中 PCG 的迭代次数；
- $Q_G,Q_N$：一次 z-update 中真正执行精确 SDF 查询的 active 样本数；
- $S_G=\mathrm{nnz}(H_G+\rho C_G^\top C_G)$。

### 4.1 泛用 QP/OSQP

如果把三个坐标打包，通用 QP 的变量规模约为 $3N$，KKT 系统规模约为

$$
D=3N+M.
$$

设 $\mathcal F_{\mathrm{KKT}}(D)$ 是稀疏 KKT 分解成本，
$S_{\mathrm{KKT}}$ 是分解后一次回代的非零量，则每次重新建立 QP 的成本可写成

$$
T_{\mathrm{generic}}
=\mathcal O\!\left(
\mathcal F_{\mathrm{KKT}}(D)
+I_{\mathrm{osqp}}S_{\mathrm{KKT}}
\right).
$$

稀疏分解的实际成本取决于图结构和 fill-in；最坏情况下时间为 $\mathcal O(D^3)$，
空间为 $\mathcal O(D^2)$。这不是说 OSQP 在本项目中一定达到最坏情况，而是通用求解器不能利用
HairADMM 后面列出的特殊结构。

### 4.2 HairADMM 的 Guide

Guide 的三个坐标共享同一个 $N_G\times N_G$ 稀疏矩阵。一次 PCG 迭代的主要成本为
$\mathcal O(S_G)$，因此

$$
\begin{aligned}
T_G
=\mathcal O\!\Bigl(
R\bigl[
S_G
+I(3J_GS_G+M_G+Q_G\log F)
+E_G\log F
\bigr]
\Bigr).
\end{aligned}
$$

其中：

- $3J_GS_G$ 是三个坐标的 PCG；
- $M_G$ 是 $C_GX$、$C_G^\top Z$ 和 u-update；
- $Q_G\log F$ 是复用人体 AABB 树后的 active SDF 查询；
- $E_G\log F$ 是 Embree 线段查询的平均输出敏感成本。

### 4.3 HairADMM 的 Normal

当前 one-step 版本没有 Edge 和时序耦合。Normal 的基础目标是逐点目标，方向项和线段
采样只连接同一根发丝上的相邻点，因此其矩阵由彼此独立、带宽受限的发丝链组成。记

$$
S_N=\mathcal O(N_N+M_N).
$$

每个 outer 只分解一次，三个坐标和全部 inner 迭代复用该分解：

$$
T_N
=\mathcal O\!\Bigl(
R\bigl[
S_N
+I(3S_N+M_N+Q_N\log F)
+E_N\log F
\bigr]
\Bigr).
$$

### 4.4 总复杂度

令 $M=M_G+M_N$、$Q=Q_G+Q_N$、$E=E_G+E_N$。HairADMM 单帧的主要时间复杂度为

$$
\boxed{
T_{\mathrm{HairADMM}}
=\mathcal O\!\left(
RI J_GS_G
+RI S_N
+RI(M+Q\log F)
+RE\log F
\right)
}
$$

空间复杂度为

$$
\boxed{
\mathcal M_{\mathrm{HairADMM}}
=\mathcal O(S_G+S_N+M+F)
}
$$

KNN 数量、outer/inner 次数、PCG 次数和单位长度采样密度固定时，
$S_G=\mathcal O(N_G+M_G)$，HairADMM 对发丝点和约束数量表现为近线性增长；人体查询还带有
AABB/BVH 的平均 $\mathcal O(\log F)$ 因子。这里的“近线性”依赖 PCG 迭代数保持稳定，不能
解释为无条件的最坏情况线性保证。

| 方法/阶段 | 被求解的结构 | 主要时间项 | 主要空间项 |
| --- | --- | --- | --- |
| 泛用 QP/OSQP | 规模约为 $D=3N+M$ 的稀疏 KKT 系统 | $\mathcal F_{\mathrm{KKT}}(D)+I_{\mathrm{osqp}}S_{\mathrm{KKT}}$ | 取决于 KKT 分解后的 fill-in；最坏 $\mathcal O(D^2)$ |
| HairADMM Guide x-update | 三个坐标共享同一个 $N_G\times N_G$ 稀疏系统 | $\mathcal O(3RIJ_GS_G)$ | $\mathcal O(S_G+M_G)$ |
| HairADMM Normal x-update | 彼此独立、带宽受限的发丝链 | $\mathcal O(RI S_N)$ | $\mathcal O(S_N+M_N)$ |
| HairADMM 几何查询 | active SDF 查询与 Embree 线段检测 | 平均 $\mathcal O(RIQ\log F+RE\log F)$ | $\mathcal O(F+M)$ |

## 5. 当前碰撞结果（纯数值）

保留实验包含 7 个发型序列，共 2,702 帧：

| 指标 | Init | 历史顶点 QP | HairADMM | 相对 QP 减少 |
| --- | ---: | ---: | ---: | ---: |
| 穿入人体的发丝点数 | 1,275 | 442 | **15** | **96.61%** |
| 线段与人体相交数 | 97,490 | 31,598 | **127** | **99.60%** |

运行时间暂不作为当前发布结论，后续将在统一硬件、输入、线程数和计时边界下重新评估。

## 6. 快速运行

创建环境：

~~~bash
conda env create -f environment.yml
conda activate hairadmm
~~~

运行公开入口：

~~~bash
python tools/run_hair_admm.py \
  --objective_tensor_dir /path/to/problem_bundles \
  --output_dir /path/to/output
~~~

运行不包含私有资产的玩具示例：

~~~bash
python examples/toy_case/run_demo.py
~~~

运行测试：

~~~bash
python -m unittest discover -s tests -v
~~~

## 7. 输入与输出

每一帧输入为一个 frame_XXXX.npz，主要包含：

- 初始迁移发丝和源对齐发丝；
- 发丝拓扑、Guide 集合和 KNN；
- 局部权重和能量权重；
- 目标人体顶点和三角面。

输出包含优化后的 NPZ 和 OBJ 发丝文件。

同时查看人体和发丝：

~~~bash
cd tools/web
python -m http.server 8888
~~~

然后打开：

~~~text
http://localhost:8888/viewer.html?manifest=frames_toy_case.json
~~~

## 8. 主要文件

~~~text
configs/one_step.json                 算法配置
hairs_adaption/qp_objective_bundle.py 输入数据格式
tools/run_hair_admm.py                公开运行入口
tools/solve_admm_qp_aligned.py        Guide/Normal ADMM 求解器
examples/toy_case/                    玩具示例
tools/web/                            人体与发丝可视化
results/                              实验结果
~~~

## 许可证

代码以 [MIT License](LICENSE)发布。数据集、人体和发型资产不包含在仓库中。
