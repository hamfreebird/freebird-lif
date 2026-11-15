# freebird-lif 第二阶段：LIF 变体在 SHD 上的对比，与 Arc GPU 训练

第一阶段的成果（MNIST 16×16，98.97%，CPU 推理 0.118 ms/样本）已归档至
`docs/PLAN-v1-lif-mnist.md` 与 `H:/Dev/archive/freebird-lif-v1-20261115.tar.gz`。

本阶段回答两个问题：

- **RQ1**：在更复杂的时序任务上，LIF 的生物学变体相对基础 LIF 究竟带来多少收益？
- **RQ2**：如何在本机 Intel Arc A750 上用 PyTorch 完成这些模型的训练？

---

## 1. 环境约束

| 探测项 | 实测结果 | 影响 |
|---|---|---|
| GPU | **Intel Arc A750**，驱动 32.0.101.8864 | 需走 XPU 路径，非 CUDA |
| 匹配轮子 | `torch-2.15.0+xpu-cp312-cp312-win_amd64.whl`（781 MB） | 官方索引存在，Python 3.12 匹配 |
| CPU | 12 逻辑核 | 作为回退路径 |
| MSVC | **无**（仅 MinGW） | **`torch.compile` 在 Windows XPU 上不可用**（需 MSVC + Windows SDK） |
| Python | 3.12.4 | 与轮子 cp312 匹配 |
| 磁盘 | 312 GB 可用 | 充足 |

**关键事实：Intel Extension for PyTorch（IPEX）已于 2026 年 3 月 EOL。**
Intel 官方要求改用原生 `torch.xpu`，CPU/GPU 支持已上游进 PyTorch 主干。
因此本阶段不引入 IPEX，只装 `--index-url https://download.pytorch.org/whl/xpu` 的官方轮子。

驱动的 A-Series 最低要求是 32.0.101.6314，本机 32.0.101.8864 满足。

---

## 2. 数据集 SHD（Spiking Heidelberg Digits）

| 项 | 值 |
|---|---|
| 类别数 | **20**（德语 + 英语的 0–9） |
| 样本 | 8,156 训练 / 2,264 测试 |
| 输入通道 | 700（Lauscher 人工耳蜗模型） |
| 格式 | HDF5，每样本含 `spikes` / `times` / `units` |
| 时长 | 约 1.4 s，变长 |
| 许可 | CC BY 4.0 |
| 下载 | `https://zenkelab.org/datasets/shd/shd_train.h5.gz`（125 MB）、`shd_test.h5.gz`（36 MB） |
| 镜像 | `https://ieee-dataport.org/open-access/heidelberg-spiking-datasets` |
| 引用 | Cramer et al., IEEE TNNLS 33:2744–2757, 2022 |

**SHD 没有官方验证集。** 期刊允许在测试集上早停，但这是数据泄漏。本阶段的做法：
从 8,156 条训练样本中划出 1,000 条作验证集，测试集全程隔离。
代价是与文献数字不完全可比——这一点会在报告中明确标注。

### 2.1 预处理

HDF5 中每个样本是稀疏的 `(time, unit)` 事件列表，需转为稠密张量 `[T, 700]`。

- 分箱：`dt = 4 ms` → `T = 350`；`dt = 8 ms` → `T = 175`。文献常用 100–250 步。
- 主设定 **T = 100**，M0 阶段实测 `T ∈ {50, 100, 250}` 的精度-耗时曲线后定稿。
- 分箱后按最大时间长度零填充（SHD 是变长序列，末尾填充不影响因果性）。

---

## 3. 六个神经元模型

这是本阶段的核心内容。所有模型放在统一接口下，**只有神经元方程不同**。

### 3.1 LIF（基线）

```
u[t] = α·(u[t-1] - u_rest) + u_rest + (1-α)·I[t]
s[t] = H(u[t] - θ)
若 s[t]=1 则 u[t] ← u_rest            (硬重置)
α = exp(-dt/τ_u)
```
参数量：0（α、θ 为固定超参）。每个神经元每步约 3 次算术操作。

### 3.2 ALIF（自适应阈值，Bellec et al. 2018）

```
u[t] = α·u[t-1] + (1-α)·I[t] - θ·s[t-1]
b[t] = ρ·b[t-1] + (1-ρ)·s[t-1]
s[t] = H(u[t] - (θ + β·b[t]))          (阈值随发放历史抬高)
```
新增状态：`b`。新增超参：`ρ`、`β`（或设为可学习）。约 5 次操作/步。

### 3.3 AdLIF（自适应电流，Bittar & Garner 2022）

```
u[t] = α·u[t-1] + (1-α)·(I[t] - w[t-1]) - θ·s[t-1]
w[t] = β·w[t-1] + (1-β)·a·u[t-1] + b·s[t-1]
s[t] = H(u[t] - θ)
```
新增状态：`w`。四个**可学习**神经元参数：`α, β, a, b`（`cAdLIF` 约束 `a,b ≥ 0`）。
约 6 次操作/步。

**这是本阶段最重要的对照。** 文献在 SHD 上给出：
同架构 3 层 128 单元，LIF 87.04% vs AdLIF **93.06%**（无 recurrent），
加 recurrent 后 RadLIF 达 94.62%。也就是说**自适应机制单独贡献约 6 个百分点**，
超过本文要考察的其他所有变体。若本阶段复现不出这个量级的差距，说明实验设置有误。

### 3.4 GLIF（门控 LIF，Yao et al., NeurIPS 2022）

在三种神经元行为上各引入一个可学习门，做两个"原语"的凸组合：

```
U[t] = L[t] + I[t] + F[t] ⊙ S[t-1]
L[t] = G_α(U[t-1]; τ_lin, τ_exp)      泄露：线性衰减 vs 指数衰减
I[t] = G_β(C[t]; g^t, 1)              积分：动态 vs 静态输入增益
F[t] = G_γ(L_exp[t]; V_re, 0)         重置：软重置 vs 硬重置
S[t] = H(U[t] - V_th)
G(x; a, b) = σ(γ_gate)·a + (1-σ(γ_gate))·b    γ_gate 可学习
```
新增可学习参数：每层 3 个门控标量（逐神经元或逐层，按原文设定为逐层）。
约 8 次操作/步（含 3 次 sigmoid）。

参考：`github.com/Ikarosy/Gated-LIF`（官方实现，用于对齐）。

### 3.5 POLARA（极化感知，Lai & Cao, AAAI 2026）

这是六个里唯一一个**我尚未拿到完整方程**的。从摘要可确认的设计意图：

- 用解析定义的膜电位动力学分别建模**去极化、复极化、超极化**三个阶段
- 将膜电位约束在生理电压范围内，从而避免梯度消失/爆炸
- 前向与反向路径直接给出，**不依赖可学习门，也不依赖替代梯度调参**
- 声称在深层架构上相对 LIF 有稳定增益

**待办（M1 首要）**：从 AAAI 2026 论文集获取原文 PDF（`ojs.aaai.org` 文章 39435，
AAAI 40(27):22734–22741），提取精确方程后再实现。**在拿到原文前不写任何近似版本**——
猜测方程会让整组对比失去意义。

### 3.6 多房室模型（MC-LIF / DH-LIF）

单个神经元含胞体 + K 个树突房室，每个房室有独立的膜时间常数：

```
d_k[t] = α_k·d_k[t-1] + (1-α_k)·I_k[t]        k = 1..K
u[t]   = β·u[t-1] + (1-β)·(Σ_k g_k·d_k[t]) - θ·s[t-1]
s[t]   = H(u[t] - θ)
```
`α_k` 可学习且异质初始化（这是 DH-LIF 的关键：不同树突分支记忆不同时间尺度）。
K=2 为主设定，K=1 退化为带输入滤波的 LIF（本身就是一个有意义的消融点）。

参考：DH-LIF（Nature Communications 2024，在 SHD/SSC 上有结果）；
MC-LIF / MC-adLIF（JKU 学位论文，SHD/SSC/QTDB，含 Loihi 2 部署）。

---

## 4. 实验设计

### 4.1 控制变量

唯一变量是神经元方程。固定：

| 项 | 值 |
|---|---|
| 架构 | 700 → 256 → 256 → 20，全连接，无 recurrent |
| 读出 | 输出层膜电位对时间求和 → softmax（与 cAdLIF 论文一致） |
| T | 100 步（M0 定稿） |
| 优化器 | AdamW，lr 2e-3，cosine 退火，batch 128 |
| epoch | 60（M0 实测耗时后可能调整） |
| 替代梯度 | fast-sigmoid，α=2（与 v1 一致） |
| 数据顺序 | 每个 seed 固定，跨变体共享同一顺序 |

**为什么先不用 recurrent**：文献表明 recurrent 本身能带来 15+ 个百分点（Cramer 基线
RSNN 71.4% vs 单层前馈 48.1%），它会淹没神经元模型的差异。
先用纯前馈把神经元的贡献隔离出来，再在 M3.5 用最优两个变体加 recurrent 验证结论是否成立。

### 4.2 统计功效——本阶段最容易出错的地方

SHD 测试集只有 2,264 条。

| 真实精度 | 二项标准误 | 95% 置信区间半宽 |
|---|---|---|
| 87% | 0.71% | **±1.39%** |
| 93% | 0.54% | **±1.05%** |

**结论：单次运行的精度差异小于约 1.5 个百分点时，不可区分。**

大量已发表的 SHD 对比只报单次结果，其 1 个点的差距很可能只是噪声。本阶段的硬性要求：

- 每个变体 **≥ 3 个 seed**，报 mean ± std
- 跨变体做配对比较（同 seed 同数据顺序），用配对 t 检验或 Wilcoxon
- 报告中同时给出效应量与置信区间，不只给均值

### 4.3 成本维度

变体不是免费的。必须同时报：

| 指标 | 计算方式 |
|---|---|
| 训练参数量 | `sum(p.numel())` |
| 神经元状态数 | 每神经元每步的浮点状态个数（LIF=1，AdLIF=2，MC-LIF=1+K） |
| 算子数/神经元/步 | 逐式统计算术操作（含 sigmoid 计为多次） |
| SOP/推理 | 分析式计算：`T × Σ_layers (活跃输入数 × 输出数)` |

一个"精度 +1% 但神经元成本 2×"的变体是坏交易。这个轴在多数论文里被省略，
本阶段要明确给出来。

---

## 5. Arc A750 训练方案

### 5.1 已验证的事实

- 官方 xpu 索引存在 `torch-2.15.0+xpu-cp312-cp312-win_amd64.whl`（781 MB），
  与本机 Python 3.12、以及 v1 使用的 torch 版本号一致
- IPEX 已 EOL，不需要也不应安装
- `torch.compile` 在 Windows XPU 上需要 MSVC + Windows SDK；**本机没有 MSVC，
  因此只能走 eager 模式**。这是一个明确的性能上限，不可绕过
- 依赖链会一并安装 `intel-sycl-rt`、`intel-opencl-rt`、`intel-openmp` 等运行时

### 5.2 落地步骤

```bash
# 1. 建环境（不要复用 v1 的 CPU 环境）
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
.venv/Scripts/python -m pip install numpy h5py pytest

# 2. 验证设备（M0 的第一道闸）
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"

# 3. 微基准：确认 XPU 相对 CPU 有实际收益
python scripts/bench_xpu.py
```

### 5.3 必须验证的三件事（M0 闸门）

1. **设备可达**：`torch.xpu.is_available()` 为 True，能读到设备属性
2. **算子覆盖**：本阶段用到的算子（`linear`、`matmul`、`sigmoid`、`softmax`、
   `cross_entropy`、`max_pool1d`）无静默回退。检测方式：用
   `torch.profiler` 或 `PYTORCH_ENABLE_XPU_FALLBACK` 日志确认无 CPU 回退
3. **有实际加速**：同模型同 batch 下，XPU 单 epoch 耗时显著低于 CPU。
   若加速比 < 2×，本阶段应按 CPU 规划，GPU 只作记录

### 5.3.1 安装成本实测（带宽是真实约束）

搭建环境时实测：**轮子下载速度约 166 KB/s**（本机代理限速）。
781 MB 的 torch 轮子需约 80 分钟，加上 `intel-sycl-rt`、`intel-opencl-rt`、
`intel-openmp` 等依赖，首次搭建预计 **1.5–2 小时**。

这不是技术障碍而是带宽约束，但它决定了 M0 的排期：第一次搭环境要预留 2 小时，
之后复用 `.venv` 即可。若中途中断，pip 缓存（约 3.4 GB）会保留已下载部分，
重试不必从头开始。

**因此设备级验证被列为 M0 的第一步，而非本计划的前置条件。**
截至本文档定稿，已确认的是轮子存在且版本/Python 匹配
（`torch-2.15.0+xpu-cp312-cp312-win_amd64.whl`，Content-Length 781,337,330）；
`torch.xpu.is_available()` 的实际返回值待 M0 首步给出。

若 M0 发现 XPU 不可用，**不影响 RQ1 的科学价值**——六个变体的对比可以在 CPU 上完成，
只是单次训练从 ~40 分钟变成数小时，需相应缩减 seed 数或模型宽度。

### 5.4 预期与风险

Arc A750 的 FP32 峰值约 8–10 TFLOPS（28 个 Xe 核心）。

粗估单次训练：前向 MACs/样本 = 100 × (700×256 + 256×256 + 256×20) ≈ 25.0 M，
反向按 2× 计 → 75 M MACs/样本；每 epoch 8,156 样本 ≈ 612 GMACs；
60 epoch ≈ 37 TMACs ≈ 73 TFLOPs。按 2 TFLOPS 有效算力 → 约 37 s/epoch，
单次训练约 40 分钟。

但 SNN 训练是 **T 步串行展开**，实际会被大量小算子启动开销拖慢，2–3× 劣化是常见的。
因此留出 1–1.5 h/次 的预算：6 变体 × 3 seed ≈ **18–27 GPU 小时**。

**这是本阶段最大的成本项，也是唯一可能迫使缩减范围的约束。**

主要风险（按发生概率排序）：

| 风险 | 概率 | 应对 |
|---|---|---|
| XPU 检测失败或驱动不兼容 | 中 | 回退 CPU；A750 驱动 32.0.101.8864 高于最低要求，但仍可能与 SYCL 运行时不匹配 |
| 部分算子静默回退 CPU，加速比不及预期 | 中高 | M0 用 profiler 核查；必要时手写融合算子或降低 T |
| eager 模式无法用 torch.compile，吞吐受限 | 确定 | 接受；或用 `torch.jit.script` 尝试部分融合（XPU 支持有限） |
| 显存不足（A750 8 GB） | 低 | batch 128、模型 < 1M 参数，余量充足 |

---

## 6. 目录结构与复用

v1 的以下资产直接复用，不重写：

- `freebird_lif/neuron.py` 的替代梯度实现（`_Spike` autograd Function）
- `freebird_lif/train.py` 的训练循环骨架与检查点约定
- `freebird_lif/data.py` 的划分/增强思路
- v1 确立的基准纪律：空载测量、预热、多 seed

```
freebird-lif/
├── docs/PLAN-v1-lif-mnist.md      第一阶段计划（归档）
├── docs/README-v1.md              第一阶段说明（归档）
├── freebird_lif/
│   ├── neuron.py                  v1 LIF（保留）
│   ├── variants/
│   │   ├── base.py                SpikingNeuron 接口：init_state / step / cost
│   │   ├── lif.py  alif.py  adlif.py  glif.py  polara.py  mc_lif.py
│   ├── shd/
│   │   ├── data.py                下载、HDF5 解析、分箱、训练/验证切分
│   │   ├── net.py                 神经元可插拔的 SNN
│   │   └── train.py               SHD 训练循环
│   └── device.py                  设备选择、XPU 兼容层、回退检测
├── scripts/
│   ├── fetch_shd.py               下载与缓存
│   ├── bench_xpu.py               XPU vs CPU 微基准
│   └── sweep_variants.py          变体 × seed 扫描
├── tests/                         神经元方程、梯度、参数计数
└── benchmarks/                    report-v2.md
```

`variants/base.py` 的接口约束：

```python
class SpikingNeuron(nn.Module):
    n_state: int                                  # 每神经元状态数
    def init_state(self, *shape, device, dtype): ...
    def step(self, x, state): ...                 # 返回 (spike, new_state)
    def op_count(self) -> int                     # 每神经元每步算术操作数
```

---

## 7. 里程碑

| 阶段 | 内容 | 完成判据 |
|---|---|---|
| M0 | XPU 环境验证、SHD 下载与分箱、T 扫描、微基准 | `torch.xpu.is_available()` 为真且实测加速 > 2×；数据形状 `[8156, T, 700]` 校验通过 |
| M1 | 六个神经元实现 + 单元测试 | 每个变体：动力学对拍（K=1 时 MC-LIF 退化为 LIF）、梯度有限、参数计数与论文一致 |
| M2 | LIF 基线跑通 SHD | 测试精度 ≥ 85%（文献 FF 3×128 LIF 为 87.04%），训练曲线无异常 |
| M3 | 变体扫描，6 × 3 seeds | 全部完成，产出 mean ± std 表 |
| M4 | 分析 | 精度-成本散点、配对检验、recurrent 复核 |
| M5 | 报告 | `benchmarks/report-v2.md` |

### 验收标准

| 指标 | 目标 |
|---|---|
| 六个变体全部实现并通过单元测试 | 必须 |
| LIF 基线测试精度 | ≥ 85% |
| 每变体 seed 数 | ≥ 3 |
| 报告含置信区间与配对检验 | 必须 |
| XPU 加速比（若可用） | ≥ 2× 相对 CPU |
| 结论明确回答"哪个变体值得用、代价多少" | 必须 |

### 预期结论（可被推翻）

基于文献，**先验判断**：AdLIF > ALIF ≈ MC-LIF > GLIF > LIF > POLARA（POLARA 可能被低估）。

AdLIF 的 6 个百分点优势是在同架构同数据下测出来的，可信度最高。
POLARA 是 2026 年新工作，独立复现少，本阶段的结论对它最有信息量。

**如果复现不出 AdLIF 的显著优势，优先怀疑自己的实现或训练配方，而不是文献。**

---

## 8. 风险登记

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| 1 | 测试集仅 2264 条，1.5 点以内差异不可分辨 | 结论可能模棱两可 | 3 seed + 配对检验；明确报告"无可分辨差异"而非强行排序 |
| 2 | POLARA 方程未获取 | 该变体无法实现 | M1 首日取原文；取不到则如实标注"未实现"，不用猜测版本凑数 |
| 3 | XPU 不可用或加速不足 | 训练时间不可控 | M0 闸门判定；回退 CPU 并缩减 seed 数 |
| 4 | 变体各自的超参未充分调优，比较不公平 | 结论失真 | 每个变体给相等的调参预算（固定 3 组 lr × 1 组结构）；调参过程记录 |
| 5 | 训练预算 18–27 GPU 小时 | 可能超期 | M0 实测单 epoch 耗时后重定 seed 数与 epoch 数，不预设 |
| 6 | 与文献数字不可直接比较（自建验证集） | 外部质疑 | 报告中明确标注差异，并单独给一组"用测试集早停"的数字作参照 |

---

## 9. 与第一阶段的关系

第一阶段解决的问题是"在极小输入（256 像素）下把 LIF 做到极致效率"，结论是：
16×16 分辨率天花板约 99%，稀疏加速受限于 17.6% 的发放密度，CPU 单样本 0.118 ms。

第二阶段换了问题：**在长时序任务上，神经元模型本身能带来多少能力上限的提升。**
输入从 256 维单帧变成 700×100 的时空序列，网络的瓶颈从"分辨率"转向"时间整合能力"。

两个阶段共享的东西：替代梯度实现、BN 折叠、基准纪律、对负面结果的记录习惯。
不共享的东西：v1 的稀疏 CPU 引擎对 SHD 不适用（输入稠密、T 上百步），
本阶段不做推理引擎优化，只做训练与能力评估。

---

## 10. 变更记录

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-11-15 | 初始版本 | 第二阶段启动，v1 归档 |
| 2026-11-15 | 补充 §5.3.1 安装成本实测：代理限速 166 KB/s，首次搭环境需 1.5–2 h；设备级验证下放至 M0 首步 | 带宽约束影响排期 |
