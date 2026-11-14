# freebird-lif

LIF（Leaky Integrate-and-Fire）脉冲神经网络，纯 CPU 环境下的 16×16 手写数字识别器，输出数字与置信度。

设计目标是两个约束同时成立：**尽可能高的推理计算效率**，以及一个**可用的分类器**（验收线 96.0%）。

完整技术方案见 [PLAN.md](PLAN.md)。本 README 只保留上手所需的信息。

## 设计要点

- **训练与推理解耦**：训练用 PyTorch（替代梯度 BPTT），推理侧完全剥离框架，仅依赖 `numpy` + `numba`
- **分层内核**：Conv1 走 im2col+sgemm（输入稠密），Conv2/FC1 走事件驱动稀疏内核，FC2 走稠密 sgemv
- **BN 折叠**：BatchNorm 在权重导出时折进卷积，推理侧零 BN 开销
- **温度校准**：置信度经温度缩放校准，附 ECE 报告与拒识阈值

## 已完成状态

| 里程碑 | 状态 |
|---|---|
| M0 环境与数据 | 完成 |
| M1 LIF 核心与前向 | 完成 |
| M2 训练 | 完成（test 98.97%） |
| M3 推理引擎 | **完成（0.136 ms / 57.7k 样本/s）** |
| M4 校准与 API | 未开始 |
| M5 基准报告 | 未开始 |

## 推理引擎

三级实现，全部与 PyTorch 逐样本一致（10000 条测试集 100% 相同预测）：

| 引擎 | 测试精度 | 单样本 p50 | 批量吞吐 |
|---|---|---|---|
| **L1 Numba 稀疏**（默认） | 98.97% | **0.136 ms** | **57,738 样本/s** |
| L1 Numba 稠密（对照） | 98.97% | 0.329 ms | 19,337 样本/s |
| L0 NumPy 参考 | 98.97% | 0.987 ms | 1,492 样本/s |

- 稠密摊销 842,496 MACs/样本，稀疏有效 183,039 SOP/样本，**理论削减 4.60×**
- 同框架 dense→sparse **实测加速 2.42×**，差额来自 scatter 的访存开销，如实记录
- 推理侧依赖仅 `numpy` + `numba`，无 PyTorch

```python
from freebird_lif.engine.export import load_bundle
from freebird_lif.engine.numba_engine import NumbaEngine

w, cfg = load_bundle("models/freebird-lif-T4.npz")
eng = NumbaEngine(w, cfg)          # 首次调用触发 JIT 编译，约 3 s
digit, conf = eng.forward(pixels)  # pixels: [B,1,16,16] float32 in [0,1]
```

复现基准：`python scripts/bench.py`（完整）或 `--quick`。结果落在 `benchmarks/engine.md`。

**注意**：首次调用含 Numba JIT 编译（约 3 秒）。做延迟基准前必须先预热，否则测到的是编译时间。`bench.py` 已内置预热。

## 定稿配置与结果

| 项 | 值 |
|---|---|
| 测试集精度 | **98.97%** |
| 时间步 T | 4 |
| 参数 | 40,798 |
| 摊销 MACs/样本 | 842,496 |
| 训练用时 | 319 s（12 核 CPU，45 epoch） |
| 模型文件 | `models/freebird-lif-T4.pt` |

训练：`python -m freebird_lif.train --epochs 45 --T 4 --name final_T4`

## 三项需要实测才能确定的超参

直觉在这三处都是错的，全部靠对照实验定值：

**1. 池化位置** —— 精度差异落在种子噪声内，但 `membrane` 快 34%。

**2. 增强强度** —— 非单调。无增强 98.62%、轻档 98.97%、强档 98.50%。强档会把 12×12 的数字内容推出边框裁掉笔画，是破坏而非正则。

**3. 时间步 T** —— 精度在 T=4 饱和。T=8 精度略低（98.91%）却要两倍算力，T=2 掉至 98.52%。

## BN 重校准（部署前必做）

训练期增强会让 BN 的 running stats 偏离真实推理分布。本网络是阈值系统，BN 输出尺度偏 1% 就足以把一批神经元推过或推不过阈值，误差逐层放大。用未增强训练集重算统计量，实测回收 **+0.40 个百分点**。

该步骤已内置于 `train.py`，无需手动调用。

## 池化位置对照（决定 `pool_on` 默认值）

50k 训练 / 8 epoch / 无增强 / T=8，两个分支共用初始化与数据顺序：

| 分支 | val 最优 | test | 用时 |
|---|---|---|---|
| `spike`（脉冲后 max-pool） | 97.98% | 98.14% | 182 s |
| `membrane`（膜电位池化） | 98.20% | 98.45% | 122 s |
| `membrane`（seed=1） | 98.08% | 98.12% | 132 s |

精度差异（+0.31%）落在种子噪声内，不可区分；`membrane` 快 34%，因为池化被移出时间循环、LIF1 神经元数从 3072 降到 768。故取 `membrane` 为默认。

复现：`python scripts/ablate_pool.py --epochs 8 --seeds 0 1`

## 已验证的性质

M1 阶段 57 项测试覆盖以下断言（`pytest -q`）：

- 两种离散化（euler / exponential）在无脉冲区间的轨迹与闭式解吻合，稳态严格为 `u_rest + R·I`
- 欧拉格式误差随 dt 线性下降，收敛比实测落在 1.7–2.4
- 四种替代梯度的数值积分均为 1.0，峰值落在阈值处，反向钩子逐元素等于声明的导数
- 硬重置 / 软重置 / 不应期行为逐时刻可预测
- BN 折叠后 `conv'(x) ≡ BN(conv(x))`，误差 < 1e-6
- Conv1 跨时间步缓存与原实现在数学上完全等价
- 三层发放率在初始化时分别为 7.3% / 5.0% / 3.6%，无死亡层

端到端冒烟训练（12,000 样本、3 epoch、约 17 秒）达到 95.7% 验证精度。

新增 `spike_rates()` 诊断接口，训练 4 epoch 后实测各层发放率：

| 层 | 发放率 | 下游消费方 |
|---|---|---|
| LIF1 | 15.2% | Conv2 输入（占 81% 计算） |
| LIF2 | 24.1% | FC1 输入 |
| LIF3 | 19.4% | FC2 输入 |

该密度高于稀疏内核的舒适区，直接影响 M3 的加速预期——详见 [PLAN.md](PLAN.md) §8.1。

## 上手

```bash
# 1. 激活虚拟环境
.venv/Scripts/activate          # Windows Git Bash: source .venv/Scripts/activate

# 2. 生成 16×16 数据缓存（首次需联网，约 12MB）
python scripts/fetch_data.py --preview

# 3. 运行测试
pytest -q
```

## 数据管线

MNIST 28×28 → 包围盒裁剪 → 等比例缩放至 12×12 内容区（保留与原 MNIST 20/28≈71% 相近的边距比例）→ 居中补零至 16×16。降采样使用面积平均（PIL BOX），避免双三次插值在细笔画上产生振铃。

产物落在 `data/`，由 `.gitignore` 排除，可用 `scripts/fetch_data.py --force` 重建。预处理参数与源文件校验和记录在 `data/manifest.json`。
