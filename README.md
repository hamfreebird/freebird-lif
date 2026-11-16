# freebird-lif

LIF 脉冲神经网络研究项目。当前处于**第二阶段**。

| 阶段 | 主题 | 状态 |
|---|---|---|
| v1 | MNIST 16×16，CPU 高效推理（98.97%，0.118 ms/样本） | 完成，已归档 |
| v2 | LIF 变体在 SHD 上的对比 + Arc GPU 训练 | M0–M4 完成，M5 待启动 |

完整方案见 [PLAN.md](PLAN.md)。第一阶段的技术基准与数据见 [docs/PLAN-v1-lif-mnist.md](docs/PLAN-v1-lif-mnist.md)，说明见 [docs/README-v1.md](docs/README-v1.md)。

## 当前进展（v2）

| 里程碑 | 状态 |
|---|---|
| M0 XPU 环境 + SHD 管线 | 完成（Arc A750，`torch.xpu` 可用） |
| M1 五个神经元变体 + 单元测试 | 完成（5/6，POLARA 因原文不可得而阻塞） |
| M2 LIF 基线调优 | **完成：test 83.04 ± 0.63%（3 seed）** |
| M3 五变体对比 | **完成：AdLIF 86.07（+3.03，显著）；ALIF/GLIF 不可区分；MC-LIF −3.09** |

| M4 分析 | **完成：机制消融、成本-精度、分类别误差、recurrent 复核** |

变体结论见 [benchmarks/m3_variants.md](benchmarks/m3_variants.md)，深度分析见
[benchmarks/m4_analysis.md](benchmarks/m4_analysis.md)。要点：
**在 54% 的弱基线上做过的那次筛查里，四个变体有三个的「优势」在 83% 基线上
消失或翻负**——弱基线会系统性高估新神经元机制的价值。

基线配方：700 → 384 → 384 → 20 前馈，无归一化，层内异构时间常数（1→40 帧），
训练期频道平移 ±35，120 epoch。从 M0 冒烟的 53.5% 提升 29.5 个百分点。
详见 [benchmarks/shd_baseline.md](benchmarks/shd_baseline.md)。

`python scripts/train_shd.py --grid r10 --seeds 0 1 2` 可复现。

## 归档

第一阶段的源码、模型权重与报告已打包：

```
H:/Dev/archive/freebird-lif-v1-20261115.tar.gz   (888 KB, 含 git 历史)
```

解压即得干净工作树（`git status` 为空），包含 7 次提交、101 项测试、最终报告与定稿模型。
可再生产物（`.venv`、`data/*.npy`、`runs/`）未纳入归档，按 `docs/PLAN-v1-lif-mnist.md` §10 的步骤重建。

## 环境

第二阶段需要 **PyTorch XPU** 轮子（Intel Arc 显卡）：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
.venv/Scripts/python -m pip install numpy h5py pytest
python -c "import torch; print(torch.xpu.is_available())"
```

注意事项：

- **不要安装 `intel-extension-for-pytorch`**。IPEX 已于 2026 年 3 月 EOL，
  原生 `torch.xpu` 已是官方路径。
- 本机无 MSVC，因此 **`torch.compile` 在 XPU 上不可用**（官方要求 MSVC + Windows SDK）。
  只能走 eager 模式，这是既定的性能上限。
- v1 的 CPU 环境（numba 引擎）与本阶段的 XPU 环境互斥，只能二选一装在 `.venv` 里。
  v1 的推理引擎在归档中，需要时单独重建。

## 目录

```
freebird-lif/
├── PLAN.md                  第二阶段技术基准
├── docs/                    v1 归档文档
├── freebird_lif/            v1 代码（neuron/network/engine 可复用）
├── scripts/                 数据、训练、基准脚本
├── tests/
└── benchmarks/              v1 的 engine.md / report.md 等
```
