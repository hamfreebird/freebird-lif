# freebird-lif

LIF 脉冲神经网络研究项目。当前处于**第三阶段**。

| 阶段 | 主题 | 状态 |
|---|---|---|
| v1 | MNIST 16×16，CPU 高效推理（98.97%，0.118 ms/样本） | 完成，已归档 |
| v2 | LIF 变体在 SHD 上的对比 + Arc GPU 训练 | 完成，已归档 |
| v3 | 循环结构的复现与诊断、机制归因的跨结构验证、多基准普适性 | 计划已定，M0 待启动 |

## 阶段二的结论（一句话版）

前馈 LIF 在 SHD 上做到 **83.04% ± 0.63%**（3 seed）；**只有 AdLIF 显著更好**（86.07%，+3.03）。
机制消融显示增益来自**发放驱动**的适应电流，而不是文献归因的膜电位驱动二阶反馈。
最有方法论价值的发现：**弱基线会系统性高估变体**——同一批变体在 54% 基线上测出
`+14.1 / +8.1 / +4.2`，在 83% 基线上变成 `+3.03 / −1.19 / −3.09`。

完整报告：[benchmarks/report-v2.md](benchmarks/report-v2.md)

## 阶段三要做什么

把"弱基线高估机制价值"这个怀疑从神经元推广到**结构**与**基准**两个维度，
并诊断循环为什么在本实现里失效（文献 87.04%，本实现前馈就已 83.04%）。
详见 [PLAN.md](PLAN.md)。

## 归档

| 阶段 | 归档包 | 内容 |
|---|---|---|
| v1 | `H:/Dev/archive/freebird-lif-v1-20261115.tar.gz` | 源码、模型、报告、git 历史（7 次提交） |
| v2 | `H:/Dev/archive/freebird-lif-v2-20261116.tar.gz` | 同上 + 6 个检查点 + 全部逐 seed 结果与逐样本预测 |

解压即得干净工作树（`git status` 为空）。可再生产物（`.venv`、`data/`、`runs/` 下的
临时文件）未纳入，按 `docs/PLAN-v2-shd.md` §8 的步骤重建。

## 环境

需要 **PyTorch XPU** 轮子（Intel Arc 显卡）：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/xpu
.venv/Scripts/python -m pip install numpy h5py pytest
python -c "import torch; print(torch.xpu.is_available())"
```

**用 SJTU 镜像**：实测 815 KB/s，官方源 166 KB/s，快 4.9 倍。

注意事项：

- **不要安装 `intel-extension-for-pytorch`**：IPEX 已于 2026 年 3 月 EOL。
- 本机无 MSVC，**`torch.compile` 在 XPU 上不可用**，只能 eager。
- 消费级 Arc **无 FP64**（`has_fp64=0`），所有 float64 校验必须放 CPU。
- **训练批大小 ≥ 512**：类 SNN 负载在内核启动受限区，B=128 时 GPU 几乎无收益。
- **AdLIF 训练不要并发**：它的正反馈回路处在发散边界，并发扰动会触发 NaN。

## 目录

```
freebird-lif/
├── PLAN.md                  第三阶段技术基准
├── docs/                    v1/v2 的 PLAN 与 README
├── freebird_lif/            v1 代码（neuron/network/engine）+ v2 变体库与 SHD 管线
├── scripts/                 数据、训练、基准、分析
├── models/                  检查点（见 models/README.md）
├── tests/                   106 项
└── benchmarks/              v1/v2 的全部报告与逐 seed 结果
```
