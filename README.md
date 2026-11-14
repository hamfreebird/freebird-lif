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
| M1 LIF 核心与前向 | 未开始 |
| M2 训练 | 未开始 |
| M3 推理引擎 | 未开始 |
| M4 校准与 API | 未开始 |
| M5 基准报告 | 未开始 |

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
