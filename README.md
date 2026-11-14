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
| M2 训练 | 未开始 |
| M3 推理引擎 | 未开始 |
| M4 校准与 API | 未开始 |
| M5 基准报告 | 未开始 |

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
