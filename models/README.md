# 模型权重

## 阶段二（SHD）

| 文件 | 模型 | 测试精度 | 参数 | 来源 |
|---|---|---|---|---|
| `shd-lif-ff-2x384.pt` | 前馈 LIF，2×384 | **82.33%**（该 seed；3 seed 均值 83.04 ± 0.63） | 425,492 | seed 0 |
| `shd-adlif-ff-2x384.pt` | 前馈 AdLIF，2×384 | **85.29%**（该 seed；3 seed 均值 86.07 ± 0.68） | 427,796 | seed 0 |

两个文件都含 `config`（完整训练配置）、`state_dict`、`test_pred`（逐样本预测）。
加载方式：

```python
import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from train_shd import SHDNet, Cfg, predict

ck = torch.load("models/shd-lif-ff-2x384.pt", weights_only=False)
model = SHDNet(Cfg(**ck["config"]))
model.load_state_dict(ck["state_dict"])
model.eval()
pred = predict(model, x_test)          # x_test: [N, 100, 700] float32
```

**配方**（`config` 字段里有完整取值）：700 → 384 → 384 → 20 前馈，无归一化，
in_scale=5，层内 τ 对数铺开 1→40 帧、层间 ×2，读出为非脉冲泄漏积分（α=0.6）对 T 取均值，
AdamW lr 3e-3 / wd 1e-4 / batch 512 / 120 epoch，训练期频道平移 ±35。

**关于 seed**：git 里只放 seed 0 各一个（每个 1.7 MB）。三个 seed 的全部权重在归档包
`archive/freebird-lif-v2-20261116.tar.gz` 的 `runs/` 下。训练经实测**逐位可复现**
（同配置同 seed 重跑得数值完全相同），因此任一 seed 都可用
`python scripts/train_shd.py --grid final_w --seeds 0 1 2` 重新生成。

---

## 阶段一（MNIST 16×16，已归档）

| 文件 | 说明 |
|---|---|
| `freebird-lif-T4.pt` | PyTorch 检查点，98.97%，170 KB |
| `freebird-lif-T4.npz` | 推理引擎权重包（BN 已折叠） |
| `calibration.json` | 温度校准参数与拒识阈值 |

阶段一的说明见 `docs/PLAN-v1-lif-mnist.md`。
