"""权重导出：把训练好的 ConvSNN 折叠 BN 并重排成引擎内部布局。

布局由性能约束决定，不是随意选择：

1. **Conv2 权重转 `[ic, ky*3+kx, oc]`，输出缓冲转通道后置 `[oy, ox, oc]`**
   稀疏 scatter 时，固定 (ic, ky, kx) 后 24 个输出通道权重连续，24 个累加目标
   也连续。若沿用 PyTorch 的 `[oc, ic, ky, kx]` 配通道首置缓冲，每次累加的写
   地址间隔 64 个 float，216 次写会打穿 216 条 cache line。

2. **FC1 列按通道后置序置换后转置 `[384, 96]`**
   使列序与 Conv2 输出缓冲的内存顺序一致，省掉一次重排；转置后每行 96 个权重
   连续，稀疏 gather 时按行访问。

3. **decay / gain 直接从模型读取**
   不在引擎侧重新推导公式，避免两处分叉。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..network import ConvSNN, SNNConfig, fold_conv_bn, fold_linear_bn


def channel_last_perm(n_side: int, n_ch: int) -> list[int]:
    """通道首置展平下标 → 通道后置展平下标的重排表。

    原序 idx = c*H*W + y*W + x；新序 idx = (y*W + x)*C + c。
    """
    return [c * n_side * n_side + y * n_side + x
            for y in range(n_side) for x in range(n_side) for c in range(n_ch)]


@torch.no_grad()
def export_weights(model: ConvSNN) -> dict[str, np.ndarray]:
    model.eval()
    c = model.cfg

    w1, b1 = fold_conv_bn(model.conv1, model.bn1)
    w2, b2 = fold_conv_bn(model.conv2, model.bn2)
    w3, b3 = fold_linear_bn(model.fc1, model.bn3)

    w1n = w1.cpu().numpy().reshape(c.n_conv1, 9).astype(np.float32)          # [12, 9]
    w2n = (w2.cpu().numpy().transpose(1, 2, 3, 0)
           .reshape(c.n_conv1, 9, c.n_conv2).astype(np.float32))             # [12, 9, 24]
    perm = channel_last_perm(4, c.n_conv2)
    w3t = np.ascontiguousarray(w3.cpu().numpy()[:, perm].T).astype(np.float32)  # [384, 96]

    return {
        "conv1_w": w1n, "conv1_b": b1.cpu().numpy().astype(np.float32),
        "conv2_w": w2n, "conv2_b": b2.cpu().numpy().astype(np.float32),
        "fc1_wt": w3t, "fc1_b": b3.cpu().numpy().astype(np.float32),
        "fc2_w": model.fc2.weight.detach().cpu().numpy().astype(np.float32),
        "fc2_b": model.fc2.bias.detach().cpu().numpy().astype(np.float32),
        "decay": np.array([model.lif1.decay], dtype=np.float32),
        "gain": np.array([model.lif1.gain], dtype=np.float32),
    }


def save_bundle(model: ConvSNN, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = export_weights(model)
    arrays["cfg"] = np.array(json.dumps(asdict(model.cfg)))
    np.savez(path, **arrays)
    return path


def load_bundle(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z["cfg"]))
    weights = {k: z[k] for k in z.files if k != "cfg"}
    return weights, cfg


def load_checkpoint(path: Path) -> tuple[ConvSNN, SNNConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SNNConfig(**ckpt["snn_config"])
    model = ConvSNN(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg
