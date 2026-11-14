"""数据加载、划分与增强。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_raw(data_dir: Path | None = None) -> dict[str, np.ndarray]:
    d = data_dir or DATA
    return {
        "x_train": np.load(d / "mnist16_train.npy"),
        "y_train": np.load(d / "mnist_labels_train.npy"),
        "x_test": np.load(d / "mnist16_test.npy"),
        "y_test": np.load(d / "mnist_labels_test.npy"),
    }


def to_tensor(x: np.ndarray) -> Tensor:
    """[N,256] uint8 → [N,1,16,16] float32，归一化到 [0,1]。"""
    return torch.from_numpy(x.reshape(-1, 1, 16, 16)).float().div_(255.0)


def to_label(y: np.ndarray) -> Tensor:
    return torch.from_numpy(y.astype(np.int64))


def split_train_val(
    x: Tensor, y: Tensor, val_size: int, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """从训练集划出验证集。官方测试集保持完全隔离，只在最终评估时使用。"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def random_shift(x: Tensor, max_shift: int = 2, generator: torch.Generator | None = None) -> Tensor:
    """整批随机平移（零填充），模拟手写位置的随机性。

    在 16×16 小图上，±2 像素平移已覆盖数字内容区（12px）约 1/6 的尺度，
    是该分辨率下性价比最高的增强手段。
    """
    n = x.shape[0]
    dy = torch.randint(-max_shift, max_shift + 1, (n,), generator=generator)
    dx = torch.randint(-max_shift, max_shift + 1, (n,), generator=generator)
    out = torch.zeros_like(x)
    for i in range(n):
        sy, sx = int(dy[i]), int(dx[i])
        ys, yd = (slice(0, 16 - sy), slice(sy, 16)) if sy >= 0 else (slice(-sy, 16), slice(0, 16 + sy))
        xs, xd = (slice(0, 16 - sx), slice(sx, 16)) if sx >= 0 else (slice(-sx, 16), slice(0, 16 + sx))
        out[i, :, yd, xd] = x[i, :, ys, xs]
    return out


@torch.no_grad()
def accuracy(model, x: Tensor, y: Tensor, batch: int = 512) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        logits, _ = model(x[i : i + batch])
        correct += (logits.argmax(1) == y[i : i + batch]).sum().item()
    model.train(was_training)
    return correct / len(x)


def random_affine(
    x: Tensor,
    max_shift: float = 2.0,
    max_deg: float = 8.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """批量随机仿射增强：平移 + 旋转，一次采样完成，零填充边界。

    用 affine_grid/grid_sample 而非逐样本 PIL 变换——整批一次算子调用，
    在 16×16 这个尺度上开销可忽略，且天然支持 GPU。
    """
    n = x.shape[0]
    # 归一化坐标下图像跨度为 2，1 像素 = 2/16
    unit = 2.0 / x.shape[-1]
    tx = (torch.rand(n, generator=generator, device=x.device) * 2 - 1) * max_shift * unit
    ty = (torch.rand(n, generator=generator, device=x.device) * 2 - 1) * max_shift * unit
    ang = (torch.rand(n, generator=generator, device=x.device) * 2 - 1) * max_deg * math.pi / 180.0

    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def random_brightness(x: Tensor, amount: float = 0.2, generator: torch.Generator | None = None) -> Tensor:
    """逐样本亮度缩放，模拟笔画粗细/书写压力差异。"""
    if amount <= 0:
        return x
    n = x.shape[0]
    scale = 1.0 + (torch.rand(n, 1, 1, 1, generator=generator, device=x.device) * 2 - 1) * amount
    return (x * scale).clamp_(0.0, 1.0)


def augment_batch(
    x: Tensor,
    max_shift: float = 2.0,
    max_deg: float = 8.0,
    brightness: float = 0.2,
    generator: torch.Generator | None = None,
) -> Tensor:
    x = random_affine(x, max_shift, max_deg, generator)
    return random_brightness(x, brightness, generator)
