"""数据加载、划分与增强。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
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
