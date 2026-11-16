"""多基准数据集管线（阶段三 RQ-C）。

统一接口：所有数据集都产出

    {
      "x_train": uint8 [N, T, C],  # 每格是否至少来过一个事件（0/1）
      "y_train": int64 [N],
      "x_test":  uint8 [M, T, C],
      "y_test":  int64 [M],
      "meta":    {"name", "T", "C", "n_classes", "duration", "density", ...},
    }

为什么是二值而不是计数：阶段二的 SHD 管线就是二值（`binary=True`），
换成计数会改变输入分布，让跨数据集比较混入一个与假设无关的变量。
N-MNIST 单样本事件数约 1e3、通道数 1156，若用计数则每格几乎恒为 1，
二值反而是更有信息量的表示。
"""

from .common import bin_events, describe

__all__ = ["bin_events", "describe"]
