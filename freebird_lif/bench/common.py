"""多基准共用的分箱与统计工具。"""

from __future__ import annotations

import numpy as np


def bin_events(t: np.ndarray, idx: np.ndarray, T: int, C: int,
               duration: float) -> np.ndarray:
    """把事件流分箱成 [T, C] 的二值占用图。

    参数
    ----
    t : float32 [E]      每个事件的时间（秒）
    idx : int32 [E]      每个事件落在的通道号（0..C-1）
    T : int              时间步数
    C : int              通道数
    duration : float     样本时长（秒），超出部分按比例截断

    返回
    ----
    uint8 [T, C]，1 表示该格至少来过一个事件。

    实现用 `np.add.at` 到一维计数再取非零，而不是逐事件写二维索引：
    单样本 1e3 量级事件、批量 512 时，前者比 Python 循环快三个数量级。
    """
    frame = (t.astype(np.float64) / max(duration, 1e-9) * T).astype(np.int64)
    np.clip(frame, 0, T - 1, out=frame)
    flat = frame * C + idx.astype(np.int64)
    out = np.zeros(T * C, dtype=np.uint8)
    np.add.at(out, flat, 1)
    return (out.reshape(T, C) > 0).astype(np.uint8)


def density(x: np.ndarray, sample: int = 2000) -> float:
    """平均占用率。用来判断分箱是否过粗（每格都满）或过细（几乎全空）。"""
    n = min(sample, len(x))
    return float(x[:n].mean())


def describe(d: dict) -> str:
    """一行摘要，供各数据集的构建脚本打印。"""
    m = d["meta"]
    xt, yt = d["x_train"], d["y_train"]
    cls, cnt = np.unique(yt, return_counts=True)
    return (f"{m['name']}: train {len(yt)} / test {len(d['y_test'])}  "
            f"T={m['T']} C={m['C']}  类数={m['n_classes']}  "
            f"密度={m['density']:.2%}  "
            f"每类 {cnt.min()}–{cnt.max()} 条  "
            f"{m['duration'] * 1000:.0f} ms/样本（{m['duration'] / m['T'] * 1000:.1f} ms/帧）")
