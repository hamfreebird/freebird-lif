"""EvTouch-Objects：事件驱动触觉物体识别。

来源：https://github.com/clear-nus/TactileSGNet （仓库内直接托管 zip，无需表单）
     论文 TactileSGNet, IROS 2020 / 数据集论文 RSS 2020 (NeuTouch 传感器)

原始格式：每个样本一个 .npy，形状 `(39, 2, 250)`：
    39   —— NeuTouch 的 39 个 taxel（传感器电极）
    2    —— ON / OFF 两个极性通道
    250  —— 时间步，20 ms/帧，合计 5 s（与原文一致："lifting 到 releasing ≈5 秒"）

36 类、每类 20 条、共 720 条；仓库自带 576/144 的 train/test 划分。

本模块把极性折叠成单通道：对每个 taxel 取 "是否来事件"（ON 或 OFF 均为 1）。
理由是极性在触觉里几乎对称（压/松），保留两通道会让输入维度翻倍而信息量
不变——而本项目用的 2×384 前馈网络对输入维度敏感（1156 vs 578）。
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

from .common import density

NAME = "evtouch-objects"
ROOT = Path(__file__).resolve().parents[2] / "data" / "bench" / "evtouch" / "Ev-Objects"

_LABEL_RE = re.compile(r"label_(\d+)\.npy$")


def _load_split(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(str(directory / "*.npy")))
    if not files:
        raise FileNotFoundError(f"{directory} 下没有 .npy；请先解压 Ev-Objects.zip")
    xs, ys = [], []
    for f in files:
        y = int(_LABEL_RE.search(f).group(1))
        a = np.load(f)                          # (39, 2, 250) float32 0/1
        if a.ndim != 3 or a.shape[0] != 39:
            raise RuntimeError(f"{Path(f).name} 形状异常 {a.shape}，预期 (39,2,T)")
        xs.append((a.max(axis=1) > 0).T.astype(np.uint8))   # (T, 39)
        ys.append(y)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def build(T: int | None = None, force: bool = False) -> dict:
    """构建缓存。T=None 用原生 250 帧；指定 T 时按 OR 池化降采样。

    池化用 OR 而不是求和：这是二值占用图，求和会造出 >1 的值，
    再把整条管线推向"计数"语义，与 SHD 的二值口径不一致。
    """
    x_tr, y_tr = _load_split(ROOT / "train")
    x_te, y_te = _load_split(ROOT / "test")

    if T is not None and T != x_tr.shape[1]:
        def pool(x: np.ndarray, t: int) -> np.ndarray:
            """按均匀边界池化。原生 250 帧不能被任意 T 整除，故用边界下标
            而不是 reshape——reshape 要求整除，会逼着 T 只能取 250 的约数。"""
            n, t0, c = x.shape
            if t > t0:
                raise ValueError(f"T={t} 大于原生帧数 {t0}")
            edges = np.linspace(0, t0, t + 1).round().astype(int)
            out = np.empty((n, t, c), dtype=x.dtype)
            for i in range(t):
                lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
                out[:, i] = x[:, lo:hi].max(axis=1)
            return out
        x_tr, x_te = pool(x_tr, T), pool(x_te, T)

    # 从训练集切出验证集：本数据集没有官方验证集，而 144 条的测试集太小
    # （每类 4 条，精度粒度 0.69 点），模型选择必须靠验证集。
    rng = np.random.default_rng(2024)
    perm = rng.permutation(len(y_tr))
    n_val = max(36, int(round(len(y_tr) * 0.15)))
    va_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    x_va, y_va = x_tr[va_idx], y_tr[va_idx]
    x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]

    n, t, c = x_tr.shape
    d = {
        "x_train": np.ascontiguousarray(x_tr), "y_train": y_tr,
        "x_val": np.ascontiguousarray(x_va), "y_val": y_va,
        "x_test": np.ascontiguousarray(x_te), "y_test": y_te,
        "meta": {
            "name": NAME, "T": t, "C": c,
            "n_classes": int(max(y_tr.max(), y_te.max())) + 1,
            "duration": 5.0,
            "density": density(x_tr),
            "source": "github.com/clear-nus/TactileSGNet",
            "modality": "tactile",
            "split": "仓库自带 576/144；从 576 里再切 15% 做验证，剩 490 训练",
        },
    }
    return d
