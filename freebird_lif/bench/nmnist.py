"""N-MNIST：事件相机版的 MNIST（Sironi et al. 2018 / Orchard et al. 2015）。

来源有两个，格式不同：
  A. 官方 zip（garrickorchard.com → Mendeley，Train.zip 1012 MB / Test.zip 170 MB），
     解压后为 `<split>/<digit>/<id>.bin`
  B. HuggingFace 镜像 `Blackrose900/N-MNIST`，已是解压后的目录树 `Train|Test/<digit>/*.bin`
     （每类 1000 个，是官方的 1/6 子集）

事件格式（两者相同）：每事件 5 字节，**时间戳跨 byte2 的低 7 位与 byte3-4**：

    byte0        x         0..33
    byte1        y         0..33
    byte2 >> 7   polarity  0=OFF, 1=ON
    byte2 & 127  timestamp 高位（第 16–22 位）
    byte3        timestamp 中位
    byte4        timestamp 低位
    ts(µs) = ((byte2 & 127) << 16) | (byte3 << 8) | byte4

以上取自官方 `Read_Ndataset.m`（Mendeley 数据集内附带）。
**不要按"x,y,p + 4 字节时间戳"去解析**——那样会把 byte2 的 7 位时间戳高位
当成极性、把 byte3 当时间戳最高位，得到非单调的时间序列。
第一版就是这么错的，表现为 `ts` 乱序。

样本时长 300 ms，分辨率 34×34。

通道布局：单通道，C = 34*34 = 1156，极性折叠（ON/OFF 都记为"来过事件"）。
理由与 EvTouch 相同——保持与 SHD 一致的二值占用口径，避免引入"计数"这个
与假设无关的变量。极性信息并非无用，但保留它会让输入维度翻倍到 2312，
使 2×384 的模型容量对比失去可比性。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import bin_events, density

NAME = "n-mnist"
SIZE = 34
DURATION = 0.300          # 秒
ROOT = Path(__file__).resolve().parents[2] / "data" / "bench" / "nmnist"

_EVENT_DTYPE = np.dtype([("x", "u1"), ("y", "u1"), ("p", "u1"), ("t", "<u4")])


def read_bin(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (t 秒, 通道号, 极性)。通道号 = y*34 + x。"""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 5 != 0:
        raise RuntimeError(f"{path.name}: 字节数 {raw.size} 不是 5 的倍数，格式不符")
    ev = raw.reshape(-1, 5)
    x = ev[:, 0].astype(np.int32)
    y = ev[:, 1].astype(np.int32)
    p = ev[:, 2] >> 7                                  # 0=OFF, 1=ON
    b2 = ev[:, 2].astype(np.uint32)
    t_us = ((b2 & 127) << 16) | (ev[:, 3].astype(np.uint32) << 8) \
        | ev[:, 4].astype(np.uint32)
    t = t_us.astype(np.float64) * 1e-6
    if x.max() >= SIZE or y.max() >= SIZE:
        raise RuntimeError(f"{path.name}: 坐标越界 x<={x.max()} y<={y.max()}")
    if t.size > 1 and not np.all(np.diff(t) >= -1e-9):
        raise RuntimeError(f"{path.name}: 时间戳非单调，字节布局可能又解析错了")
    return t, y * SIZE + x, p


def _collect(split_dir: Path) -> tuple[list[Path], np.ndarray]:
    files, labels = [], []
    for d in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        digit = int(d.name)
        fs = sorted(d.glob("*.bin"))
        files += fs
        labels += [digit] * len(fs)
    if not files:
        raise FileNotFoundError(f"{split_dir} 下没有 .bin")
    return files, np.asarray(labels, dtype=np.int64)


def build(T: int = 100, limit_per_class: int | None = None) -> dict:
    """分箱成 [N, T, 1156]。limit_per_class 用于子集化（HF 镜像每类仅 1000 个）。"""
    out = {}
    for split in ("Train", "Test"):
        files, labels = _collect(ROOT / split)
        if limit_per_class is not None:
            keep = []
            seen: dict[int, int] = {}
            for f, y in zip(files, labels):
                c = seen.get(int(y), 0)
                if c < limit_per_class:
                    keep.append(True)
                    seen[int(y)] = c + 1
                else:
                    keep.append(False)
            keep = np.asarray(keep)
            files = [f for f, k in zip(files, keep) if k]
            labels = labels[keep]
        xs = np.empty((len(files), T, SIZE * SIZE), dtype=np.uint8)
        for i, f in enumerate(files):
            t, idx, _p = read_bin(f)
            xs[i] = bin_events(t, idx, T, SIZE * SIZE, DURATION)
        out[split] = (xs, labels)
        print(f"  {split}: {len(labels)} 条，{len(np.unique(labels))} 类", flush=True)

    xtr, ytr = out["Train"]
    xte, yte = out["Test"]

    # 从训练集切 15% 做验证：官方 N-MNIST 没有验证集，而 RQ-C 要跨数据集
    # 用同一套模型选择协议（见 PLAN §5.2），不能这里看测试集、那里看验证集。
    rng = np.random.default_rng(2024)
    perm = rng.permutation(len(ytr))
    n_val = max(len(np.unique(ytr)), int(round(len(ytr) * 0.15)))
    va, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    xva, yva = xtr[va], ytr[va]
    xtr, ytr = xtr[tr], ytr[tr]

    return {
        "x_train": xtr, "y_train": ytr,
        "x_val": xva, "y_val": yva,
        "x_test": xte, "y_test": yte,
        "meta": {
            "name": NAME, "T": T, "C": SIZE * SIZE,
            "n_classes": int(max(ytr.max(), yte.max())) + 1,
            "duration": DURATION,
            "density": density(xtr),
            "source": "huggingface.co/datasets/Blackrose900/N-MNIST (官方 Train.zip 的 1/6 子集)",
            "modality": "vision",
            "split": "镜像自带 Train/Test；再从 Train 切 15% 做验证",
        },
    }


def save(d: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in d.items() if k != "meta"},
                        meta=np.array([d["meta"]], dtype=object))
    return path
