"""公开推理 API。

用法：
    from freebird_lif.api import Predictor

    p = Predictor("models/freebird-lif-T4.npz", calibration="models/calibration.json")
    digit, conf = p.predict_one(pixels)          # pixels: [16,16]
    digits, confs = p.predict(batch)             # batch: [B,16,16]

输入形状与值域自动规范化：
    - 形状接受 [16,16]、[256]、[1,16,16]、[B,16,16]、[B,1,16,16]、[B,256]
    - 值域接受 uint8 [0,255] 与 float [0,1]；float 若最大值 > 1 则按 255 归一化
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine.export import load_bundle
from .engine.numba_engine import NumbaEngine

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "models" / "freebird-lif-T4.npz"
DEFAULT_CALIB = ROOT / "models" / "calibration.json"


@dataclass
class Prediction:
    digit: int
    confidence: float
    accepted: bool


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def as_batch(pixels) -> np.ndarray:
    """把各种输入形状/值域规范成 [B,1,16,16] float32 且落在 [0,1]。"""
    x = np.asarray(pixels)

    if x.ndim == 1:
        if x.size != 256:
            raise ValueError(f"一维输入长度应为 256，实际 {x.size}")
        x = x.reshape(1, 16, 16)
    elif x.ndim == 2:
        if x.shape == (16, 16):
            x = x[None]
        elif x.shape[1] == 256:
            x = x.reshape(-1, 16, 16)
        else:
            raise ValueError(f"二维输入应为 (16,16) 或 (B,256)，实际 {x.shape}")
    elif x.ndim == 3:
        if x.shape[1:] == (256,):
            x = x.reshape(-1, 16, 16)
        elif x.shape[1:] != (16, 16):
            raise ValueError(f"三维输入应为 (B,16,16) 或 (B,256)，实际 {x.shape}")
    elif x.ndim == 4:
        if x.shape[1] != 1:
            raise ValueError(f"四维输入第二维应为 1（单通道），实际 {x.shape[1]}")
        x = x[:, 0]
    else:
        raise ValueError(f"不支持的输入维度 {x.ndim}")

    if x.shape[1:] != (16, 16):
        raise ValueError(f"空间形状应为 16×16，实际 {x.shape[1:]}")

    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("输入含 NaN/Inf")
    if x.size and x.max() > 1.0:
        x = x / 255.0
    if x.min() < 0.0 or x.max() > 1.0:
        raise ValueError(f"输入值域应为 [0,1] 或 [0,255]，实际 [{x.min():.3f}, {x.max():.3f}]")
    return np.ascontiguousarray(x.reshape(-1, 1, 16, 16))


class Predictor:
    def __init__(self, bundle: str | Path = DEFAULT_BUNDLE,
                 calibration: str | Path | None = DEFAULT_CALIB,
                 threshold: float | None = None):
        bundle = Path(bundle)
        if not bundle.exists():
            raise FileNotFoundError(f"权重包不存在: {bundle}。先运行 scripts/bench.py 或 engine.export.save_bundle")
        self.weights, self.cfg = load_bundle(bundle)
        self.engine = NumbaEngine(self.weights, self.cfg)
        self.temperature = 1.0
        self.threshold = threshold
        if calibration is not None and Path(calibration).exists():
            cal = json.loads(Path(calibration).read_text("utf-8"))
            self.temperature = float(cal.get("temperature", 1.0))
            if threshold is None:
                self.threshold = cal.get("suggested_tau")

    # ---- 核心 -------------------------------------------------------------

    def probabilities(self, pixels) -> np.ndarray:
        logits = self.engine.forward(as_batch(pixels))
        return _softmax(logits / self.temperature)

    def predict(self, pixels) -> tuple[np.ndarray, np.ndarray]:
        """返回 (digits, confidences)，均为长度为 B 的数组。"""
        p = self.probabilities(pixels)
        return p.argmax(1), p.max(1)

    def predict_one(self, pixels) -> Prediction:
        """单张图。低于拒识阈值时 accepted=False，digit 仍给出最可能类别。"""
        d, c = self.predict(pixels)
        return Prediction(int(d[0]), float(c[0]), bool(self.threshold is None or c[0] >= self.threshold))

    def explain(self, pixels, top_k: int = 3) -> list[dict]:
        """返回每个样本的前 top_k 个候选，便于上层展示。"""
        p = self.probabilities(pixels)
        out = []
        for row in p:
            idx = np.argsort(row)[::-1][:top_k]
            out.append({"top": [(int(i), float(row[i])) for i in idx]})
        return out
