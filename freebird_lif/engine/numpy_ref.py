"""L0 参考引擎：NumPy 向量化实现，作为数值真值。

内部统一使用**通道后置**布局（与 export.py 的权重排布一致）：
    空间张量形状 [B, H, W, C]，展平后下标 = (y*W + x)*C + c
这样 Conv2 的稀疏 scatter 与 FC1 的列序天然对齐，无需任何重排。

本实现刻意保持"一眼可读"，不做稀疏或分块优化——它的职责是给 L1 提供判据。
"""

from __future__ import annotations

import numpy as np

K = 3
PAD = 1


class NumpyEngine:
    def __init__(self, weights: dict[str, np.ndarray], cfg: dict) -> None:
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.cfg = cfg
        self.T = int(cfg["T"])
        self.thr = np.float32(cfg["thr"])
        self.u_rest = np.float32(cfg["u_rest"])
        self.decay = np.float32(self.w["decay"][0])
        self.gain = np.float32(self.w["gain"][0])

    # ---- 单步构件 ---------------------------------------------------------

    def _conv_front(self, x: np.ndarray) -> np.ndarray:
        """Conv1 + BN（已折叠）+ 池化，输出 [B,8,8,12]。输入与 t 无关，只算一次。"""
        b = x.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)))
        cols = np.concatenate(
            [xp[:, :, ky : ky + 16, kx : kx + 16].reshape(b, 1, 256)
             for ky in range(K) for kx in range(K)], axis=1)          # [B,9,256]
        out = np.einsum("ok,bkp->bpo", self.w["conv1_w"], cols)       # [B,256,12]
        out += self.w["conv1_b"]
        out = out.reshape(b, 8, 2, 8, 2, 12).max(axis=(2, 4))         # 2×2 池化
        return np.ascontiguousarray(out)

    def _conv2(self, s1: np.ndarray) -> np.ndarray:
        """Conv2 + BN，输出 [B,8,8,24]（池化前）。"""
        b = s1.shape[0]
        sp = np.pad(s1, ((0, 0), (PAD, PAD), (PAD, PAD), (0, 0)))
        out = np.broadcast_to(self.w["conv2_b"], (b, 8, 8, 24)).copy()
        for ky in range(K):
            for kx in range(K):
                k = ky * K + kx
                # w2n[ic, k, oc] × sp[b, y+ky, x+kx, ic] → out[b, y, x, oc]
                out += np.einsum("io,bhwi->bhwo", self.w["conv2_w"][:, k, :],
                                 sp[:, ky : ky + 8, kx : kx + 8, :])
        return out

    def _lif(self, x: np.ndarray, u: np.ndarray, spiking: bool = True):
        u = self.decay * (u - self.u_rest) + self.u_rest + self.gain * x
        if not spiking:
            return u, u
        s = (u >= self.thr).astype(np.float32)
        u = np.where(s > 0, self.u_rest, u)
        return s, u

    # ---- 前向 -------------------------------------------------------------

    def forward(self, x: np.ndarray, return_logits_only: bool = True) -> np.ndarray:
        """x: [B,1,16,16] 或 [B,256]，值域 [0,1] 的 float32。返回 logits [B,10]。"""
        if x.ndim == 2:
            x = x.reshape(-1, 1, 16, 16)
        x = np.asarray(x, dtype=np.float32)
        b = x.shape[0]

        h1 = self._conv_front(x * np.float32(self.cfg.get("input_gain", 1.0)))
        u1 = np.zeros((b, 8, 8, 12), np.float32)
        u2 = np.zeros((b, 4, 4, 24), np.float32)
        u3 = np.zeros((b, 96), np.float32)
        u4 = np.zeros((b, 10), np.float32)

        for _ in range(self.T):
            s1, u1 = self._lif(h1, u1)                       # [B,8,8,12]
            h2 = self._conv2(s1)                             # [B,8,8,24]
            h2 = h2.reshape(b, 4, 2, 4, 2, 24).max(axis=(2, 4))   # 2×2 池化
            s2, u2 = self._lif(np.ascontiguousarray(h2), u2)       # [B,4,4,24]
            h3 = s2.reshape(b, 384) @ self.w["fc1_wt"] + self.w["fc1_b"]
            s3, u3 = self._lif(h3, u3)                       # [B,96]
            i4 = s3 @ self.w["fc2_w"].T + self.w["fc2_b"]
            u4, _ = self._lif(i4, u4, spiking=False)         # 泄漏积分读出

        return u4

    def predict(self, x: np.ndarray):
        logits = self.forward(x)
        conf = _softmax(logits).max(axis=-1)
        return logits.argmax(axis=-1), conf


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
