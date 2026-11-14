"""L1 生产引擎：Numba 编译内核，事件驱动稀疏。

每层按输入性质选内核（PLAN §8.1）：
    Conv1    稠密，且与 t 无关 → 在批维度上用 NumPy/BLAS 一次算完
    Conv2    稀疏 scatter-add（输入密度约 18%）
    FC1      稀疏 gather-add（输入密度约 24%）
    FC2      稠密 sgemv（仅 960 MAC）

两个关键实现选择：

1. **Conv2 输出用填充缓冲 `[10,10,24]`**
   稀疏 scatter 时若对每个 (ky,kx) 做边界判断，每次累加都要分支。
   改成向四周各补 1 像素的缓冲后，索引恒在界内，循环体完全无分支。

2. **通道后置布局**
   见 export.py。稀疏 scatter 的 24 个累加目标连续，写带宽降低一个数量级。

同时保留 dense 变体（不做活跃列表、不跳零）用于 A/B 对照，
量化"稀疏到底省了多少"——这是本项目必须回答的问题。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

F = np.float32
I32 = np.int32

W1_N, W1_K = 12, 9
S1_N = 8 * 8 * 12          # 768
OUT2P_N = 10 * 10 * 24     # 2400
S1P_N = 10 * 10 * 12       # 1200（稠密变体的填充输入）
S2_N = 4 * 4 * 24          # 384
S3_N = 96
N_CLS = 10


@njit(cache=True, fastmath=True, inline="always")
def _lif_step(u, s, x, n, decay, gain, thr, u_rest):
    for i in range(n):
        v = decay * (u[i] - u_rest) + u_rest + gain * x[i]
        if v >= thr:
            s[i] = 1.0
            u[i] = u_rest
        else:
            s[i] = 0.0
            u[i] = v


@njit(cache=True, fastmath=True, inline="always")
def _lif_leaky_out(u, x, n, decay, gain, u_rest):
    for i in range(n):
        u[i] = decay * (u[i] - u_rest) + u_rest + gain * x[i]


@njit(cache=True, fastmath=True)
def _conv2_sparse(s1, w2, b2, out2p, act):
    """事件驱动：只为发放的输入散射 3×3×24 贡献。

    s1: [768] 通道后置 (y*8+x)*12+ic
    w2: [12,9,24] 即 w2[ic*216 + (ky*3+kx)*24 + oc]
    out2p: [2400] (py*10+px)*24+oc 的填充缓冲
    """
    for j in range(OUT2P_N):
        out2p[j] = 0.0
    na = 0
    for p in range(S1_N):
        if s1[p] > 0.0:
            act[na] = p
            na += 1
    for a in range(na):
        p = act[a]
        # s1 是通道后置：p = (y*8 + x)*12 + ic
        pos = p // 12
        ic = p - pos * 12
        y = pos // 8
        x = pos - y * 8
        wbase = ic * 216
        for oy in range(3):
            py = y + oy
            ky = 2 - oy
            for ox in range(3):
                px = x + ox
                kx = 2 - ox
                wrow = wbase + (ky * 3 + kx) * 24
                dst = (py * 10 + px) * 24
                for oc in range(24):
                    out2p[dst + oc] += w2[wrow + oc]
    return na


@njit(cache=True, fastmath=True)
def _conv2_dense(s1p, w2, b2, out2p):
    """稠密对照：不建活跃列表、不跳零，遍历全部 12×3×3 输入。"""
    for j in range(OUT2P_N):
        out2p[j] = 0.0
    for y in range(8):
        for x in range(8):
            dst = ((y + 1) * 10 + (x + 1)) * 24
            for oc in range(24):
                out2p[dst + oc] = 0.0
            for ky in range(3):
                for kx in range(3):
                    sbase = ((y + ky) * 10 + (x + kx)) * 12
                    wrow = (ky * 3 + kx) * 24
                    for ic in range(12):
                        sv = s1p[sbase + ic]
                        wb = ic * 216 + wrow
                        for oc in range(24):
                            out2p[dst + oc] += w2[wb + oc] * sv


@njit(cache=True, fastmath=True, inline="always")
def _pool2(out2p, h2, b2):
    """从填充缓冲直接 2×2 最大池化，中间不做裁剪，省一次 1536 次拷贝。"""
    for u in range(4):
        for v in range(4):
            base = (u * 4 + v) * 24
            py = 2 * u + 1
            px = 2 * v + 1
            for oc in range(24):
                m = out2p[(py * 10 + px) * 24 + oc]
                a1 = out2p[(py * 10 + px + 1) * 24 + oc]
                a2 = out2p[((py + 1) * 10 + px) * 24 + oc]
                a3 = out2p[((py + 1) * 10 + px + 1) * 24 + oc]
                if a1 > m:
                    m = a1
                if a2 > m:
                    m = a2
                if a3 > m:
                    m = a3
                h2[base + oc] = m + b2[oc]


@njit(cache=True, fastmath=True)
def _fc1_sparse(s2, w3t, b3, h3, act):
    for i in range(S3_N):
        h3[i] = b3[i]
    na = 0
    for j in range(S2_N):
        if s2[j] > 0.0:
            act[na] = j
            na += 1
    for a in range(na):
        base = act[a] * S3_N
        for i in range(S3_N):
            h3[i] += w3t[base + i]
    return na


@njit(cache=True, fastmath=True, inline="always")
def _fc2(s3, w4, b4, i4):
    for o in range(N_CLS):
        acc = b4[o]
        row = o * S3_N
        for i in range(S3_N):
            acc += w4[row + i] * s3[i]
        i4[o] = acc


@njit(cache=True, fastmath=True)
def run_sample(h1, w2, b2, w3t, b3, w4, b4, p, T, sparse,
               u1, s1, s1p, out2p, h2, u2, s2, h3, u3, s3, i4, u4, act, logits):
    """单样本完整 T 步前向。p = [decay, gain, thr, u_rest]。"""
    decay, gain, thr, u_rest = p[0], p[1], p[2], p[3]
    for i in range(S1_N):
        u1[i] = 0.0
    for i in range(S2_N):
        u2[i] = 0.0
    for i in range(S3_N):
        u3[i] = 0.0
    for i in range(N_CLS):
        u4[i] = 0.0

    for _ in range(T):
        _lif_step(u1, s1, h1, S1_N, decay, gain, thr, u_rest)
        if sparse:
            _conv2_sparse(s1, w2, b2, out2p, act)
        else:
            for j in range(S1P_N):
                s1p[j] = 0.0
            for y in range(8):
                for x in range(8):
                    for ic in range(12):
                        s1p[((y + 1) * 10 + (x + 1)) * 12 + ic] = s1[(y * 8 + x) * 12 + ic]
            _conv2_dense(s1p, w2, b2, out2p)
        _pool2(out2p, h2, b2)
        _lif_step(u2, s2, h2, S2_N, decay, gain, thr, u_rest)
        _fc1_sparse(s2, w3t, b3, h3, act)
        _lif_step(u3, s3, h3, S3_N, decay, gain, thr, u_rest)
        _fc2(s3, w4, b4, i4)
        _lif_leaky_out(u4, i4, N_CLS, decay, gain, u_rest)

    for o in range(N_CLS):
        logits[o] = u4[o]


@njit(cache=True, fastmath=True)
def _conv1_pool(x, w1t, b1, h1):
    """Conv1 + BN + 池化，融合在单趟内完成。

    x: [16,16]；w1t: [9,12]（(ky*3+kx), oc）；h1: [8,8,12] 通道后置。

    不走 im2col——12 个输出通道让 GEMM 的 K 维只有 9，整段退化为带宽受限，
    实测比本内核慢 2 倍以上。这里 12 个累加器全程驻留寄存器，
    输入经 18×18 填充后循环体无分支。
    """
    xp = np.zeros((18, 18), np.float32)
    for y in range(16):
        for x_ in range(16):
            xp[y + 1, x_ + 1] = x[y, x_]

    acc = np.zeros(12, np.float32)
    buf = np.zeros((16, 16, 12), np.float32)
    for y in range(16):
        for x_ in range(16):
            for o in range(12):
                acc[o] = b1[o]
            for ky in range(3):
                for kx in range(3):
                    v = xp[y + ky, x_ + kx]
                    k = ky * 3 + kx
                    for o in range(12):
                        acc[o] += w1t[k, o] * v
            for o in range(12):
                buf[y, x_, o] = acc[o]

    for u in range(8):
        for v in range(8):
            for o in range(12):
                m = buf[2 * u, 2 * v, o]
                a1 = buf[2 * u, 2 * v + 1, o]
                a2 = buf[2 * u + 1, 2 * v, o]
                a3 = buf[2 * u + 1, 2 * v + 1, o]
                if a1 > m:
                    m = a1
                if a2 > m:
                    m = a2
                if a3 > m:
                    m = a3
                h1[(u * 8 + v) * 12 + o] = m


@njit(cache=True, fastmath=True, parallel=True)
def _conv1_batch(xs, w1t, b1, out):
    for i in prange(xs.shape[0]):
        _conv1_pool(xs[i, 0], w1t, b1, out[i])


@njit(cache=True, fastmath=True, parallel=True)
def _run_batch(h1, w2, b2, w3, b3, w4, b4, p, T, sparse, out,
               u1, s1, s1p, out2p, h2, u2, s2, h3, u3, s3, i4, u4, act):
    """批内并行。prange 必须在 @njit(parallel=True) 函数内才真正并行——
    写在普通 Python 方法里等同于 range，这是本项目踩过的一个实质性坑。"""
    for k in prange(h1.shape[0]):
        run_sample(h1[k], w2, b2, w3, b3, w4, b4, p, T, sparse,
                   u1[k], s1[k], s1p[k], out2p[k], h2[k], u2[k], s2[k],
                   h3[k], u3[k], s3[k], i4[k], u4[k], act[k], out[k])


def conv_front_batch(x: np.ndarray, w1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Conv1 + BN + 池化，在批维度上用一次 GEMM 算完。返回 [B,768] 通道后置。

    im2col 后把 (B,256) 个位置压成一个维度做单次 matmul：
    `(B*256, 9) @ (9, 12)`。相比之下 einsum('ok,bkp->bpo') 的 k=9 收缩
    用不上 BLAS，实测慢约一个数量级。仅在需要纯 NumPy 路径时使用——
    L1 引擎默认走 `_conv1_batch`，后者更快。
    """
    b = x.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.concatenate(
        [xp[:, :, ky : ky + 16, kx : kx + 16].reshape(b, 1, 256)
         for ky in range(3) for kx in range(3)], axis=1)          # [B,9,256]
    flat = np.ascontiguousarray(cols.transpose(0, 2, 1)).reshape(b * 256, 9)
    out = flat @ w1.T + b1                                        # [B*256,12]
    out = out.reshape(b, 16, 16, 12)
    out = out.reshape(b, 8, 2, 8, 2, 12).max(axis=(2, 4))         # [B,8,8,12]
    return np.ascontiguousarray(out.reshape(b, S1_N), dtype=F)


class NumbaEngine:
    """L1 引擎。chunk 控制批内并发粒度与缓冲内存的折中。"""

    def __init__(self, weights: dict, cfg: dict, sparse: bool = True, chunk: int = 256,
                 fused_conv1: bool = True):
        self.w = {k: np.ascontiguousarray(v, dtype=F) for k, v in weights.items()}
        self.cfg = cfg
        self.T = int(cfg["T"])
        self.p = np.array([float(self.w["decay"][0]), float(self.w["gain"][0]),
                           float(cfg["thr"]), float(cfg["u_rest"])], dtype=F)
        self.sparse = sparse
        self.chunk = chunk
        # 内核按一维下标访问，必须展平；保留原数组供 NumPy 侧（Conv1）使用
        self.w2f = np.ascontiguousarray(self.w["conv2_w"].reshape(-1))
        self.w3f = np.ascontiguousarray(self.w["fc1_wt"].reshape(-1))
        self.w4f = np.ascontiguousarray(self.w["fc2_w"].reshape(-1))
        self.b1f = np.ascontiguousarray(self.w["conv1_b"].reshape(-1))
        self.b2f = np.ascontiguousarray(self.w["conv2_b"].reshape(-1))
        self.b3f = np.ascontiguousarray(self.w["fc1_b"].reshape(-1))
        self.b4f = np.ascontiguousarray(self.w["fc2_b"].reshape(-1))
        self.w1t = np.ascontiguousarray(self.w["conv1_w"].T)   # [9,12]
        self.fused_conv1 = fused_conv1
        self._buf = None

    def _buffers(self, n: int):
        if self._buf is not None and self._buf[0].shape[0] == n:
            return self._buf
        z = lambda k: np.zeros((n, k), F)
        # 顺序与 _run_chunk(*bufs) 一致：u1 s1 s1p out2p h2 u2 s2 h3 u3 s3 i4 u4 act
        b = (z(S1_N), z(S1_N), z(S1P_N), z(OUT2P_N), z(S2_N), z(S2_N), z(S2_N),
             z(S3_N), z(S3_N), z(S3_N), z(N_CLS), z(N_CLS),
             np.zeros((n, S1_N), I32))
        self._buf = b
        return b

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: [B,1,16,16] 或 [B,256]，[0,1] float32。返回 logits [B,10]。"""
        if x.ndim == 2:
            x = x.reshape(-1, 1, 16, 16)
        x = np.asarray(x, dtype=F)
        b = x.shape[0]
        out = np.empty((b, N_CLS), F)
        step = min(self.chunk, b) if self.chunk else b
        for s in range(0, b, step):
            e = min(s + step, b)
            n = e - s
            if self.fused_conv1:
                h1 = np.empty((n, S1_N), F)
                _conv1_batch(np.ascontiguousarray(x[s:e]), self.w1t, self.b1f, h1)
            else:
                h1 = conv_front_batch(x[s:e], self.w["conv1_w"], self.w["conv1_b"])
            bufs = self._buffers(n)
            self._run_chunk(h1, out[s:e], *bufs)
        return out

    def _run_chunk(self, h1, out, u1, s1, s1p, out2p, h2, u2, s2, h3, u3, s3, i4, u4, act):
        _run_batch(h1, self.w2f, self.b2f, self.w3f, self.b3f, self.w4f, self.b4f,
                   self.p, self.T, self.sparse, out,
                   u1, s1, s1p, out2p, h2, u2, s2, h3, u3, s3, i4, u4, act)
