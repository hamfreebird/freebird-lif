"""脉冲神经元变体的统一接口。

设计约束来自实验需求：六个变体必须**只差在神经元方程**，其余全部相同。
因此接口只暴露状态与单步推进，不掺入任何层结构或优化逻辑。

三个成本指标是并行的：
    n_state     每神经元的状态**张量**个数（实现细节，含二值的 s_prev）
    float_state 每神经元的**浮点**状态个数（部署内存：LIF=1，AdLIF=2，MC-LIF=1+K）
    op_count    每神经元每步的算术操作数（含 sigmoid 折算）
一个"精度 +1% 但成本 2×"的变体是坏交易，所以成本必须和精度一起报。
`n_state` 与 `float_state` 必须分开：s_prev 是二值脉冲，不占浮点存储。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..neuron import spike as _spike


class SpikingNeuron(nn.Module):
    """所有变体的基类。

    子类必须实现 `step`，并声明 `n_state`。状态用 tuple 传递，
    元素顺序由子类自行约定，外部只做透传。
    """

    n_state: int = 1
    float_state: int = 1
    # 胞体膜电位在 state tuple 里的下标。LIF/AdLIF 是 0；
    # 两隔室变体（TCLIF/ATLIF）的 0 是树突、1 才是胞体。仪器（PCI/ITC）
    # 需要读胞体，靠这个属性避免按 n_state 猜。
    soma_index: int = 0

    def __init__(self, thr: float = 0.5, alpha: float = 2.0,
                 surrogate: str = "fast_sigmoid", reset: str = "subtract"):
        super().__init__()
        self.thr = float(thr)
        self.surrogate_alpha = float(alpha)
        self.surrogate = surrogate
        if reset not in ("subtract", "zero"):
            raise ValueError(f"reset 必须是 'subtract' 或 'zero'，实际 {reset!r}")
        self.reset = reset

    # ---- 供子类复用 -------------------------------------------------------

    def _fire(self, u: Tensor) -> Tensor:
        return _spike(u - self.thr, self.surrogate_alpha, self.surrogate)

    def _reset(self, u: Tensor, s: Tensor) -> Tensor:
        """s 必须 detach——重置项不应再贡献一条经由 surrogate 的梯度路径。"""
        if self.reset == "subtract":
            return u - self.thr * s.detach()
        return u * (1.0 - s.detach())

    # ---- 接口 -------------------------------------------------------------

    def init_state(self, batch: int, features: int, device=None, dtype=None) -> tuple:
        raise NotImplementedError

    def step(self, x: Tensor, state: tuple) -> tuple[Tensor, tuple]:
        raise NotImplementedError

    def op_count(self) -> float:
        raise NotImplementedError

    def input_scale(self) -> Tensor | None:
        """返回使"输入耦合增益 = 1"所需的逐神经元缩放，None 表示该变体不适用。

        背景（阶段三 H1）：`u = α·u + (1-α)·x` 里的 `(1-α)` 是输入耦合系数。
        若把循环电流 `W_rec·s` 与外部输入 `W_in·x` 相加后再送进神经元，循环项
        也会被乘上 `(1-α)`；而文献写法 `u = α·u + W_in·x + W_rec·s` 不缩放。
        因此 `rec_gain=1` 在两种写法下相差 `1/(1-α)` 倍。

        本项目的 τ 是逐神经元异构的（1–40 帧），`1-α` 跨 0.025–0.632（约 25 倍），
        所以缩放必须**逐神经元**，不能用标量。

        只有输入形如 `(1-α)·x` 的变体能给出精确值（LIF / ALIF / AdLIF）。
        GLIF 的输入是门控 `gi·(x·gt) + (1-gi)·x`、MC-LIF 是 `(1-α_d)` 与 `(1-β)`
        两级串联，都不存在单一系数，故返回 None——它们的循环连接需要单独标定。
        """
        return None

    def extra_repr(self) -> str:
        return (f"thr={self.thr}, reset={self.reset}, "
                f"surrogate={self.surrogate}(alpha={self.surrogate_alpha}), n_state={self.n_state}")


def init_param(value, n: int, learnable: bool = True) -> Tensor:
    """把标量或数组初始化成每神经元的参数。"""
    t = torch.as_tensor(value, dtype=torch.float32)
    if t.numel() == 1:
        t = t.repeat(n)
    if t.numel() != n:
        raise ValueError(f"参数长度 {t.numel()} 与神经元数 {n} 不符")
    if learnable:
        return nn.Parameter(t.clone())
    return t.clone()


def decay_from_tau(tau_ms: float, dt_ms: float) -> float:
    return math.exp(-dt_ms / tau_ms)
