"""脉冲神经元变体的统一接口。

设计约束来自实验需求：六个变体必须**只差在神经元方程**，其余全部相同。
因此接口只暴露状态与单步推进，不掺入任何层结构或优化逻辑。

两类成本指标是并行的：
    n_state   每神经元的浮点状态个数（LIF=1，AdLIF=2，MC-LIF=1+K）
    op_count  每神经元每步的算术操作数（含 sigmoid 折算）
一个"精度 +1% 但成本 2×"的变体是坏交易，所以成本必须和精度一起报。
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
