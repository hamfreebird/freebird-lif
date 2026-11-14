"""LIF 神经元与替代梯度。

微分方程（PLAN.md §7.1）：
    τ_m · du/dt = -(u - u_rest) + R·I(t)

两种离散化在数学上共享同一个更新式，仅 decay 的定义不同：

    u_next = decay · (u - u_rest) + u_rest + (1 - decay) · R·I

    euler       : decay = 1 - dt/τ_m       （一阶精度，O(dt) 截断误差）
    exponential : decay = exp(-dt/τ_m)     （零阶保持下的精确解，对恒定输入无截断误差）

该统一形式使得两种方法的稳态严格相同：u_∞ = u_rest + R·I。
这是 tests/test_neuron.py 中对解析解对拍的依据。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

SURROGATES = ("fast_sigmoid", "atan", "triangle", "rect")


def surrogate_derivative(x: Tensor, alpha: float, kind: str = "fast_sigmoid") -> Tensor:
    """替代导数 g(x)，x 为 (u - θ)。

    四种实现均已归一化，满足 ∫ g(x) dx = 1，即形态上等价于某个平滑阶跃函数的导数。
    这一性质由 tests/test_neuron.py::test_surrogate_integrates_to_one 断言保证，
    使得不同 surrogate 之间仅比较"陡峭度"，不混淆"尺度"。
    """
    if kind == "fast_sigmoid":
        # Zenke & Ganguli (SuperSpike) 形式
        return alpha / (2.0 * (1.0 + alpha * x.abs()).pow(2))
    if kind == "atan":
        return (alpha / 2.0) / (1.0 + (math.pi * alpha * x / 2.0).pow(2))
    if kind == "triangle":
        return (1.0 / alpha) * (1.0 - x.abs() / alpha).clamp(min=0.0)
    if kind == "rect":
        return (x.abs() < alpha).to(x.dtype) / (2.0 * alpha)
    raise ValueError(f"未知 surrogate: {kind!r}，可选 {SURROGATES}")


class _Spike(torch.autograd.Function):
    """前向 Heaviside，反向用替代导数。"""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float, kind: str) -> Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.kind = kind
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        return grad_output * surrogate_derivative(x, ctx.alpha, ctx.kind), None, None


def spike(x: Tensor, alpha: float = 2.0, kind: str = "fast_sigmoid") -> Tensor:
    """x 为 (u - θ)。返回 0/1 脉冲张量。"""
    return _Spike.apply(x, alpha, kind)


class LIF(nn.Module):
    """单个 LIF 层。

    无状态模块：膜电位由调用方持有并逐时间步传入，便于 BPTT 展开与推理引擎复用。

    spiking=False 时退化为纯泄漏积分器（无阈值、无重置），用于读出层。
    读出层必须如此——硬重置会把已累积的类别证据清零，直接摧毁分类信息。
    """

    def __init__(
        self,
        tau: float = 2.0,
        dt: float = 1.0,
        thr: float = 1.0,
        u_rest: float = 0.0,
        alpha: float = 2.0,
        surrogate: str = "fast_sigmoid",
        reset: str = "hard",
        method: str = "euler",
        spiking: bool = True,
        refractory: int = 0,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau 必须为正")
        if not 0 < dt <= tau:
            raise ValueError(f"要求 0 < dt <= tau，当前 dt={dt}, tau={tau}")
        if reset not in ("hard", "soft"):
            raise ValueError(f"reset 必须是 'hard' 或 'soft'，当前 {reset!r}")
        if surrogate not in SURROGATES:
            raise ValueError(f"未知 surrogate {surrogate!r}")

        self.tau = float(tau)
        self.dt = float(dt)
        self.thr = float(thr)
        self.u_rest = float(u_rest)
        self.alpha = float(alpha)
        self.surrogate = surrogate
        self.reset = reset
        self.method = method
        self.spiking = bool(spiking)
        self.refractory = int(refractory)

        if method == "euler":
            self.decay = 1.0 - dt / tau
        elif method == "exponential":
            self.decay = math.exp(-dt / tau)
        else:
            raise ValueError(f"method 必须是 'euler' 或 'exponential'，当前 {method!r}")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError(f"decay={self.decay} 越界，检查 dt/tau")

        self.gain = 1.0 - self.decay

    def extra_repr(self) -> str:
        return (
            f"tau={self.tau}, dt={self.dt}, thr={self.thr}, decay={self.decay:.4f}, "
            f"reset={self.reset}, method={self.method}, "
            f"spiking={self.spiking}, surrogate={self.surrogate}(alpha={self.alpha})"
        )

    def init_state(self, *shape, device=None, dtype=None):
        """返回 (膜电位, 不应期计数)。refractory=0 时第二项为 None。"""
        u = torch.zeros(shape, device=device, dtype=dtype)
        ref = torch.zeros(shape, device=device, dtype=torch.int16) if self.refractory else None
        return u, ref

    def forward(self, x: Tensor, u: Tensor, ref: Tensor | None = None):
        """单步推进。返回 (输出, 新膜电位, 新不应期计数)。

        无脉冲时"输出"即膜电位，使上层逻辑无需区分是否脉冲。
        """
        # 泄漏 + 积分
        u = self.decay * (u - self.u_rest) + self.u_rest + self.gain * x

        if ref is not None:
            # 不应期内膜电位钳位于静息态，且不参与阈值判定
            blocked = ref > 0
            u = torch.where(blocked, torch.full_like(u, self.u_rest), u)

        if not self.spiking:
            return u, u, ref

        s = spike(u - self.thr, self.alpha, self.surrogate)

        if self.reset == "hard":
            u = torch.where(s > 0, torch.full_like(u, self.u_rest), u)
        else:
            # 软重置：仅减去阈值。s.detach() 是刻意的——重置项不应再贡献一条
            # 经由 surrogate 的梯度路径，否则替代梯度会被重复计入。
            u = u - s.detach() * self.thr

        if ref is not None:
            ref = torch.where(s > 0, torch.full_like(ref, self.refractory), (ref - 1).clamp(min=0))
            ref = ref.to(torch.int16)

        return s, u, ref
