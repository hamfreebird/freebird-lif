"""TC-LIF 与 AT-LIF：两隔室神经元（阶段四）。

方程取自 TC-LIF 原文（arXiv:2308.13250v3，AAAI 2024）式 (10)–(12)：

    U^D[t] = U^D[t-1] + β1·U^S[t-1] + I[t] - γ·S[t-1]
    U^S[t] = U^S[t-1] + β2·U^D[t]        - θ·S[t-1]
    S[t]   = H(U^S[t] - θ)
    β1 = -σ(c1) ∈ (-1, 0),   β2 = σ(c2) ∈ (0, 1)

与 LIF 的三个结构性差别，也是本阶段要检验的东西：

1. **两个隔室都没有衰减因子**。作者刻意去掉 α，使 `U^D` 成为不衰减的积分器，
   承担长程记忆。代价是它同时没有吸引子：无输入、无发放时线性化矩阵

       M = [[1, β1], [β2, 1 + β1·β2]]

   **det(M) = 1**，trace = 2 + β1β2 ∈ (1, 2)，特征值是一对模为 1 的共轭复根
   ——一个**无阻尼旋转**。`linear_spectrum()` 把这件事暴露给测试与诊断，
   不让它停留在纸面推导上。
2. **β1 与 β2 反号**，形成负反馈，抑制持续输入导致的过度发放。
3. **复位分两路**：胞体按 θ 复位，树突按 γ 复位（模拟反向传播的脉冲）。

AT-LIF 在 TC-LIF 上叠加 AdLIF 的适应电流 `w`（带泄漏的慢变量）。动机有两条：

- **耗散**：把中性稳定的旋转变成有吸引子的动力学，避免漂移或爆炸；
- **增益控制**：持续输入下 `w` 抬高、压低胞体的净驱动，使 `U^S` 不至于长期停在
  阈值之上——这正是「不丢失梯度对输入幅值的敏感性」的机制来源（PLAN-v4 §2.3）。

两个必须在阶段四报告里交代的接口事实：

- **`tau_ms` 被接受但被忽略**。TC-LIF 没有衰减因子，因此阶段三的「层内异构 τ」
  配方在这一变体上**不适用**。保留参数只为接口兼容；如果静默丢弃，会让
  「异构 τ 开/关」的对照变成假对照。M1 的基线必须相应换成「固定 τ 的 LIF」，
  否则比较的是两个不同配方。
- **`in_scale` 不能沿用 5**。`U^D` 不衰减，输入是直接累加而不是乘过 `(1-α)`；
  同样的 `in_scale` 会让胞体长期停在阈值之上。M0 用发放率匹配重新标定。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .base import SpikingNeuron
from .neurons import _alpha_param, _decay_of


def _logit(p: float) -> float:
    """把概率映射回 logit，便于把 β 的初值写成可读形式。"""
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def _logit_tensor(p: Tensor) -> Tensor:
    """张量版 logit，用于逐神经元异构初始化。"""
    return torch.log(p.clamp(1e-4, 1 - 1e-4) / (1.0 - p.clamp(1e-4, 1 - 1e-4)))


class TCLIF(SpikingNeuron):
    """两隔室 LIF（TC-LIF）。状态：`(U^D, U^S, s_prev)`。

    参数 `c1`、`c2`（决定 β1、β2）与 `γ`（树突复位强度）。三者在原文中均可学：
    β 由 sigmoid 约束在 (-1, 0) 与 (0, 1)。
    """

    n_state = 3
    float_state = 2          # U^D、U^S
    soma_index = 1           # state = (U^D, U^S, s_prev)，胞体在 1

    def __init__(self, n: int, tau_ms=2.0, dt_ms: float = 1.0,
                 learn_tau: bool = True, beta1: float = 0.1, beta2: float = 0.2,
                 gamma: float = 0.5, learn_gate: bool | None = None,
                 intrinsic_het: bool = False, period_lo: float = 8.0,
                 period_hi: float = 80.0, **kw):
        """`tau_ms`/`dt_ms`/`learn_tau` 仅为接口兼容而接受，**在本变体中被忽略**。

        TC-LIF 没有衰减因子，时间尺度由 β1·β2 决定。把这三个参数留着是为了让
        `build_neuron(..., tau_ms=..., learn_tau=...)` 的调用方式对所有变体一致；
        静默忽略是有意的，并在模块 docstring 里写明，避免"以为开了异构 τ"。
        """
        super().__init__(**kw)
        if learn_gate is None:
            learn_gate = learn_tau
        if intrinsic_het:
            # 逐神经元的**内在周期谱**：T 在 [period_lo, period_hi] 对数铺开，
            # 再由 cosθ = 1 + β1β2/2 反解 |β1β2| = 2(1-cos(2π/T))。
            # 取 β1=-β、β2=+β（β=sqrt(|β1β2|)），既满足原文的反号约束，
            # 又让 det=1 的中性稳定性质逐神经元保留——只是各自的旋转速度不同。
            # 这是「补快的那一端」：同构时所有神经元都只有一个 44 帧周期，
            # 无法表示短于 ~6 帧的瞬态。
            periods = torch.logspace(math.log10(period_lo), math.log10(period_hi), n)
            mag = torch.sqrt(torch.clamp(
                2.0 * (1.0 - torch.cos(2.0 * math.pi / periods)), 1e-6, 0.99))
            c1_init = _logit_tensor(mag)
            c2_init = _logit_tensor(mag)
        else:
            c1_init = torch.full((n,), _logit(beta1))
            c2_init = torch.full((n,), _logit(beta2))
        self.c1 = nn.Parameter(c1_init, requires_grad=learn_gate)
        self.c2 = nn.Parameter(c2_init, requires_grad=learn_gate)
        self.gamma = nn.Parameter(torch.full((n,), float(gamma)),
                                  requires_grad=learn_gate)

    # ---- 参数到系数的映射 -------------------------------------------------

    @property
    def beta1(self) -> Tensor:
        """∈ (-1, 0)：树突受到的胞体反馈（负反馈）。"""
        return -torch.sigmoid(self.c1)

    @property
    def beta2(self) -> Tensor:
        """∈ (0, 1)：胞体受到的树突驱动。"""
        return torch.sigmoid(self.c2)

    def input_scale(self) -> Tensor:
        """输入以系数 1 直接进入 `U^D`（没有 `(1-α)` 这一层），故无需缩放。

        与 LIF/ALIF 的 `1/(1-α)` 形成对照：这里的输入耦合是**无衰减累加**。
        这既是长程记忆的来源，也是容易饱和的原因。
        """
        return torch.ones_like(self.c1)

    def init_state(self, batch: int, features: int, device=None, dtype=None) -> tuple:
        z = torch.zeros(batch, features, device=device, dtype=dtype)
        return (z, z.clone(), z.clone())      # (U^D, U^S, s_prev)

    def step(self, x: Tensor, state: tuple) -> tuple[Tensor, tuple]:
        u_d, u_s, s_prev = state
        b1, b2 = self.beta1, self.beta2
        u_d = u_d + b1 * u_s + x - self.gamma * s_prev
        u_s = u_s + b2 * u_d - self.thr * s_prev
        s = self._fire(u_s)
        return s, (u_d, u_s, s)

    def op_count(self) -> float:
        # 2 个 sigmoid（各约 4 次算术）+ 4 乘 4 加
        return 14.0

    # ---- 分析用（不进前向） -----------------------------------------------

    def period_spectrum(self) -> Tensor:
        """逐神经元的内在周期（帧）。同构时全为同一个值。"""
        with torch.no_grad():
            prod = (self.beta1 * self.beta2).clamp(-0.999, -1e-6)
            theta = torch.acos((1.0 + prod / 2.0).clamp(-0.999, 0.999))
        return 2.0 * math.pi / theta

    def linear_spectrum(self) -> dict:
        """无输入、无发放时线性化矩阵的谱。

        TC-LIF 的判别性性质是 `det ≡ 1` 且特征值模全为 1——这不是巧合，
        而是「没有衰减因子」的直接后果。测试用它钉住实现，诊断脚本用它解释漂移。
        """
        with torch.no_grad():
            b1 = float(self.beta1.mean())
            b2 = float(self.beta2.mean())
        det = 1.0
        trace = 2.0 + b1 * b2
        disc = trace * trace - 4.0 * det
        if disc < 0:
            modulus = math.sqrt(det)
            angle = math.acos(max(min(trace / 2.0, 1.0), -1.0))
            period = 2.0 * math.pi / angle if angle > 0 else float("inf")
        else:
            modulus = float("nan")
            angle = 0.0
            period = float("inf")
        return {"beta1": b1, "beta2": b2, "det": det, "trace": trace,
                "discriminant": disc, "modulus": modulus,
                "angle_rad": angle, "period_frames": period}


class ATLIF(TCLIF):
    """TC-LIF + AdLIF 适应电流。状态：`(U^D, U^S, w, s_prev)`。

        U^D[t] = U^D[t-1] + β1·U^S[t-1] + I[t] - γ·S[t-1]
        U^S[t] = U^S[t-1] + β2·U^D[t] - w[t-1] - θ·S[t-1]
        w[t]   = β_w·w[t-1] + (1-β_w)·a·U^S[t-1] + b·S[t-1]

    `w` 用**上一时刻**的膜电位驱动（与 `neurons.AdLIF` 的写法一致），
    这样 `a=b=0` 时 AT-LIF 与 TC-LIF 逐时间步等价——退化性有测试钉住。
    """

    n_state = 4
    float_state = 3          # U^D、U^S、w
    soma_index = 1           # state = (U^D, U^S, w, s_prev)

    def __init__(self, n: int, tau_w_ms=10.0, dt_ms: float = 1.0,
                 a0: float = 0.0, b0: float = 0.0,
                 learn_a: bool = True, learn_b: bool = True, **kw):
        super().__init__(n, dt_ms=dt_ms, **kw)
        self.beta_w = _alpha_param(_decay_of(tau_w_ms, dt_ms), n, learn_a)
        self.a = nn.Parameter(torch.full((n,), float(a0)), requires_grad=learn_a)
        self.b = nn.Parameter(torch.full((n,), float(b0)), requires_grad=learn_b)
        self.learn_a = learn_a
        self.learn_b = learn_b

    def init_state(self, batch: int, features: int, device=None, dtype=None) -> tuple:
        z = torch.zeros(batch, features, device=device, dtype=dtype)
        return (z, z.clone(), z.clone(), z.clone())   # (U^D, U^S, w, s_prev)

    def step(self, x: Tensor, state: tuple) -> tuple[Tensor, tuple]:
        u_d, u_s, w, s_prev = state
        b1, b2 = self.beta1, self.beta2
        a_w = torch.sigmoid(self.beta_w)
        u_s_prev = u_s                                  # 适应电流取 t-1 的膜电位
        u_d = u_d + b1 * u_s + x - self.gamma * s_prev
        u_s = u_s + b2 * u_d - w - self.thr * s_prev
        w = a_w * w + (1.0 - a_w) * self.a * u_s_prev + self.b * s_prev
        s = self._fire(u_s)
        return s, (u_d, u_s, w, s)

    def op_count(self) -> float:
        return 20.0
