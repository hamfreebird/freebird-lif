"""五个 LIF 变体的具体实现。

每个类只实现"与 LIF 不同的那一部分"，公共的阈值判定与重置在 base 里。
方程出处均标注在原位，不做无出处的改动。

第六个变体 POLARA 因原文未获取而暂缺，见 polara.py，不用猜测版本充数。

操作数是**估算值**，用于相对比较，不是精确的 FLOP 计数。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .base import SpikingNeuron, decay_from_tau


def _decay_of(tau_ms, dt_ms: float):
    """τ（标量或每神经元数组，单位 ms）→ α = exp(-dt/τ)。"""
    t = torch.as_tensor(tau_ms, dtype=torch.float32)
    if t.dim() == 0:
        return float(decay_from_tau(float(t), dt_ms))
    return torch.exp(-dt_ms / t.clamp_min(1e-3))


def _alpha_param(a0, n: int, learn: bool) -> nn.Parameter:
    """用 logit 参数化保证 α ∈ (0,1)；不可学习时也保留 Parameter 以便统一 .to(device)。

    a0 可以是标量（全体同 τ）或长度 n 的数组（异构 τ）。异构初始化是本阶段的
    重点实验：同一层内从快到慢铺开一组时间常数，等价于一个可学的滤波器组，
    比全体共享一个 τ 有更强的时间特征表达力。
    """
    a = torch.as_tensor(a0, dtype=torch.float32).reshape(-1)
    if a.numel() == 1:
        a = a.repeat(n)
    if a.numel() != n:
        raise ValueError(f"α 初始化长度 {a.numel()} 与神经元数 {n} 不符")
    a = a.clamp(1e-4, 1.0 - 1e-4)
    # 两种模式都存 **logit**，因为 `decay` 属性统一对 alpha_u 做 sigmoid。
    # 早期版本在 learn=False 时直接存 α 本身，导致 decay 返回 sigmoid(α)——
    # τ=2 帧的衰减变成 sigmoid(0.6065)=0.6471（等效 τ≈2.3 帧）。
    # 这是静默偏差：不影响 learn=True 的训练（M2 全部走这条路径），
    # 但会让任何"固定 τ"的对照实验悄悄偏离声明值。
    raw = torch.log(a / (1.0 - a))
    return nn.Parameter(raw.clone(), requires_grad=learn)


class LIF(SpikingNeuron):
    """基线。

        u[t] = α·u[t-1] + (1-α)·x[t] - θ·s[t-1]
        s[t] = H(u[t] - θ)
        α = exp(-dt/τ)

    注意膜电位**存储的是重置前的值**，重置以 `-θ·s[t-1]` 的形式在下一步的更新里
    完成（subtractive reset）。这是 Bellec 2018 / Bittar & Garner 2022 的写法，
    与"存重置后值、判定后就地清零"的写法在数学上不等价（相差 (1-α)θs）。
    四个变体统一采用本形式，退化性才成立。

    τ 可学习时每神经元 1 个参数。
    """

    n_state = 2        # (u, s_prev)
    float_state = 1    # u

    def __init__(self, n: int, tau_ms=2.0, dt_ms: float = 1.0,
                 learn_tau: bool = False, **kw):
        super().__init__(**kw)
        self.alpha_u = _alpha_param(_decay_of(tau_ms, dt_ms), n, learn_tau)

    @property
    def decay(self) -> Tensor:
        return torch.sigmoid(self.alpha_u)

    def init_state(self, batch: int, features: int, device=None, dtype=None) -> tuple:
        z = torch.zeros(batch, features, device=device, dtype=dtype)
        return (z, z.clone())  # (u_pre_reset, s_prev)

    def step(self, x: Tensor, state: tuple) -> tuple[Tensor, tuple]:
        u, s_prev = state
        a = self.decay
        u = a * u + (1.0 - a) * x - self.thr * s_prev
        s = self._fire(u)
        return s, (u, s)

    def op_count(self) -> float:
        return 7.0


class ALIF(SpikingNeuron):
    """自适应阈值（Bellec et al. 2018）。

        u[t] = α·u[t-1] + (1-α)·x[t] - θ·s[t-1]
        b[t] = ρ·b[t-1] + (1-ρ)·s[t-1]
        s[t] = H(u[t] - (θ + β·b[t]))

    与 AdLIF 的关键区别：这里抬高的是**阈值**，只依赖发放历史；
    AdLIF 用的是**适应电流**，同时受膜电位与发放影响，因此能产生
    膜电位振荡（class-2 兴奋性），而自适应阈值不能。
    """

    n_state = 3       # (u, b, s_prev)
    float_state = 2   # u, b

    def __init__(self, n: int, tau_ms=2.0, dt_ms: float = 1.0,
                 tau_b_ms: float = 10.0, beta: float = 0.5,
                 learn_tau: bool = False, learn_beta: bool = True, **kw):
        super().__init__(**kw)
        self.alpha_u = _alpha_param(_decay_of(tau_ms, dt_ms), n, learn_tau)
        self.rho = float(decay_from_tau(tau_b_ms, dt_ms))
        self.beta_b = nn.Parameter(torch.full((n,), float(beta)), requires_grad=learn_beta)

    @property
    def decay(self) -> Tensor:
        return torch.sigmoid(self.alpha_u)

    def init_state(self, batch: int, features: int, device=None, dtype=None) -> tuple:
        z = torch.zeros(batch, features, device=device, dtype=dtype)
        return (z, z.clone(), z.clone())

    def step(self, x: Tensor, state: tuple) -> tuple[Tensor, tuple]:
        u, b, s_prev = state
        a = self.decay
        u = a * u + (1.0 - a) * x - self.thr * s_prev
        b = self.rho * b + (1.0 - self.rho) * s_prev
        # 阈值判定用抬高后的阈值；重置仍按基准 θ
        s = self._fire(u - self.beta_b * b)
        return s, (u, b, s)

    def op_count(self) -> float:
        return 13.0


class AdLIF(SpikingNeuron):
    """自适应电流（Bittar & Garner 2022；离散式见 Deckers et al. 2024 式(1)）。

        u[t] = α·u[t-1] + (1-α)·(x[t] - w[t-1]) - θ·s[t-1]
        w[t] = β·w[t-1] + (1-β)·a·u[t-1] + b·s[t-1]
        s[t] = H(u[t] - θ)

    四个可学习神经元参数：α, β, a, b。SHD 上这是最强基线之一，
    但**文献数字属于循环结构**：RSNN 的 LIF 87.04% → RadLIF(adLIF) 94.62%
    （Bittar & Garner 2022，141k / 3.9M 参数）。早期版本误记为"前馈 3×128"，
    见 PLAN §3.3.1 的勘误。
    """

    n_state = 3       # (u, w, s_prev)
    float_state = 2   # u, w

    def __init__(self, n: int, tau_ms=2.0, tau_w_ms=10.0,
                 dt_ms: float = 1.0, a0: float = 0.0, b0: float = 0.0,
                 learn_tau: bool = True, learn_a: bool = True,
                 learn_b: bool = True, **kw):
        super().__init__(**kw)
        self.alpha_u = _alpha_param(_decay_of(tau_ms, dt_ms), n, learn_tau)
        self.beta_w = _alpha_param(_decay_of(tau_w_ms, dt_ms), n, learn_tau)
        # 两项适应电流的驱动源不同，消融时分别冻结：
        #   a·u[t-1]  膜电位驱动，构成 u→w→u 的二阶反馈，产生 class-2 兴奋性（共振）
        #   b·s[t-1]  发放驱动，是经典的脉冲频率适应
        # 冻结时保持 a0/b0=0 且 requires_grad=False，等价于该机制完全断开。
        self.a = nn.Parameter(torch.full((n,), float(a0)), requires_grad=learn_a)
        self.b = nn.Parameter(torch.full((n,), float(b0)), requires_grad=learn_b)
        self.learn_a = learn_a
        self.learn_b = learn_b

    def init_state(self, batch: int, features: int, device=None, dtype=None) -> tuple:
        z = torch.zeros(batch, features, device=device, dtype=dtype)
        return (z, z.clone(), z.clone())

    def step(self, x: Tensor, state: tuple) -> tuple[Tensor, tuple]:
        u, w, s_prev = state
        a_u = torch.sigmoid(self.alpha_u)
        a_w = torch.sigmoid(self.beta_w)
        u_new = a_u * u + (1.0 - a_u) * (x - w) - self.thr * s_prev
        w_new = a_w * w + (1.0 - a_w) * self.a * u + self.b * s_prev
        s = self._fire(u_new)
        return s, (u_new, w_new, s)

    def op_count(self) -> float:
        return 16.0


class GLIF(SpikingNeuron):
    """门控 LIF（Yao et al., NeurIPS 2022）。

    三种神经元行为各用一个可学习门，做两个"原语"的凸组合：

        U[t] = L[t] + I[t] + F[t] ⊙ s[t-1]
        L[t] = G_γl(U[t-1]; 线性衰减 U-θ·0.1, 指数衰减 α·U)
        I[t] = G_γi(x[t]·g^t, x[t])            动态增益 vs 静态增益
        F[t] = G_γr(-θ, 0)                     软重置 vs 硬重置
        G(·; p, q) = σ(γ)·p + (1-σ(γ))·q

    每神经元 3 个门控参数。`g^t` 通过状态里携带的 gt 逐 step 乘 g 实现。

    **参数化与官方仓库 Ikarosy/Gated-LIF 可能存在细节差异**，本实现忠于
    论文的三门结构意图，报告中会注明这一点。
    """

    n_state = 3       # (u, s_prev, gt[n])
    float_state = 2   # u, gt

    def __init__(self, n: int, tau_ms=2.0, dt_ms: float = 1.0,
                 learn_tau: bool = True, **kw):
        super().__init__(**kw)
        self.alpha_u = _alpha_param(_decay_of(tau_ms, dt_ms), n, learn_tau)
        self.g_leak = nn.Parameter(torch.zeros(n))
        self.g_in = nn.Parameter(torch.zeros(n))
        self.g_reset = nn.Parameter(torch.zeros(n))
        self.g_decay = nn.Parameter(torch.full((n,), 0.9))

    def init_state(self, batch: int, features: int, device=None, dtype=None) -> tuple:
        z = torch.zeros(batch, features, device=device, dtype=dtype)
        gt = torch.ones(features, device=device, dtype=dtype)
        return (z, z.clone(), gt)

    def step(self, x: Tensor, state: tuple) -> tuple[Tensor, tuple]:
        u, s_prev, gt = state
        a = torch.sigmoid(self.alpha_u)
        gl = torch.sigmoid(self.g_leak)
        gi = torch.sigmoid(self.g_in)
        gr = torch.sigmoid(self.g_reset)

        leak = gl * (u - self.thr * 0.1) + (1.0 - gl) * (a * u)
        gt = gt * self.g_decay
        inp = gi * (x * gt) + (1.0 - gi) * x
        rst = gr * (-self.thr) + (1.0 - gr) * 0.0
        u_new = leak + inp + rst * s_prev
        s = self._fire(u_new)
        return s, (u_new, s, gt)

    def op_count(self) -> float:
        # 3 个 sigmoid 约各 4 次运算，另 3 组凸组合各 3 次
        return 27.0


class MCLIF(SpikingNeuron):
    """多房室 LIF（DH-LIF 结构，Nature Communications 2024）。

        d_k[t] = α_k·d_k[t-1] + (1-α_k)·x[t]        k = 1..K
        u[t]   = β·u[t-1] + (1-β)·Σ_k d_k[t] - θ·s[t-1]
        s[t]   = H(u[t] - θ)

    每个树突分支有独立时间常数，对数均匀初始化——这是 DH-LIF 的核心：
    不同分支记住不同时间尺度的输入，使单个神经元具备多尺度记忆。

    退化性：K=1 且 α_1=1 时 d[t]=x[t]，严格退化为 LIF（单元测试据此校验）。
    """

    def __init__(self, n: int, n_branch: int = 2, tau_ms=2.0,
                 tau_d_range_ms: tuple[float, float] = (1.0, 20.0),
                 dt_ms: float = 1.0, learn_tau: bool = True, **kw):
        super().__init__(**kw)
        self.K = int(n_branch)
        # n_state 计的是 init_state 返回的张量个数：膜电位 + 叠在一起的 K 个树突 + 上一步脉冲
        self.n_state = 3
        self.float_state = 1 + self.K   # u + K 个树突电位
        self.beta_u = _alpha_param(_decay_of(tau_ms, dt_ms), n, learn_tau)

        lo, hi = tau_d_range_ms
        taus = torch.logspace(math.log10(lo), math.log10(hi), self.K)
        a_d = torch.exp(-dt_ms / taus)
        raw = torch.log(a_d / (1 - a_d))                       # [K]
        self.alpha_d = nn.Parameter(raw.unsqueeze(1).repeat(1, n), requires_grad=learn_tau)

    def init_state(self, batch: int, features: int, device=None, dtype=None) -> tuple:
        u = torch.zeros(batch, features, device=device, dtype=dtype)
        d = torch.zeros(self.K, batch, features, device=device, dtype=dtype)
        return (u, d, u.clone())  # (u_pre_reset, d[K], s_prev)

    def step(self, x: Tensor, state: tuple) -> tuple[Tensor, tuple]:
        u, d, s_prev = state
        a_d = torch.sigmoid(self.alpha_d).unsqueeze(1)          # [K,1,n]
        d = a_d * d + (1.0 - a_d) * x.unsqueeze(0)
        beta = torch.sigmoid(self.beta_u)
        u = beta * u + (1.0 - beta) * d.sum(0) - self.thr * s_prev
        s = self._fire(u)
        return s, (u, d, s)

    def op_count(self) -> float:
        return 4.0 * self.K + 5.0
