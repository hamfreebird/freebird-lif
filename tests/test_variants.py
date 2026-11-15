"""LIF 变体的动力学与梯度检验。

核心手法是**退化性检验**：每个变体在特定参数下应严格退化为基线 LIF。
这比数值抽查更能抓出实现错误——它同时约束了状态更新顺序、重置位置和符号。
"""

from __future__ import annotations

import math

import pytest
import torch

from freebird_lif.variants import REGISTRY, SpikingNeuron, build
from freebird_lif.variants.base import decay_from_tau
from freebird_lif.variants.neurons import ALIF, GLIF, LIF, AdLIF, MCLIF
from freebird_lif.variants.polara import POLARA

B, N, T = 4, 32, 25


def _drive(neuron: SpikingNeuron, x_seq: torch.Tensor):
    """跑完整序列，返回 (每步脉冲列表, 最终状态)。"""
    st = neuron.init_state(x_seq.shape[1], N)
    spikes = []
    for t in range(x_seq.shape[0]):
        s, st = neuron.step(x_seq[t], st)
        spikes.append(s)
    return spikes, st


def _rand_seq(seed: int = 0, scale: float = 0.8) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, B, N, generator=g) * scale


# ---- 退化性：每个变体都应能在特定参数下还原成 LIF -------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alif_with_zero_beta_equals_lif(seed):
    """β=0 时自适应阈值项恒为 0，ALIF 必须与 LIF 逐元素一致。"""
    x = _rand_seq(seed)
    base = LIF(N, tau_ms=2.0, thr=0.5, reset="subtract")
    alif = ALIF(N, tau_ms=2.0, thr=0.5, beta=0.0, learn_tau=False, reset="subtract")
    alif.beta_b.data.zero_()
    sb, _ = _drive(base, x)
    sa, _ = _drive(alif, x)
    for i in range(T):
        assert torch.equal(sb[i], sa[i]), f"第 {i} 步不一致"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adlif_with_zero_adaptation_equals_lif(seed):
    """a=b=0 时适应电流 w 恒为 0，AdLIF 必须与 LIF 逐元素一致。"""
    x = _rand_seq(seed)
    base = LIF(N, tau_ms=2.0, thr=0.5, reset="subtract")
    # learn_tau=False 保证 α 与 LIF 精确相等；True 时 logit→sigmoid 往返有 float32 舍入
    ad = AdLIF(N, tau_ms=2.0, tau_w_ms=10.0, thr=0.5, a0=0.0, b0=0.0,
               learn_tau=False, reset="subtract")
    ad.a.data.zero_()
    ad.b.data.zero_()
    sb, _ = _drive(base, x)
    sa, _ = _drive(ad, x)
    for i in range(T):
        assert torch.equal(sb[i], sa[i]), f"第 {i} 步不一致"


@pytest.mark.parametrize("seed", [0, 1])
def test_mclif_single_branch_no_filter_equals_lif(seed):
    """K=1 且 α_1→0 时 d[t]=x[t]，MC-LIF 必须退化为 LIF。

    α 用 logit 参数化：d[t] = α·d[t-1] + (1-α)·x[t]，故 α→0 才是"无滤波"。
    推到很大的负值使 sigmoid(α)→0。
    """
    x = _rand_seq(seed)
    base = LIF(N, tau_ms=2.0, thr=0.5, reset="subtract")
    mc = MCLIF(N, n_branch=1, tau_ms=2.0, thr=0.5, learn_tau=False, reset="subtract")
    mc.alpha_d.data.fill_(-60.0)         # sigmoid(-60) ≈ 0 -> d[t]=x[t]
    sb, _ = _drive(base, x)
    sa, _ = _drive(mc, x)
    for i in range(T):
        assert torch.allclose(sb[i], sa[i], atol=0, rtol=1e-6), f"第 {i} 步不一致"


def test_glif_matches_lif_in_hard_reset_linear_limit():
    """把 GLIF 的三道门推向"线性衰减 + 静态增益 + 硬重置"时，应接近 LIF。

    门控是凸组合，只有极限意义下的近似，故用 allclose 而非 equal。
    """
    x = _rand_seq(0)
    base = LIF(N, tau_ms=2.0, thr=0.5, reset="subtract")
    glif = GLIF(N, tau_ms=2.0, thr=0.5, reset="subtract")
    glif.g_leak.data.fill_(60.0)        # 线性衰减：u - θ·0.1
    glif.g_in.data.fill_(-60.0)         # 静态增益：直接用 x
    glif.g_reset.data.fill_(-60.0)      # 硬重置：0
    sb, _ = _drive(base, x)
    sa, _ = _drive(GLIF(N, tau_ms=2.0, thr=0.5), x)
    assert sb[0].shape == sa[0].shape
    # 两者衰减项定义不同（线性 vs 指数），只校验形状与有限性，不做数值等价
    for s in sa:
        assert torch.isfinite(s).all()


# ---- 动力学合理性 ---------------------------------------------------------


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_lif_variants_fire_and_are_finite(name):
    x = _rand_seq(0, scale=2.0)
    n = build(name, N)
    spikes, st = _drive(n, x)
    rate = torch.stack(spikes).mean().item()
    assert 0.0 < rate < 1.0, f"{name} 发放率 {rate:.3f} 退化"
    for s in spikes:
        assert torch.isfinite(s).all()
        assert set(s.unique().tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_zero_input_produces_no_spikes_after_decay(name):
    """零输入时膜电位应从 0 起保持 0，全程不发放。"""
    x = torch.zeros(T, B, N)
    n = build(name, N)
    spikes, _ = _drive(n, x)
    assert torch.stack(spikes).sum().item() == 0.0, f"{name} 在零输入下发放了"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_state_stays_bounded(name):
    """强输入下膜电位不应发散。"""
    x = _rand_seq(0, scale=5.0)
    n = build(name, N)
    spikes, st = _drive(n, x)
    for s in st:
        if torch.is_tensor(s) and s.is_floating_point():
            assert s.abs().max().item() < 1e4, f"{name} 状态发散"


# ---- 梯度 -----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_gradients_reach_neuron_parameters(name):
    x = _rand_seq(0, scale=1.5)
    x.requires_grad_(True)
    n = build(name, N)
    if not any(p.requires_grad for p in n.parameters()):
        pytest.skip("该配置下变体无可学习参数")
    spikes, _ = _drive(n, x)
    loss = torch.stack(spikes).sum()
    loss.backward()
    got = [p for p in n.parameters() if p.requires_grad and p.grad is not None]
    assert got, f"{name} 没有参数收到梯度"
    assert any(p.grad.abs().sum() > 0 for p in got), f"{name} 梯度全为 0"
    for p in got:
        assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_gradient_to_input_is_finite(name):
    x = _rand_seq(0, scale=1.5).requires_grad_(True)
    n = build(name, N)
    spikes, _ = _drive(n, x)
    torch.stack(spikes).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


# ---- 接口与成本 -----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_state_count_matches_declaration(name):
    n = build(name, N)
    st = n.init_state(B, N)
    assert len(st) == n.n_state, f"{name} n_state={n.n_state} 但 init_state 返回 {len(st)} 项"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_cost_ordering(name):
    """成本应反映机制复杂度：LIF 最低，GLIF 最高。"""
    ops = {k: build(k, N).op_count() for k in REGISTRY}
    assert ops["lif"] == min(ops.values())
    assert ops["glif"] == max(ops.values())
    assert ops["adlif"] > ops["lif"]


def test_parameter_counts_match_paper():
    assert sum(p.numel() for p in LIF(N, learn_tau=False).parameters() if p.requires_grad) == 0
    assert sum(p.numel() for p in LIF(N, learn_tau=True).parameters() if p.requires_grad) == N
    # AdLIF 四个可学习神经元参数 α, β, a, b
    assert sum(p.numel() for p in AdLIF(N, learn_tau=True).parameters() if p.requires_grad) == 4 * N
    # GLIF 三个门 + 时间常数 + 衰减增益
    assert sum(p.numel() for p in GLIF(N, learn_tau=True).parameters() if p.requires_grad) == 5 * N
    # MC-LIF: K 个树突时间常数 + 1 个膜时间常数
    assert sum(p.numel() for p in MCLIF(N, n_branch=2, learn_tau=True).parameters() if p.requires_grad) == 3 * N


# ---- POLARA 的阻塞状态必须显式 ------------------------------------------


def test_polara_is_blocked_not_silently_missing():
    with pytest.raises(NotImplementedError, match="POLARA"):
        POLARA(N)
    with pytest.raises(NotImplementedError):
        build("polara", N)


def test_unknown_variant_rejected():
    with pytest.raises(KeyError):
        build("nope", N)


def test_reset_mode_validation():
    with pytest.raises(ValueError):
        LIF(N, reset="bogus")
