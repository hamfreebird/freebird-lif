"""Conv SNN 前向、BN 折叠与统计口径对拍。"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from freebird_lif.network import ConvSNN, SNNConfig, confidence, fold_conv_bn

B = 4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = ConvSNN(SNNConfig(T=8))
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(1)
    x = torch.rand(B, 1, 16, 16)
    # 模拟真实输入的稀疏背景：四边清零（与数据缓存的边距不变量一致）
    x[:, :, :2, :] = 0
    x[:, :, 14:, :] = 0
    x[:, :, :, :2] = 0
    x[:, :, :, 14:] = 0
    return x


# ---- 结构与口径 ------------------------------------------------------------


def test_forward_shapes(model, batch):
    logits, traces = model(batch)
    assert logits.shape == (B, 10)
    assert len(traces) == model.cfg.T
    assert all(t.shape == (B, 10) for t in traces)


@pytest.mark.parametrize("pool_on", ["spike", "membrane"])
def test_pooling_modes_run(pool_on):
    m = ConvSNN(SNNConfig(pool_on=pool_on, T=4))
    m.eval()
    out, _ = m(torch.rand(2, 1, 16, 16))
    assert out.shape == (2, 10)


def test_param_count_matches_plan(model):
    """PLAN §6 的 40,666 为"无 BN"口径。实现上做了两处修正：

    - 移除 conv1/conv2/fc1 的偏置（紧接 BN，偏置在数学上被抵消，梯度恒为 0）
    - 在 fc1 后补 BN1d（缺失时 LIF3 初始化即死亡，见 network.py 注释）

    净变化：40,666 - 132（三个偏置）+ 192（BN3）+ 72（BN1/BN2）= 40,798
    """
    assert model.param_count(include_bn=False) == 40_534
    assert model.param_count(include_bn=True) == 40_798


def test_no_dead_parameters(model):
    """所有参数都应拿到非零梯度；BN 前的冗余偏置已全部移除。"""
    m = ConvSNN(SNNConfig(T=8))
    m.train()
    logits, _ = m(torch.rand(B, 1, 16, 16))
    F.cross_entropy(logits, torch.arange(B) % 10).backward()
    dead = [n for n, p in m.named_parameters() if p.grad.abs().max().item() == 0.0]
    assert dead == [], f"死亡参数: {dead}"


def test_macs_match_plan(model):
    m = model.macs_report()
    assert m == {"conv1": 27_648, "conv2": 165_888, "fc1": 36_864, "fc2": 960}
    assert sum(m.values()) == 231_360


def test_amortized_macs(model):
    """Conv1 因电流注入恒定而只计一次，总 MACs 应低于朴素 T 倍展开。"""
    assert model.amortized_macs(T=8) == 27_648 + 8 * 203_712
    assert model.amortized_macs(T=8) == 1_657_344
    assert model.amortized_macs(T=8) < 8 * 231_360


# ---- BN 折叠 ---------------------------------------------------------------


def test_fold_conv_bn_equivalence(model):
    """折叠后卷积必须与 BN(Conv(x)) 数值等价。"""
    x = torch.rand(8, 1, 16, 16)
    w, b = fold_conv_bn(model.conv1, model.bn1)
    ref = model.bn1(model.conv1(x))
    got = F.conv2d(x, w, b, padding=1)
    assert torch.allclose(ref, got, atol=1e-6)


def test_fold_preserves_zero_input(model):
    """全零输入经 BN 后应是常数 β - γμ/σ，折叠后须一致。"""
    x = torch.zeros(2, 1, 16, 16)
    w, b = fold_conv_bn(model.conv1, model.bn1)
    ref = model.bn1(model.conv1(x))
    got = F.conv2d(x, w, b, padding=1)
    assert torch.allclose(ref, got, atol=1e-6)


# ---- 缓存等价性 ------------------------------------------------------------


def _naive_forward(m: ConvSNN, x: torch.Tensor, T: int):
    """每步重算 Conv1 的参考实现，用于验证缓存不改变结果。"""
    state = m.init_states(x.shape[0], device=x.device, dtype=x.dtype)
    out = None
    for _ in range(T):
        out, _, state = m._step(m._conv_front(x * m.cfg.input_gain), state)
    return out


def test_cache_is_mathematically_exact(model, batch):
    cached, _ = model(batch)
    naive = _naive_forward(model, batch, model.cfg.T)
    assert torch.allclose(cached, naive, atol=1e-6)


def test_equivalent_input_forms(model, batch):
    flat, _ = model(batch.reshape(B, 256))
    img, _ = model(batch)
    assert torch.allclose(flat, img, atol=1e-6)


# ---- 数值行为 --------------------------------------------------------------


def test_deterministic_in_eval(model, batch):
    a, _ = model(batch)
    b, _ = model(batch)
    assert torch.equal(a, b)


def test_readout_is_nonspiking_and_unbounded(model, batch):
    assert model.lif4.spiking is False
    logits, _ = model(batch)
    # 若读出层带硬重置，值域会被压到 [0, thr] 附近；此处应可超出
    assert logits.max() > 1.0 or logits.min() < 0.0


def test_spikes_are_not_degenerate():
    """三层发放率都必须非零，否则深层通路在初始化即死亡。

    必须在 train 模式下检验：eval 模式下 BN 用的是 running stats，
    未训练时仍是初始值 (μ=0, σ=1)，与真实激活分布无关，会让 LIF 全不发放。
    训练态 BN 按批统计归一化，通道输出约为 N(0,1)，配合 thr=1 应产生稳定发放。
    """
    torch.manual_seed(0)
    m = ConvSNN(SNNConfig(T=8))
    m.train()
    rates = m.spike_rates(torch.rand(32, 1, 16, 16))
    for name, rate in rates.items():
        assert 0.005 < rate < 0.6, f"{name} 发放率 {rate:.4f} 退化"


def test_eval_mode_before_training_is_silent():
    """记录一条已知性质：未训练模型在 eval 下不发放。

    这不是缺陷，而是 BN running stats 尚未积累所致。M2 训练后该性质消失；
    此处显式断言，防止后续误把"静默"当成实现 bug 排查。
    """
    torch.manual_seed(0)
    m = ConvSNN(SNNConfig(T=8))
    m.eval()
    rates = m.spike_rates(torch.rand(B, 1, 16, 16))
    assert all(r == 0.0 for r in rates.values()), f"eval 下意外发放: {rates}"


def test_confidence_sums_to_one(model, batch):
    logits, _ = model(batch)
    pred, conf = confidence(logits)
    assert pred.shape == conf.shape == (B,)
    assert (conf > 0).all() and (conf <= 1).all()
    prob = torch.softmax(logits, dim=-1)
    assert torch.allclose(prob.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_spike_rates_api(model, batch):
    """诊断接口：三层齐全、取值在 [0,1]、且不改变模型状态。"""
    rates = model.spike_rates(batch)
    assert set(rates) == {"lif1", "lif2", "lif3"}
    for v in rates.values():
        assert 0.0 <= v <= 1.0
    # no_grad 调用不应污染 running stats
    before = model.bn1.running_mean.clone()
    model.spike_rates(batch)
    assert torch.equal(before, model.bn1.running_mean)


def test_gradients_finite(model, batch):
    m = ConvSNN(SNNConfig(T=8))
    m.train()
    logits, _ = m(batch)
    loss = F.cross_entropy(logits, torch.randperm(10)[:B])
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} 无梯度"
        assert torch.isfinite(p.grad).all(), f"{name} 梯度非有限"
    assert math.isfinite(loss.item()), "损失为 NaN/Inf"


def test_spike_input_path(model, batch):
    """脉冲输入通路（用于泊松/TTFS 对照）应可运行且形状一致。"""
    spikes = (torch.rand(8, B, 1, 16, 16) < 0.3).float()
    out, traces = model(batch, spikes=spikes)
    assert out.shape == (B, 10)
    assert len(traces) == 8


def test_spike_input_length_mismatch(model, batch):
    with pytest.raises(ValueError):
        model(batch, T=4, spikes=(torch.rand(8, B, 1, 16, 16) < 0.3).float())


def test_invalid_pool_mode_rejected():
    with pytest.raises(ValueError):
        ConvSNN(SNNConfig(pool_on="avg"))
