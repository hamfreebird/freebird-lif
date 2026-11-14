"""LIF 动力学与替代梯度对拍。M1 的核心判据。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from freebird_lif.neuron import SURROGATES, LIF, spike, surrogate_derivative


def _drive(lif: LIF, x: float, steps: int, shape=(1,)):
    """恒定输入驱动，返回逐步膜电位（不含脉冲的时刻用 after-reset 值）。"""
    u, ref = lif.init_state(*shape, dtype=torch.float64)
    hist = []
    for _ in range(steps):
        s, u, ref = lif(x=torch.full(shape, x, dtype=torch.float64), u=u, ref=ref)
        hist.append((u.clone(), s.clone()))
    return hist


# ---- 离散化与解析解 --------------------------------------------------------


@pytest.mark.parametrize("method", ["euler", "exponential"])
def test_decay_definition(method):
    tau, dt = 4.0, 1.0
    lif = LIF(tau=tau, dt=dt, method=method)
    expect = 1 - dt / tau if method == "euler" else math.exp(-dt / tau)
    assert lif.decay == pytest.approx(expect)
    assert lif.gain == pytest.approx(1 - expect)


@pytest.mark.parametrize("method", ["euler", "exponential"])
def test_subthreshold_steady_state(method):
    """两种离散化必须收敛到同一稳态 u_rest + R·I。"""
    thr, x = 10.0, 0.5  # 阈值远高于稳态，全程不发放
    lif = LIF(tau=2.0, dt=1.0, thr=thr, method=method)
    hist = _drive(lif, x, steps=60)
    assert all(s.item() == 0 for _, s in hist), "不应出现脉冲"
    assert hist[-1][0].item() == pytest.approx(x, abs=1e-9)


def test_exponential_matches_analytic():
    """指数解对恒定输入无截断误差，应逐元素吻合闭式解。"""
    tau, dt, u_rest, x = 3.0, 0.5, -0.2, 0.7
    lif = LIF(tau=tau, dt=dt, thr=99.0, u_rest=u_rest, method="exponential")
    u_inf = u_rest + x
    hist = _drive(lif, x, steps=20)

    u0 = 0.0
    for n, (u, _) in enumerate(hist, start=1):
        expect = u_inf + (u0 - u_inf) * math.exp(-n * dt / tau)
        assert u.item() == pytest.approx(expect, abs=1e-10), f"第 {n} 步偏离"


def test_euler_is_first_order():
    """欧拉格式误差应随 dt 线性下降，收敛比约 2。"""

    def err(dt: float, horizon: float = 1.0) -> float:
        tau, u_rest, x = 1.0, 0.0, 1.0
        lif = LIF(tau=tau, dt=dt, thr=99.0, u_rest=u_rest, method="euler")
        steps = int(round(horizon / dt))
        hist = _drive(lif, x, steps)
        exact = u_rest + x + (0.0 - u_rest - x) * math.exp(-horizon / tau)
        return abs(hist[-1][0].item() - exact)

    e1, e2 = err(0.1), err(0.05)
    assert e2 < e1
    assert 1.7 < e1 / e2 < 2.4, f"收敛比 {e1 / e2:.2f} 不像一阶"


# ---- 阈值与重置 ------------------------------------------------------------


def test_spike_condition_and_hard_reset():
    # tau=2, dt=1 → decay=0.5, gain=0.5；u_prev=0, x=4 → u_pre=2.0 >= thr=1
    lif = LIF(tau=2.0, dt=1.0, thr=1.0, reset="hard")
    s, u, _ = lif(torch.tensor([4.0], dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    assert s.item() == 1.0
    assert u.item() == pytest.approx(0.0)


def test_soft_reset_subtracts_threshold():
    lif = LIF(tau=2.0, dt=1.0, thr=1.0, reset="soft")
    s, u, _ = lif(torch.tensor([4.0], dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    assert s.item() == 1.0
    assert u.item() == pytest.approx(1.0)  # 2.0 - 1.0


def test_no_spike_below_threshold():
    lif = LIF(tau=2.0, dt=1.0, thr=1.0, reset="hard")
    s, u, _ = lif(torch.tensor([1.5], dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    assert s.item() == 0.0
    assert u.item() == pytest.approx(0.75)  # 0.5*1.5


def test_refractory_blocks_firing():
    lif = LIF(tau=2.0, dt=1.0, thr=1.0, refractory=2)
    hist = _drive(lif, 4.0, steps=8)
    fired = [i for i, (_, s) in enumerate(hist) if s.item() > 0]
    assert fired == [0, 3, 6], f"不应期未生效，实际发放时刻 {fired}"


def test_refractory_zero_is_noop():
    a = LIF(tau=2.0, dt=1.0, thr=1.0, refractory=0)
    b = LIF(tau=2.0, dt=1.0, thr=1.0)
    ha, hb = _drive(a, 4.0, steps=6), _drive(b, 4.0, steps=6)
    assert all(x[1].item() == y[1].item() for x, y in zip(ha, hb))


def test_nonspiking_readout_passes_membrane():
    """spiking=False 时应为纯泄漏积分，输出不做阈值化。"""
    lif = LIF(tau=2.0, dt=1.0, thr=0.1, spiking=False)
    out, u, _ = lif(torch.tensor([-3.0], dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    assert out.item() == pytest.approx(-1.5)  # 0.5 * -3，未被阈值截断
    assert torch.equal(out, u)


# ---- 替代梯度 --------------------------------------------------------------


@pytest.mark.parametrize("kind", SURROGATES)
def test_surrogate_integrates_to_one(kind):
    alpha = 2.0
    x = np.arange(-1000, 1000, 0.01, dtype=np.float64)
    g = surrogate_derivative(torch.from_numpy(x), alpha, kind).numpy()
    integral = np.trapezoid(g, x)
    assert integral == pytest.approx(1.0, abs=1e-3), f"{kind} 积分={integral:.5f}"


@pytest.mark.parametrize("kind", SURROGATES)
def test_surrogate_peaks_at_threshold(kind):
    """g(0) 应为全域最大值，保证梯度在阈值处最强。"""
    alpha = 2.0
    x = torch.linspace(-5, 5, 2001, dtype=torch.float64)
    g = surrogate_derivative(x, alpha, kind)
    assert g[1000] > 0
    assert g[1000].item() == g.max().item()


@pytest.mark.parametrize("kind", SURROGATES)
def test_spike_backward_matches_formula(kind):
    """反向钩子必须逐元素等于声明的替代导数。"""
    alpha = 2.0
    x = torch.linspace(-3, 3, 501, dtype=torch.float64, requires_grad=True)
    s = spike(x, alpha, kind)
    s.backward(torch.ones_like(s))
    expect = surrogate_derivative(x.detach(), alpha, kind)
    assert torch.allclose(x.grad, expect, atol=1e-12)
    assert x.grad.abs().sum() > 0


def test_spike_forward_is_binary():
    x = torch.randn(1000, dtype=torch.float64, requires_grad=True)
    s = spike(x, 2.0, "fast_sigmoid")
    assert set(s.detach().unique().tolist()) <= {0.0, 1.0}
    assert (s.detach() == (x.detach() >= 0).double()).all()


def test_gradient_through_layer_is_finite():
    lif = LIF(tau=2.0, dt=1.0, thr=1.0)
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    u = torch.zeros(4, 8, dtype=torch.float64)
    total = torch.zeros((), dtype=torch.float64)
    for _ in range(5):
        s, u, _ = lif(x, u)
        total = total + u.sum()
    total.backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


# ---- 参数校验 --------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau": 0.0},
        {"tau": 1.0, "dt": 2.0},
        {"method": "rk4"},
        {"reset": "none"},
        {"surrogate": "relu"},
    ],
)
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        LIF(**kwargs)
