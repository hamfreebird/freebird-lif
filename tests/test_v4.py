"""阶段四新增变体的不变量。

两条退化性是实现正确性的锚：

1. AT-LIF 在 `a=b=0` 时必须与 TC-LIF **逐时间步等价**（否则 M2 的 2×2 不是在测适应电流）；
2. TC-LIF 在 `β1=β2=γ=0` 时，树突就是纯累加器（否则「无衰减积分」这个前提不成立）。

另外钉住一个容易被当成 bug 的行为：`tau_ms` 对 TC-LIF **无效**。
它是接口兼容参数，静默忽略是有意的（见 twocomp.py 的模块注释）。
"""

from __future__ import annotations

import torch

from freebird_lif.variants import build


def _run(ne, x, n_steps=None):
    """跑一段输入，返回 (脉冲, 末状态)。"""
    b, T, c = x.shape
    st = ne.init_state(b, c)
    outs = []
    for t in range(T if n_steps is None else n_steps):
        s, st = ne.step(x[:, t], st)
        outs.append(s)
    return torch.stack(outs, dim=1), st


def test_tclif_linear_spectrum_is_neutral():
    """无衰减因子的直接后果：det=1，特征值模为 1，是中性稳定的旋转。"""
    ne = build("tclif", 8, beta1=0.1, beta2=0.2, gamma=0.5)
    sp = ne.linear_spectrum()
    assert abs(sp["det"] - 1.0) < 1e-12, f"det 应为 1，实际 {sp['det']}"
    assert abs(sp["modulus"] - 1.0) < 1e-9, "特征值模应为 1（无阻尼）"
    assert 1.0 < sp["trace"] < 2.0, f"trace 应在 (1,2)，实际 {sp['trace']}"
    assert 10.0 < sp["period_frames"] < 200.0, "周期应落在 SHD 的 100 帧窗口附近"
    # 换个 β 组合仍应是中性稳定
    ne2 = build("tclif", 8, beta1=0.3, beta2=0.6)
    sp2 = ne2.linear_spectrum()
    assert abs(sp2["det"] - 1.0) < 1e-12
    assert abs(sp2["modulus"] - 1.0) < 1e-9


def test_atlif_reduces_to_tclif_when_adaptation_off():
    """a=b=0 时 AT-LIF 与 TC-LIF 必须逐时间步一致。"""
    torch.manual_seed(0)
    x = (torch.rand(3, 40, 16) < 0.1).float() * 2.0
    tc = build("tclif", 16, beta1=0.1, beta2=0.2, gamma=0.5)
    at = build("atlif", 16, beta1=0.1, beta2=0.2, gamma=0.5, a0=0.0, b0=0.0)
    at.load_state_dict(tc.state_dict(), strict=False)   # 共享 c1/c2/gamma
    s1, st1 = _run(tc, x)
    s2, st2 = _run(at, x)
    assert torch.equal(s1, s2), "脉冲序列应逐元素一致"
    assert torch.allclose(st1[1], st2[1]), "胞体膜电位应一致"


def test_tclif_dendrite_is_pure_accumulator():
    """β1=β2=γ=0 时，U^D[t] = Σ_{k<=t} x[k]（无衰减、无复位）。"""
    ne = build("tclif", 4, beta1=1e-6, beta2=1e-6, gamma=0.0)
    x = torch.full((1, 25, 4), 0.3)
    _, st = _run(ne, x)
    expect = 0.3 * 25
    assert torch.allclose(st[0], torch.full((1, 4), expect), atol=1e-4), \
        f"树突应累加到 {expect}，实际 {st[0].mean().item():.4f}"


def test_tclif_input_scale_is_unity():
    """输入以系数 1 进入树突，不需要 1/(1-α) 缩放。"""
    ne = build("tclif", 6)
    sc = ne.input_scale()
    assert sc.shape == (6,)
    assert torch.allclose(sc, torch.ones(6))


def test_tclif_ignores_tau_ms_by_design():
    """tau_ms 对 TC-LIF 无效——这是有意为之，不是漏传。

    如果哪天有人"修好"了它，这条测试会失败，提醒他先改 PLAN-v4 §3.3 里
    关于「异构 τ 对两隔室变体不适用」的说明。
    """
    torch.manual_seed(0)
    x = (torch.rand(2, 20, 8) < 0.2).float()
    a = build("tclif", 8, tau_ms=2.0)
    b = build("tclif", 8, tau_ms=400.0)
    b.load_state_dict(a.state_dict())
    sa, _ = _run(a, x)
    sb, _ = _run(b, x)
    assert torch.equal(sa, sb), "tau_ms 不应影响 TC-LIF 的动力学"


def test_atlif_learn_flags_are_honored():
    """learn_a=False 时 a 不可学，learn_b 仍可学。"""
    ne = build("atlif", 5, learn_a=False, learn_b=True)
    assert ne.a.requires_grad is False
    assert ne.b.requires_grad is True


def test_state_counts_are_declared_correctly():
    """n_state / float_state / soma_index 与实际状态元组一致。"""
    tc = build("tclif", 3)
    at = build("atlif", 3)
    st_tc = tc.init_state(1, 3)
    st_at = at.init_state(1, 3)
    assert len(st_tc) == tc.n_state == 3
    assert len(st_at) == at.n_state == 4
    assert tc.float_state == 2 and at.float_state == 3
    # 胞体下标必须指向一个浮点状态，而不是二值的 s_prev
    assert tc.soma_index == 1 and at.soma_index == 1
    assert torch.allclose(st_tc[tc.soma_index], torch.zeros(1, 3))
