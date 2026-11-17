"""训练侧组件（模型构造、循环初始化）的测试。

重点在**初始化不变量**：E/I 平衡这类改动一旦符号或尺度错了，训练照样能跑，
只是变成一个和预期不同的实验。必须用断言把它钉住。
"""

from __future__ import annotations

import math

import pytest
import torch

from train_shd import Cfg, SHDNet


def _one_layer(hidden=128, **kw) -> torch.Tensor:
    m = SHDNet(Cfg(recurrent=True, hidden=hidden, n_layers=1, **kw))
    return m.recs[0].weight.detach()


def test_default_rec_init_has_nonzero_row_sums():
    """默认初始化的行和不为零——这正是 H2 要修的东西，先把它测出来。"""
    w = _one_layer(rec_gain=1.0)
    assert w.sum(dim=1).abs().mean() > 0.1


def test_ei_balanced_rows_sum_to_zero():
    """E/I 平衡的核心：行和严格为零，循环电流没有均值分量。

    第一版把「兴奋列之和」当成要减的量，结果新行和等于原抑制和，
    均值分量原封不动（实测 |行和| 平均 3.97）。这条断言就是钉它的。
    """
    w = _one_layer(rec_init="ei_balanced", rec_gain=1.0)
    assert w.sum(dim=1).abs().max() < 1e-5


def test_ei_balanced_positive_ratio():
    """兴奋性连接应占 ei_ratio，实测约 0.8。

    用「对称高斯 × ±1」会让正负各半（实测 0.501），等于没设 E/I；
    必须是「幅度 × 符号」。
    """
    w = _one_layer(rec_init="ei_balanced", ei_ratio=0.8, rec_gain=1.0)
    frac = float((w > 0).float().mean())
    assert 0.75 < frac < 0.85, f"正比例 {frac:.3f} 偏离 0.8 太远"


def test_ei_balanced_scale_matches_default():
    """两种初始化的整体尺度必须可比，否则 rec_gain 的含义会差一个常数。

    nn.Linear 默认 U(-1/sqrt(N), 1/sqrt(N))，std = 1/sqrt(3N)。
    """
    n = 128
    w_def = _one_layer(hidden=n, rec_gain=1.0)
    w_ei = _one_layer(hidden=n, rec_init="ei_balanced", rec_gain=1.0)
    want = 1.0 / math.sqrt(3 * n)
    assert w_def.std().item() == pytest.approx(want, rel=0.15)
    assert w_ei.std().item() == pytest.approx(want, rel=0.15)


def test_unknown_rec_init_rejected():
    with pytest.raises(ValueError, match="未知 rec_init"):
        SHDNet(Cfg(recurrent=True, rec_init="nope", hidden=32))


def test_ei_balanced_forward_and_backward():
    """初始化改了之后前向形状与梯度仍要正常。"""
    m = SHDNet(Cfg(recurrent=True, rec_init="ei_balanced", hidden=32,
                   n_layers=1, tau_mode="het", frames=8))
    x = torch.randn(2, 8, 700)
    out = m(x)
    assert out.shape == (2, 20)
    out.sum().backward()
    assert m.recs[0].weight.grad is not None
    assert torch.isfinite(m.recs[0].weight.grad).all()


def test_rec_gain_scales_linearly():
    """rec_gain 应等价于整体缩放，语义在两种 init 下一致。"""
    a = _one_layer(rec_gain=1.0)
    b = _one_layer(rec_gain=4.0)
    torch.manual_seed(0)
    a2 = _one_layer(rec_gain=1.0)
    torch.manual_seed(0)
    b2 = _one_layer(rec_gain=4.0)
    assert torch.allclose(b2, a2 * 4.0, rtol=1e-5)
