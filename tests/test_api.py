"""公开 API 与校准模块。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# v1 的推理引擎依赖 numba；XPU 环境（v2）未安装，整模块跳过。
# 这不是掩盖失败：v1 引擎测试应在 v1 环境运行，见 docs/PLAN-v1-lif-mnist.md。
pytest.importorskip("numba")

from freebird_lif.api import Prediction, Predictor, as_batch
from freebird_lif.calibrate import ece, fit_temperature, nll, rejection_curve, softmax, suggest_tau


ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "models" / "freebird-lif-T4.npz"
CALIB = ROOT / "models" / "calibration.json"
DATA = ROOT / "data" / "mnist16_test.npy"

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="权重包缺失")


# ---- 输入规范化 ------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(16, 16), (256,), (1, 16, 16), (4, 16, 16), (4, 1, 16, 16), (4, 256)],
)
def test_as_batch_shapes(shape):
    x = np.random.rand(*shape)
    out = as_batch(x)
    assert out.shape[1:] == (1, 16, 16)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_as_batch_uint8_scaled():
    x = np.full((16, 16), 255, np.uint8)
    out = as_batch(x)
    assert out.max() == pytest.approx(1.0)


def test_as_batch_float_above_one_scaled():
    x = np.full((16, 16), 255.0, np.float32)
    assert as_batch(x).max() == pytest.approx(1.0)


def test_as_batch_float_unit_untouched():
    x = np.full((16, 16), 0.5, np.float32)
    assert as_batch(x).max() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10)),                 # 空间形状错误
        np.zeros((2, 2, 16, 16)),           # 通道维不为 1
        np.zeros((3, 128)),                 # 一维长度错误
        np.full((16, 16), np.nan),          # 含 NaN
        np.full((16, 16), -5.0),            # 越界
    ],
)
def test_as_batch_rejects(bad):
    with pytest.raises(ValueError):
        as_batch(bad)


# ---- 预测 ----------------------------------------------------------------


@pytest.fixture(scope="module")
def predictor():
    return Predictor(BUNDLE, CALIB if CALIB.exists() else None)


def test_predict_one_returns_prediction(predictor):
    r = predictor.predict_one(np.zeros((16, 16), np.float32))
    assert isinstance(r, Prediction)
    assert 0 <= r.digit <= 9
    assert 0.0 < r.confidence <= 1.0
    assert isinstance(r.accepted, bool)


def test_predict_batch_shapes(predictor):
    d, c = predictor.predict(np.random.rand(5, 16, 16))
    assert d.shape == c.shape == (5,)
    assert d.dtype.kind in "iu"


def test_probabilities_sum_to_one(predictor):
    p = predictor.probabilities(np.random.rand(7, 16, 16))
    assert p.shape == (7, 10)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)


def test_single_and_batch_consistent(predictor):
    x = np.random.rand(4, 16, 16)
    d_batch, c_batch = predictor.predict(x)
    for i in range(4):
        r = predictor.predict_one(x[i])
        assert r.digit == d_batch[i]
        assert r.confidence == pytest.approx(c_batch[i], abs=1e-6)


def test_explain_topk(predictor):
    out = predictor.explain(np.random.rand(3, 16, 16), top_k=3)
    assert len(out) == 3
    assert len(out[0]["top"]) == 3
    probs = [p for _, p in out[0]["top"]]
    assert probs == sorted(probs, reverse=True)


def test_predictor_applies_calibration(predictor):
    if not CALIB.exists():
        pytest.skip("未校准")
    T = json.loads(CALIB.read_text("utf-8"))["temperature"]
    assert predictor.temperature == pytest.approx(T)


def test_threshold_marks_rejection():
    p = Predictor(BUNDLE, None, threshold=0.999999)
    r = p.predict_one(np.zeros((16, 16), np.float32))
    assert r.accepted is False


# ---- 校准 ----------------------------------------------------------------


def _overconfident(n=2000, scale=4.0, acc=0.8, seed=0):
    """构造"高置信度但准确率只有 acc"的 logits，用于检验温度缩放往上回调。"""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n)
    correct = rng.random(n) < acc
    wrong = (y + rng.integers(1, 10, n)) % 10
    target = np.where(correct, y, wrong)
    logits = rng.normal(0, 0.2, (n, 10)).astype(np.float32)
    logits[np.arange(n), target] += scale
    return logits, y


def test_fit_temperature_reduces_nll():
    logits, y = _overconfident()
    t = fit_temperature(logits, y)
    assert 0.1 < t < 10.0
    assert nll(logits, y, t) < nll(logits, y, 1.0)


def test_fit_temperature_direction_for_overconfidence():
    """过度自信（logits 被放大）时应学出 T > 1。"""
    logits, y = _overconfident(scale=4.0)
    assert fit_temperature(logits, y) > 1.0


def test_fit_temperature_is_deterministic():
    logits, y = _overconfident(n=500)
    assert fit_temperature(logits, y) == fit_temperature(logits, y)


def test_ece_bounds_and_improvement():
    logits, y = _overconfident(n=3000, scale=5.0)
    t = fit_temperature(logits, y)
    before = ece(softmax(logits), y)
    after = ece(softmax(logits / t), y)
    assert 0.0 <= after["ece"] <= 1.0
    assert after["ece"] < before["ece"]
    assert sum(b["n"] for b in after["bins"]) == len(y)


def test_rejection_curve_monotone():
    logits, y = _overconfident(n=1000, scale=2.0)
    curve = rejection_curve(softmax(logits), y, n_points=11)
    cov = [p["coverage"] for p in curve]
    assert all(a >= b - 1e-9 for a, b in zip(cov, cov[1:])), "覆盖率应随 τ 单调下降"


def test_suggest_tau_meets_target():
    logits, y = _overconfident(n=2000, scale=2.0)
    p = softmax(logits)
    curve = rejection_curve(p, y, n_points=41)
    tau = suggest_tau(curve, 0.9)
    assert tau is not None
    sel = [q for q in curve if q["tau"] >= tau and q["accuracy"] is not None]
    assert sel and max(q["accuracy"] for q in sel) >= 0.9
