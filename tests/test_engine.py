"""推理引擎数值一致性。M3 的核心判据：三条路径必须与 PyTorch 等价。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

# v1 的推理引擎依赖 numba；XPU 环境（v2）未安装，整模块跳过。
# 这不是掩盖失败：v1 引擎测试应在 v1 环境运行，见 docs/PLAN-v1-lif-mnist.md。
pytest.importorskip("numba")
import torch

from freebird_lif.engine.export import (
    channel_last_perm,
    export_weights,
    load_bundle,
    load_checkpoint,
    save_bundle,
)
from freebird_lif.engine.numba_engine import NumbaEngine, conv_front_batch
from freebird_lif.engine.numpy_ref import NumpyEngine


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "freebird-lif-T4.pt"
DATA = ROOT / "data" / "mnist16_test.npy"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="模型文件缺失")


@pytest.fixture(scope="module")
def bundle():
    model, _ = load_checkpoint(MODEL)
    cfg = dataclasses.asdict(model.cfg)
    return model, cfg, export_weights(model)


@pytest.fixture(scope="module")
def xb():
    if not DATA.exists():
        pytest.skip("数据缓存缺失")
    x = np.load(DATA)[:256].reshape(-1, 1, 16, 16).astype(np.float32) / 255.0
    return x


def test_export_shapes(bundle):
    model, cfg, w = bundle
    assert w["conv1_w"].shape == (12, 9)
    assert w["conv2_w"].shape == (12, 9, 24)
    assert w["fc1_wt"].shape == (384, 96)
    assert w["fc2_w"].shape == (10, 96)
    assert w["decay"].shape == (1,)
    assert w["conv1_w"].dtype == np.float32


def test_channel_last_perm_is_bijection():
    perm = channel_last_perm(4, 24)
    assert len(perm) == 384
    assert sorted(perm) == list(range(384))


def test_bn_folding_matches_torch(bundle, xb):
    """折叠后引擎输出必须复现 PyTorch 的 logits（误差在 float32 累加序差异内）。"""
    model, cfg, w = bundle
    with torch.no_grad():
        ref, _ = model(torch.from_numpy(xb))
    got = NumpyEngine(w, cfg).forward(xb)
    assert np.abs(ref.numpy() - got).max() < 1e-4


def test_l0_l1_agree(bundle, xb):
    model, cfg, w = bundle
    a = NumpyEngine(w, cfg).forward(xb).argmax(1)
    b = NumbaEngine(w, cfg, sparse=True).forward(xb).argmax(1)
    assert np.array_equal(a, b)


def test_sparse_and_dense_agree(bundle, xb):
    model, cfg, w = bundle
    a = NumbaEngine(w, cfg, sparse=True).forward(xb)
    b = NumbaEngine(w, cfg, sparse=False).forward(xb)
    assert np.abs(a - b).max() < 1e-5
    assert np.array_equal(a.argmax(1), b.argmax(1))


def test_conv1_kernels_agree(bundle, xb):
    """融合内核与 im2col+GEMM 必须给出相同的第一层电流。"""
    model, cfg, w = bundle
    eng = NumbaEngine(w, cfg)
    fused = np.empty((len(xb), 768), np.float32)
    from freebird_lif.engine.numba_engine import _conv1_batch

    _conv1_batch(np.ascontiguousarray(xb), eng.w1t, eng.b1f, fused)
    gemm = conv_front_batch(xb, w["conv1_w"], w["conv1_b"])
    assert np.abs(fused - gemm).max() < 1e-4


def test_single_sample_matches_batch(bundle, xb):
    """B=1 的单样本推理必须与批量结果逐元素一致（无批内状态泄漏）。"""
    model, cfg, w = bundle
    eng = NumbaEngine(w, cfg, sparse=True)
    full = eng.forward(xb[:8])
    for i in range(4):
        one = eng.forward(xb[i : i + 1])
        assert np.allclose(one[0], full[i], atol=1e-6)


def test_bundle_roundtrip(bundle, tmp_path):
    model, cfg, w = bundle
    p = save_bundle(model, tmp_path / "m.npz")
    w2, cfg2 = load_bundle(p)
    assert cfg2["T"] == cfg["T"]
    for k in w:
        assert np.array_equal(w[k], w2[k]), f"{k} 未正确往返"


def test_engine_accuracy(bundle):
    """引擎与 PyTorch 必须在**同一条样本**上给出相同预测。

    刻意不断言绝对精度阈值：测试集前 2000 条本身偏难（98.15%），
    全量 10000 条为 98.97%。用绝对阈值会把数据分布差异误报成引擎缺陷。
    真正的不变量是"两条路径逐样本一致"。
    """
    model, cfg, w = bundle
    if not DATA.exists():
        pytest.skip("数据缓存缺失")
    n = 2000
    x = np.load(DATA)[:n].reshape(-1, 1, 16, 16).astype(np.float32) / 255.0
    y = np.load(ROOT / "data" / "mnist_labels_test.npy")[:n].astype(np.int64)
    with torch.no_grad():
        ref = model(torch.from_numpy(x))[0].argmax(1).numpy()
    got = NumbaEngine(w, cfg).forward(x).argmax(1)
    assert np.array_equal(ref, got), "引擎与 PyTorch 预测不一致"
    acc = (got == y).mean()
    assert acc > 0.97, f"引擎精度 {acc:.4f} 异常"
