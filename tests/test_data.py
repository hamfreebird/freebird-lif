"""数据缓存校验：M0 完成判据的核心断言。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TRAIN = DATA / "mnist16_train.npy"
TEST = DATA / "mnist16_test.npy"
LABELS_TRAIN = DATA / "mnist_labels_train.npy"
LABELS_TEST = DATA / "mnist_labels_test.npy"

pytestmark = pytest.mark.skipif(
    not TRAIN.exists(), reason="数据缓存缺失，先运行 python scripts/fetch_data.py"
)

CONTENT = 12  # 与 fetch_data.py 默认值一致


@pytest.fixture(scope="module")
def cache():
    return (
        np.load(TRAIN),
        np.load(LABELS_TRAIN),
        np.load(TEST),
        np.load(LABELS_TEST),
    )


def test_shapes_and_dtype(cache):
    x_tr, y_tr, x_te, y_te = cache
    assert x_tr.shape == (60000, 256)
    assert x_te.shape == (10000, 256)
    assert x_tr.dtype == np.uint8
    assert x_te.dtype == np.uint8
    assert y_tr.shape == (60000,)
    assert y_te.shape == (10000,)


def test_label_range_and_balance(cache):
    _, y_tr, _, y_te = cache
    assert y_tr.min() == 0 and y_tr.max() == 9
    assert y_te.min() == 0 and y_te.max() == 9
    counts = np.bincount(y_tr, minlength=10)
    assert counts.min() > 5000, f"训练集类别失衡: {counts.tolist()}"
    assert np.bincount(y_te, minlength=10).min() > 800


def test_value_range(cache):
    x_tr, _, x_te, _ = cache
    assert x_tr.max() <= 255
    # 背景必须为纯零，否则电流注入编码的语义被破坏
    assert (x_tr == 0).mean() > 0.5, "零像素占比过低，背景可能被填充污染"
    assert 0 < x_tr.mean() < 255


def test_margin_is_zero(cache):
    """content=12 居中于 16×16，四边各留 2 像素，必然全零。

    该不变量同时校验了 bbox 裁剪、缩放上限与居中逻辑三者一致。
    """
    x_tr, _, _, _ = cache
    margin = 16 - CONTENT
    half = margin // 2
    g = x_tr.reshape(-1, 16, 16)
    assert g[:, :half, :].max() == 0, "上边距非零"
    assert g[:, 16 - half :, :].max() == 0, "下边距非零"
    assert g[:, :, :half].max() == 0, "左边距非零"
    assert g[:, :, 16 - half :].max() == 0, "右边距非零"


def test_no_empty_sample(cache):
    """bbox 裁剪后每个样本都应含有效笔画（MNIST 无全零图）。"""
    x_tr, _, x_te, _ = cache
    assert (x_tr.sum(axis=1) == 0).sum() == 0
    assert (x_te.sum(axis=1) == 0).sum() == 0


def test_manifest_matches(cache):
    manifest = json.loads((DATA / "manifest.json").read_text("utf-8"))
    x_tr, _, x_te, _ = cache
    assert manifest["size"] == 16
    assert manifest["content"] == CONTENT
    assert manifest["train_shape"] == list(x_tr.shape)
    assert manifest["test_shape"] == list(x_te.shape)
    assert len(manifest["source_sha256"]) == 4
