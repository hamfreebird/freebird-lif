"""多基准数据管线的测试。

重点不是"能跑"，而是几个容易静默出错的不变量：
  * 分箱不能凭空造出事件（占用格数 ≤ 事件数）
  * 越界时间必须被裁到最后一帧，而不是丢弃或溢出到下一行
  * N-MNIST 的时间戳必须单调（字节布局错了会立刻违反这一条）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from freebird_lif.bench.common import bin_events, density

ROOT = Path(__file__).resolve().parents[1]
NMNIST_DIR = ROOT / "data" / "bench" / "nmnist"
EVTOUCH_DIR = ROOT / "data" / "bench" / "evtouch" / "Ev-Objects"


def test_bin_events_shape_and_binary():
    t = np.array([0.0, 0.01, 0.05], dtype=np.float32)
    idx = np.array([0, 1, 0], dtype=np.int32)
    out = bin_events(t, idx, T=10, C=4, duration=0.1)
    assert out.shape == (10, 4)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_bin_events_never_creates_events():
    """占用格数不能超过事件数——分箱只能合并，不能增殖。"""
    rng = np.random.default_rng(0)
    n = 500
    t = rng.uniform(0, 0.3, n).astype(np.float32)
    idx = rng.integers(0, 20, n).astype(np.int32)
    out = bin_events(t, idx, T=50, C=20, duration=0.3)
    assert out.sum() <= n


def test_bin_events_clips_out_of_range():
    """超出 duration 的时间戳必须落进最后一帧，而不是被丢掉或越界。

    注意：三个事件若落进同一格，二值化后和为 1——不能拿 sum 当"事件数"。
    所以这里分别验证"合法事件在自己帧"与"越界事件在末帧"两个位置。
    """
    # 0.01 -> 帧 0；5.0 远超 duration -> 帧 9
    out = bin_events(np.array([0.01, 5.0], np.float32), np.array([0, 0], np.int32),
                     T=10, C=2, duration=0.3)
    assert out.shape == (10, 2)
    assert out[0, 0] == 1, "合法事件应在第 0 帧"
    assert out[9, 0] == 1, "越界事件应落入最后一帧"
    assert out.sum() == 2


def test_bin_events_single_event_position():
    """时间 0 的事件必须落在第 0 帧；这是最容易被 off-by-one 搞错的地方。"""
    out = bin_events(np.array([0.0], np.float32), np.array([3], np.int32),
                     T=8, C=4, duration=0.4)
    assert out[0, 3] == 1
    assert out.sum() == 1


def test_density_range():
    x = np.zeros((4, 10, 5), dtype=np.uint8)
    x[:, :5, :] = 1
    assert density(x, sample=4) == pytest.approx(0.5)


# ---- N-MNIST -------------------------------------------------------------

def _any_nmnist_file():
    if not NMNIST_DIR.exists():
        pytest.skip("N-MNIST 未下载")
    fs = sorted(NMNIST_DIR.rglob("*.bin"))
    if not fs:
        pytest.skip("N-MNIST 未下载")
    return fs[0]


def test_nmnist_event_stream_is_wellformed():
    from freebird_lif.bench import nmnist

    t, idx, p = nmnist.read_bin(_any_nmnist_file())
    assert len(t) > 100
    assert t.min() >= 0
    assert t.max() < 0.5, "单样本时长应约 300 ms"
    assert idx.min() >= 0 and idx.max() < 34 * 34
    assert set(np.unique(p).tolist()) <= {0, 1}
    # 极性应当两派都有；全 0 或全 1 说明极性位解析错了
    assert 0.2 < p.mean() < 0.8


def test_nmnist_timestamps_are_monotonic():
    """官方格式里时间戳跨 byte2 的低 7 位。漏掉这 7 位会让序列乱序。

    这条断言的作用就是钉住那次错误：第一版按"x,y,p + 4 字节时间戳"解析，
    ts 立刻变成非单调。
    """
    from freebird_lif.bench import nmnist

    t, _, _ = nmnist.read_bin(_any_nmnist_file())
    assert np.all(np.diff(t) >= -1e-9)


def test_nmnist_rejects_bad_length(tmp_path):
    from freebird_lif.bench import nmnist

    f = tmp_path / "bad.bin"
    f.write_bytes(b"\x00" * 7)          # 不是 5 的倍数
    with pytest.raises(RuntimeError, match="5 的倍数"):
        nmnist.read_bin(f)


# ---- EvTouch -------------------------------------------------------------

def test_evtouch_shapes_and_split():
    if not EVTOUCH_DIR.exists():
        pytest.skip("EvTouch 未解压")
    from freebird_lif.bench import evtouch

    d = evtouch.build(T=250)
    for k in ("x_train", "x_val", "x_test"):
        assert d[k].dtype == np.uint8
        assert d[k].ndim == 3 and d[k].shape[2] == 39
    assert d["meta"]["n_classes"] == 36
    # 三者不能重叠：验证集是从训练集切的，必须真的切走
    assert len(d["y_train"]) + len(d["y_val"]) == 576
    assert len(d["y_test"]) == 144


def test_evtouch_pooling_is_or_not_sum():
    """降采样用 OR：结果必须仍是二值。若误用求和，会出现 >1 的值。"""
    if not EVTOUCH_DIR.exists():
        pytest.skip("EvTouch 未解压")
    from freebird_lif.bench import evtouch

    d = evtouch.build(T=50)
    assert d["x_train"].max() == 1
