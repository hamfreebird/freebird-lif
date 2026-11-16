"""SHD（Spiking Heidelberg Digits）数据管线。

原始格式：HDF5，每组样本是稀疏的 (time[s], unit) 事件列表，unit ∈ [0,700)。
本模块把它分箱成稠密张量 [T, 700]。

三个设计判断：

1. **缓存用 uint8 而非 float32。**
   T=100 时训练集稠密为 8156×100×700，float32 是 2.13 GB，uint8 是 0.53 GB。
   每批再转 float32（batch=128 时 35.8 MB），代价可忽略，省下 1.6 GB 常驻内存。

2. **分箱用 bincount 而非 np.add.at。**
   add.at 是逐元素慢路径，bincount 是 O(N) 的向量化路径，快一个数量级。

3. **时间轴用全局统一时长，不做逐样本归一化。**
   SHD 的样本时长本身携带信息（说话快慢），逐样本拉伸会破坏这个特征。
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
N_CHANNELS = 700
N_CLASSES = 20

URLS = {
    "train": "https://zenkelab.org/datasets/shd/shd_train.h5.gz",
    "test": "https://zenkelab.org/datasets/shd/shd_test.h5.gz",
}

# 归档字节数。SHD 下载偶发截断（实测 test 集断在 36.7/38.1 MB），
# gzip 解压时才报 EOFError，必须在写盘后立刻校验长度。
EXPECTED_BYTES = {
    "train": 130_839_480,
    "test": 38_137_408,
}


def _fetch_range(url: str, start: int, end: int, dst: Path, retries: int = 5) -> int:
    """下载 [start, end] 闭区间到一个分片文件，返回字节数。

    **用 curl 子进程，不用 urllib。** 本机经代理访问时，urllib 的 Range 请求
    会挂在 0 字节不再推进（8 个并发全部如此，等了 20 分钟无进展）；同一网络下
    `curl -r` 实测 78 KB/s。这类差异不值得深究，能验证的是哪个work。
    """
    import subprocess

    want = end - start + 1
    if dst.exists() and dst.stat().st_size == want:
        return want                      # 分片级续传：重跑时不重下已完成的分片
    for k in range(retries):
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "900", "-r", f"{start}-{end}", "-o", str(dst), url],
            capture_output=True,
        )
        if r.returncode == 0 and dst.exists() and dst.stat().st_size == want:
            return want
        time.sleep(2.0 * (k + 1))
    raise RuntimeError(f"分片 {start}-{end} 下载失败（重试 {retries} 次）")


def download(dest_dir: Path, force: bool = False, workers: int = 8) -> dict[str, Path]:
    """下载并解压 SHD。

    **必须并行分段。** 本机到 zenkelab 的单流实测只有 12–18 KB/s，130 MB 要 2–3 小时；
    而该站支持 Range（`Accept-Ranges: bytes`），8 路分段实测 78 KB/s，快 4–6 倍。
    这在阶段三 M0 实测得到，阶段二用的单流实现因此被替换。

    续传：若目标文件已存在但是不完整，把它当作前缀复用，只并行下载剩余部分。
    """
    from concurrent.futures import ThreadPoolExecutor

    dest_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, url in URLS.items():
        gz = dest_dir / f"shd_{split}.h5.gz"
        h5 = dest_dir / f"shd_{split}.h5"
        if h5.exists() and not force:
            out[split] = h5
            continue
        expect = EXPECTED_BYTES[split]
        have = gz.stat().st_size if gz.exists() else 0
        if have > expect:                       # 明显损坏，重来
            gz.unlink()
            have = 0
        if have == expect:
            print(f"  {gz.name} 已完整（{have} 字节），跳过下载")
        else:
            if have:
                print(f"  续传 {gz.name}：已有 {have / 1e6:.1f} MB / {expect / 1e6:.1f} MB")
            else:
                print(f"  分段下载 {url}（{workers} 路）")
            rest = expect - have
            # 分片数量按剩余大小定，单片约 4 MB，上限 workers*3 以摊薄尾部等待
            nchunk = max(1, min(workers * 3, rest // (4 << 20) + 1))
            size = (rest + nchunk - 1) // nchunk
            spans = [(have + i * size, min(have + (i + 1) * size - 1, expect - 1))
                     for i in range(nchunk)]
            parts = [dest_dir / f".{gz.name}.part{i}" for i in range(nchunk)]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda a: _fetch_range(url, a[0][0], a[0][1], a[1]),
                            zip(spans, parts)))
            with open(gz, "ab") as f:            # 追加到已有前缀之后
                for part in parts:
                    with open(part, "rb") as pf:
                        shutil.copyfileobj(pf, f, length=1 << 22)
                    part.unlink()
        got = gz.stat().st_size
        if got != expect:
            raise RuntimeError(
                f"{gz.name} 下载不完整：{got} 字节，应为 {expect}。")
        print(f"  解压 {gz.name}")
        with gzip.open(gz, "rb") as src, open(h5, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
        out[split] = h5
    return out


def read_h5(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """返回 (times 秒, units 通道号, labels, speaker)。

    实测结构（与 zenkelab 文档的描述略有出入，以文件为准）：
        /labels            (N,) uint16
        /spikes/times      (N,) object，每元素是变长 float 数组
        /spikes/units      (N,) object，每元素是对应通道号
        /extra/speaker     (N,) uint16，12 位说话人，其中 2 位只在测试集出现
    """
    import h5py

    with h5py.File(path, "r") as f:
        labels = np.asarray(f["labels"][:]).astype(np.int64)
        speaker = (np.asarray(f["extra"]["speaker"][:]).astype(np.int64)
                   if "extra" in f and "speaker" in f["extra"] else np.zeros(len(labels), np.int64))
        g = f["spikes"]
        times = [np.asarray(t, dtype=np.float32) for t in g["times"][:]]
        units = [np.asarray(u, dtype=np.int32) for u in g["units"][:]]
    if not (len(times) == len(units) == len(labels) == len(speaker)):
        raise RuntimeError(f"{path.name}: times/units/labels/speaker 长度不一致")
    return times, units, labels, speaker


def bin_one(times: np.ndarray, units: np.ndarray, T: int, duration: float,
            n_channels: int = N_CHANNELS, binary: bool = True) -> np.ndarray:
    """单样本 → [T, n_channels] uint8。"""
    out = np.zeros((T, n_channels), dtype=np.uint8)
    if times.size == 0:
        return out
    # 时间归一化到 [0, T)，超出窗口的事件丢弃（clamp 会污染最后一帧，改为过滤）
    keep = (times >= 0) & (times < duration)
    if not np.any(keep):
        return out
    t = times[keep]
    u = units[keep]
    valid = (u >= 0) & (u < n_channels)
    t, u = t[valid], u[valid]
    frame = np.minimum((t / duration * T).astype(np.int32), T - 1)
    flat = frame.astype(np.int64) * n_channels + u.astype(np.int64)
    counts = np.bincount(flat, minlength=T * n_channels).reshape(T, n_channels)
    if binary:
        out = (counts > 0).astype(np.uint8)
    else:
        out = np.minimum(counts, 255).astype(np.uint8)
    return out


def build(root: Path | None = None, T: int = 100, binary: bool = True,
          val_size: int = 1000, val_seed: int = 2024, force: bool = False) -> dict:
    """构建（或从缓存加载）分箱后的数据集。

    返回 dict：
        x_train [N,T,700] uint8, y_train [N]
        x_val, y_val      —— 从训练集切出，官方测试集全程隔离
        x_test, y_test
        meta {T, duration, binary, val_size, val_seed}
    """
    root = root or (ROOT / "data" / "shd")
    cache = root / f"binned_T{T}_{'bin' if binary else 'cnt'}.npz"
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=False)
        out = {
            "x_train": z["x_train"], "y_train": z["y_train"],
            "x_val": z["x_val"], "y_val": z["y_val"],
            "x_test": z["x_test"], "y_test": z["y_test"],
            "meta": {"T": T, "duration": float(z["duration"]), "binary": binary,
                     "val_size": val_size, "val_seed": val_seed},
        }
        # 说话人标签用于按说话人留出验证（SHD 测试集 81% 来自未见说话人）
        for k in ("speaker_train", "speaker_val", "speaker_test"):
            if k in z.files:
                out[k] = z[k]
        return out

    paths = download(root, force=force)
    raw = {s: read_h5(p) for s, p in paths.items()}

    duration = float(max(
        max((t.max() for t in raw[s][0] if t.size), default=0.0) for s in raw))
    n_spk_tr = len(np.unique(raw["train"][3]))
    n_spk_te = len(np.unique(raw["test"][3]))
    print(f"  说话人：训练 {n_spk_tr} 位，测试 {n_spk_te} 位（其中部分不在训练集）")
    duration = float(np.ceil(duration * 100) / 100)  # 向上取到 10ms，避免边界抖动
    print(f"  统一下时长 = {duration:.2f} s，T = {T}，每帧 {duration / T * 1000:.1f} ms")

    def conv(split: str):
        times, units, labels, speaker = raw[split]
        n = len(labels)
        x = np.empty((n, T, N_CHANNELS), dtype=np.uint8)
        for i in range(n):
            x[i] = bin_one(times[i], units[i], T, duration, binary=binary)
            if (i + 1) % 2000 == 0:
                print(f"    {split} {i + 1}/{n}")
        return x, labels, speaker

    x_tr_all, y_tr_all, spk_tr_all = conv("train")
    x_te, y_te, spk_te = conv("test")

    g = np.random.default_rng(val_seed)
    perm = g.permutation(len(x_tr_all))
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    out = {
        "x_train": x_tr_all[tr_idx], "y_train": y_tr_all[tr_idx],
        "x_val": x_tr_all[val_idx], "y_val": y_tr_all[val_idx],
        "x_test": x_te, "y_test": y_te,
        "speaker_train": spk_tr_all[tr_idx], "speaker_val": spk_tr_all[val_idx],
        "speaker_test": spk_te,
        "meta": {"T": T, "duration": duration, "binary": binary,
                 "val_size": val_size, "val_seed": val_seed},
    }
    np.savez_compressed(cache, duration=duration, **{k: v for k, v in out.items() if k != "meta"})
    print(f"  缓存写入 {cache.name}")
    return out


def describe(d: dict) -> str:
    meta = d["meta"]
    lines = [
        f"SHD  T={meta['T']}  duration={meta['duration']:.2f}s  "
        f"binary={meta['binary']}",
        f"  train {d['x_train'].shape}  val {d['x_val'].shape}  test {d['x_test'].shape}",
    ]
    for s in ("train", "val", "test"):
        x = d[f"x_{s}"]
        rate = float(x.mean())
        per_sample = x.sum(axis=(1, 2))
        lines.append(
            f"  {s:5s} 脉冲密度 {rate:.4f}  每样本脉冲数 "
            f"min={per_sample.min():.0f} med={np.median(per_sample):.0f} "
            f"max={per_sample.max():.0f}")
    counts = np.bincount(d["y_train"], minlength=N_CLASSES)
    lines.append(f"  训练集类别分布 {counts.tolist()}")
    return "\n".join(lines)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
