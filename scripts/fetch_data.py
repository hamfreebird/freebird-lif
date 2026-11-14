"""MNIST 下载与 16×16 预处理缓存生成。

不依赖 torchvision：自实现 IDX 解析，仅需 numpy + Pillow。

预处理顺序（与 PLAN.md §4 一致，不可调整）：
    包围盒裁剪 → 等比例缩放至 content 内容区 → 居中补零至 size×size

用法:
    python scripts/fetch_data.py                 # 下载并生成缓存
    python scripts/fetch_data.py --force         # 强制重新生成
    python scripts/fetch_data.py --preview       # 生成后打印 ASCII 预览
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://raw.githubusercontent.com/fgnt/mnist/master",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def read_idx(path: Path) -> np.ndarray:
    """解析 IDX 格式。magic 首字节为 0x00，后续三字节为维度编码。"""
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic = int.from_bytes(buf[0:4], "big")
    ndim = magic & 0xFF
    if ndim not in (1, 3):
        raise ValueError(f"{path.name}: 非预期 IDX 维度 {ndim}（magic={magic:#x}）")
    if ndim == 1:
        n = int.from_bytes(buf[4:8], "big")
        return np.frombuffer(buf, dtype=np.uint8, offset=8, count=n).copy()
    n = int.from_bytes(buf[4:8], "big")
    rows = int.from_bytes(buf[8:12], "big")
    cols = int.from_bytes(buf[12:16], "big")
    return np.frombuffer(buf, dtype=np.uint8, offset=16, count=n * rows * cols).reshape(
        n, rows, cols
    ).copy()


def download(name: str, dest_dir: Path, force: bool = False) -> Path:
    dest = dest_dir / name
    if dest.exists() and not force:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for base in MIRRORS:
        url = f"{base}/{name}"
        try:
            print(f"  下载 {url}")
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            if len(data) < 1024:
                raise ValueError(f"响应过小（{len(data)} 字节），疑似错误页")
            dest.write_bytes(data)
            print(f"  完成 {name}  ({len(data) / 1024:.0f} KB)")
            return dest
        except Exception as exc:  # 逐个镜像回退
            print(f"  失败（{type(exc).__name__}: {exc}），尝试下一个镜像")
            last_err = exc
    raise RuntimeError(f"所有镜像均失败: {name}") from last_err


def to_16x16(img: np.ndarray, size: int, content: int, bbox: bool) -> np.ndarray:
    """单张 28×28 uint8 → size×size uint8。

    bbox=True 时先裁掉空白边距，使数字内容占 content 像素（保留与原
    MNIST 20/28≈71% 相近的边距比例），再用 BOX（面积平均）降采样，
    避免双三次插值在细笔画上的振铃伪影。
    """
    if bbox:
        ys, xs = np.nonzero(img)
        if ys.size:
            img = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    h, w = img.shape
    scale = content / max(h, w)
    # 至少保留 1 像素，且不得超过 content（避免 round 误差越界）
    new_h = min(content, max(1, int(round(h * scale))))
    new_w = min(content, max(1, int(round(w * scale))))

    resized = np.asarray(
        Image.fromarray(img, mode="L").resize((new_w, new_h), Image.Resampling.BOX),
        dtype=np.uint8,
    )

    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def build_split(images: np.ndarray, size: int, content: int, bbox: bool, tag: str) -> np.ndarray:
    n = images.shape[0]
    out = np.empty((n, size * size), dtype=np.uint8)
    for i in range(n):
        out[i] = to_16x16(images[i], size, content, bbox).reshape(-1)
        if (i + 1) % 10000 == 0:
            print(f"  {tag} {i + 1}/{n}")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preview(x: np.ndarray, y: np.ndarray, size: int, k: int = 8) -> None:
    ramp = " .:-=+*#%@"
    levels = len(ramp) - 1
    for i in range(k):
        print(f"\n  样本 #{i}  标签={int(y[i])}")
        g = x[i].reshape(size, size)
        for row in g:
            # 转 int 后再运算：NumPy 2.x 下 uint8*256 会触发 OverflowError
            print("   " + "".join(ramp[min(levels, int(v) * levels // 255)] for v in row))


def main() -> int:
    ap = argparse.ArgumentParser(description="MNIST → 16×16 缓存")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--content", type=int, default=12,
                    help="数字内容所占像素边长（默认 12，对应 MNIST 20/28 边距比例）")
    ap.add_argument("--no-bbox", action="store_true", help="关闭包围盒裁剪")
    ap.add_argument("--force", action="store_true", help="重新下载与重建")
    ap.add_argument("--preview", action="store_true", help="打印 ASCII 预览")
    args = ap.parse_args()

    if not 1 <= args.content <= args.size:
        ap.error(f"--content 必须落在 [1, {args.size}]")

    root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] 获取原始 IDX 文件")
    paths = {k: download(v, raw_dir, args.force) for k, v in FILES.items()}

    print("[2/3] 解析与预处理")
    x_train = read_idx(paths["train_images"])
    y_train = read_idx(paths["train_labels"])
    x_test = read_idx(paths["test_images"])
    y_test = read_idx(paths["test_labels"])
    print(f"  原始: train={x_train.shape}, test={x_test.shape}")
    if y_train.shape[0] != x_train.shape[0] or y_test.shape[0] != x_test.shape[0]:
        raise RuntimeError("图像与标签数量不匹配")

    bbox = not args.no_bbox
    train16 = build_split(x_train, args.size, args.content, bbox, "train")
    test16 = build_split(x_test, args.size, args.content, bbox, "test")

    print("[3/3] 落盘")
    suffix = args.size
    out_pairs = [
        (f"mnist{suffix}_train.npy", train16),
        (f"mnist{suffix}_test.npy", test16),
        ("mnist_labels_train.npy", y_train),
        ("mnist_labels_test.npy", y_test),
    ]
    for name, arr in out_pairs:
        p = out_dir / name
        np.save(p, arr)
        print(f"  {name:26s} shape={str(arr.shape):16s} {arr.dtype}  {p.stat().st_size / 1024:.0f} KB")

    manifest = {
        "size": args.size,
        "content": args.content,
        "bbox": bbox,
        "interpolation": "BOX (area average)",
        "normalization": "x in [0,1], 不做标准化",
        "train_shape": list(train16.shape),
        "test_shape": list(test16.shape),
        "class_counts_train": np.bincount(y_train, minlength=10).tolist(),
        "class_counts_test": np.bincount(y_test, minlength=10).tolist(),
        "source_sha256": {k: sha256(v) for k, v in paths.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    print(f"  manifest.json 已写入")

    print("\n校验")
    print(f"  非零像素占比      : {(train16 > 0).mean():.3f}")
    print(f"  全零样本数        : {int((train16.sum(1) == 0).sum())}")
    print(f"  像素均值/标准差   : {train16.mean():.2f} / {train16.std():.2f}")
    print(f"  标签分布(训练)    : {np.bincount(y_train, minlength=10).tolist()}")

    if args.preview:
        preview(train16, y_train, args.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
