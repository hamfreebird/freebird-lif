"""命令行入口。

    python -m freebird_lif.cli predict digit.npy
    python -m freebird_lif.cli predict digit.png --invert auto
    python -m freebird_lif.cli eval

温度校准是一次性离线步骤，用 `python scripts/calibrate.py`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .api import DEFAULT_BUNDLE, DEFAULT_CALIB, Predictor
from .data import to_16x16

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_input(path: Path, invert: str = "auto") -> np.ndarray:
    """读入一张图并规范成 [16,16] uint8。

    .npy 直接按数组读取（不做任何缩放/裁剪，视为已预处理）；
    位图走与训练一致的预处理，可选反色：MNIST 是白字黑底，
    而多数手写拍照是黑字白底。
    """
    if path.suffix.lower() == ".npy":
        a = np.load(path)
        if a.ndim == 2 and a.shape == (16, 16):
            return a.astype(np.uint8)
        if a.ndim == 1 and a.size == 256:
            return a.reshape(16, 16).astype(np.uint8)
        raise ValueError(f"npy 形状应为 (16,16) 或 (256,)，实际 {a.shape}")

    if path.suffix.lower() not in IMG_EXT:
        raise ValueError(f"不支持的扩展名 {path.suffix}，接受 .npy 与 {sorted(IMG_EXT)}")

    from PIL import Image

    img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if invert == "auto":
        # 边缘均值偏亮 → 判定为白底黑字，需反色
        border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
        do_inv = border.mean() > 127
    else:
        do_inv = invert == "yes"
    if do_inv:
        img = 255 - img
    return to_16x16(img)


def cmd_predict(args) -> int:
    path = Path(args.image)
    if not path.exists():
        print(f"文件不存在: {path}", file=sys.stderr)
        return 2
    x = load_input(path, args.invert)
    pred = Predictor(args.bundle, args.calibration, args.threshold)
    r = pred.predict_one(x)
    if args.json:
        print(json.dumps({"digit": r.digit, "confidence": round(r.confidence, 6),
                          "accepted": r.accepted}, ensure_ascii=False))
    else:
        tag = "" if r.accepted else "  [低于拒识阈值]"
        print(f"数字 {r.digit}    置信度 {r.confidence:.4f}{tag}")
        if args.top:
            for i, p in pred.explain(x, args.top)[0]["top"]:
                print(f"  {i}: {p:.4f}")
    return 0


def cmd_eval(args) -> int:
    from .data import load_raw, to_tensor

    raw = load_raw()
    n = args.n
    x = to_tensor(raw["x_test"][:n]).numpy()
    y = raw["y_test"][:n].astype(np.int64)
    pred = Predictor(args.bundle, args.calibration, args.threshold)
    d, c = pred.predict(x)
    acc = float((d == y).mean())
    print(f"样本 {n}  精度 {acc:.4f}  平均置信度 {c.mean():.4f}  (T_cal={pred.temperature:.4f})")
    if pred.threshold is not None:
        keep = c >= pred.threshold
        print(f"拒识 τ={pred.threshold}: 覆盖率 {keep.mean():.3f}  覆盖内精度 {float((d[keep] == y[keep]).mean()):.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="freebird-lif", description="LIF 脉冲神经网络数字识别")
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--calibration", default=str(DEFAULT_CALIB))
    ap.add_argument("--threshold", type=float, default=None, help="覆盖 calibration.json 中的 τ")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="识别一张图")
    p.add_argument("image")
    p.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--top", type=int, default=0, help="额外输出前 k 个候选")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_predict)

    e = sub.add_parser("eval", help="在测试集上评估")
    e.add_argument("--n", type=int, default=10_000)
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
