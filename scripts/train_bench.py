"""在多基准数据集上训练（RQ-C 用）。

存在的理由：`train_shd.py` 的 `run()` 已经是数据集无关的——它只读
`d["x_train"/"y_train"/"x_val"/"y_val"/"x_test"/"y_test"]`。所以这里不重写
训练逻辑，只负责：加载某个基准 → 按该数据集的维度构造 Cfg → 调 run()。

**刻意不复制训练循环。** 阶段一的教训是两条训练路径必然漂移；本阶段所有
数据集共用同一份 `run()`，差异只允许出现在数据层和 Cfg 的几个维度上。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from freebird_lif.bench import evtouch, nmnist  # noqa: E402
from train_shd import Cfg, run  # noqa: E402

LOADERS = {"evtouch": evtouch, "nmnist": nmnist}
OUT = ROOT / "benchmarks"


def load(name: str, T: int | None = None, limit_per_class: int | None = None) -> dict:
    mod = LOADERS[name]
    if name == "nmnist":
        return mod.build(T=T or 100, limit_per_class=limit_per_class)
    return mod.build(T=T)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--variant", default="lif")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--T", type=int, default=None, help="时间步（默认用数据集原生/100）")
    ap.add_argument("--in-scale", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rec-gain", type=float, default=1.0)
    ap.add_argument("--recurrent", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    d = load(args.dataset, T=args.T, limit_per_class=args.limit_per_class)
    m = d["meta"]
    print(f"{m['name']}: train {len(d['y_train'])} / val {len(d['y_val'])} / "
          f"test {len(d['y_test'])}  T={m['T']} C={m['C']} 类={m['n_classes']}",
          flush=True)

    tag = args.tag or f"{args.dataset}_{args.variant}"
    rows = []
    for seed in args.seeds:
        cfg = Cfg(
            tag=tag, variant=args.variant, epochs=args.epochs, seed=seed,
            n_layers=args.layers, hidden=args.hidden,
            n_in=m["C"], n_classes=m["n_classes"],
            in_scale=args.in_scale, lr=args.lr, batch=args.batch,
            recurrent=args.recurrent, rec_gain=args.rec_gain,
            # 跨数据集迁移的配方规则见 PLAN §5.2：τ 用帧数标定，不换算物理时间
            tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            # 归一化保持"无"——阶段二实测 LN 有害，这里不重复那个错误
            norm="none",
            # 增强沿用频道平移（对 700 通道的 SHD 是按频率轴；对触觉/视觉
            # 通道它等价于通道置换，语义不同，故默认关闭，由调用方显式开启）
            aug_chan=0,
            val_mode="random",
            frames=0,
        )
        rows.append(run(cfg, d, dev))
        OUT.mkdir(exist_ok=True)
        path = OUT / f"bench_{tag}.json"
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), "utf-8")

    print(f"\n{'配置':22s} {'seed':>5s} {'val':>8s} {'test':>8s} {'train':>8s}")
    for r in rows:
        print(f"{r['tag']:22s} {r['seed']:>5d} {r['val']:>8.4f} {r['test']:>8.4f} "
              f"{r['final_train_acc']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
