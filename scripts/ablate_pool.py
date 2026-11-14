"""pool_on 对照实验：脉冲后池化 vs 膜电位池化。

两个分支除池化位置外完全一致：相同参数形状、相同初始化（同一 manual_seed）、
相同数据顺序（每 epoch 用固定种子重建打乱序列）、相同优化器与学习率曲线。
因此精度差异只能归因于池化位置。

用法:
    python scripts/ablate_pool.py                    # 默认 8 epoch，单种子
    python scripts/ablate_pool.py --epochs 12 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from freebird_lif.data import accuracy, load_raw, split_train_val, to_label, to_tensor
from freebird_lif.network import ConvSNN, SNNConfig


def train_arm(pool_on: str, xt, yt, xv, yv, epochs: int, seed: int, lr: float, bs: int):
    torch.manual_seed(seed)
    model = ConvSNN(SNNConfig(T=8, pool_on=pool_on))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    curve, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        g = torch.Generator().manual_seed(10_000 + ep)  # 两个分支共享同一批次顺序
        perm = torch.randperm(len(xt), generator=g)
        total = 0.0
        for i in range(0, len(xt), bs):
            idx = perm[i : i + bs]
            logits, _ = model(xt[idx])
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        acc = accuracy(model, xv, yv)
        curve.append(acc)
        print(f"  [{pool_on:8s}] ep{ep + 1:2d}/{epochs}  loss={total / len(xt):.4f}  "
              f"val={acc:.4f}  {time.time() - t0:5.1f}s", flush=True)

    return model, curve, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arms", nargs="+", default=["spike", "membrane"],
                    choices=["spike", "membrane"], help="只跑指定分支")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--val-size", type=int, default=10_000)
    args = ap.parse_args()

    raw = load_raw()
    xt, yt, xv, yv = split_train_val(
        to_tensor(raw["x_train"]), to_label(raw["y_train"]), args.val_size, seed=2024
    )
    xte, yte = to_tensor(raw["x_test"]), to_label(raw["y_test"])
    print(f"train={len(xt)}  val={len(xv)}  test={len(xte)}  epochs={args.epochs}\n")

    results: dict[str, list[dict]] = {"spike": [], "membrane": []}
    for seed in args.seeds:
        for pool_on in args.arms:
            print(f"seed={seed}  pool_on={pool_on}")
            model, curve, secs = train_arm(
                pool_on, xt, yt, xv, yv, args.epochs, seed, args.lr, args.batch
            )
            te = accuracy(model, xte, yte)
            results[pool_on].append({"seed": seed, "curve": curve, "test": te, "secs": secs})
            print(f"  -> test_acc={te:.4f}  用时 {secs:.0f}s\n")

    print("=" * 62)
    print(f"{'pool_on':10s}{'val 最优':>10s}{'val 末轮':>10s}{'test':>10s}{'用时':>10s}")
    for pool_on in args.arms:
        runs = results[pool_on]
        best = sum(max(r["curve"]) for r in runs) / len(runs)
        last = sum(r["curve"][-1] for r in runs) / len(runs)
        test = sum(r["test"] for r in runs) / len(runs)
        secs = sum(r["secs"] for r in runs) / len(runs)
        print(f"{pool_on:10s}{best:10.4f}{last:10.4f}{test:10.4f}{secs:9.0f}s")

    if len(args.seeds) > 1:
        print("\n逐种子 test 精度：")
        for pool_on in args.arms:
            vals = [f"{r['test']:.4f}" for r in results[pool_on]]
            print(f"  {pool_on:10s} {' '.join(vals)}")

    d = (sum(r["test"] for r in results["membrane"]) - sum(r["test"] for r in results["spike"])) / len(args.seeds)
    print(f"\nmembrane - spike = {d:+.4f}")


if __name__ == "__main__":
    main()
