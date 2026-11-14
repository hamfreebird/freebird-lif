"""训练入口：全量训练、增强、检查点与曲线记录。

用法：
    python -m freebird_lif.train                        # 默认 45 epoch，T=8
    python -m freebird_lif.train --T 16 --epochs 12     # 扫描用
    python -m freebird_lif.train --no-aug               # 消融用

产物（runs/<name>/）：
    best.pt       state_dict + SNNConfig
    config.json   训练超参
    history.json  逐 epoch 曲线
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import accuracy, augment_batch, load_raw, split_train_val, to_label, to_tensor
from .network import ConvSNN, SNNConfig, recalibrate_bn

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
    epochs: int = 45
    batch: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    val_size: int = 5_000
    seed: int = 0
    aug_shift: float = 1.0
    aug_deg: float = 5.0
    aug_bright: float = 0.1
    name: str = "main"
    recalibrate_bn: bool = True
    snn: SNNConfig = field(default_factory=SNNConfig)


def train(cfg: TrainConfig, out_root: Path, data_dir: Path | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    raw = load_raw(data_dir)
    xt, yt, xv, yv = split_train_val(
        to_tensor(raw["x_train"]), to_label(raw["y_train"]), cfg.val_size, seed=2024
    )
    xte, yte = to_tensor(raw["x_test"]), to_label(raw["y_test"])

    model = ConvSNN(cfg.snn)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    out = out_root / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{cfg.name}] train={len(xt)} val={len(xv)} test={len(xte)} "
          f"T={cfg.snn.T} params={model.param_count()} MACs={model.amortized_macs()}", flush=True)

    hist = {"epoch": [], "loss": [], "val_acc": [], "lr": []}
    best_acc, best_ep = 0.0, -1
    t0 = time.time()

    for ep in range(cfg.epochs):
        model.train()
        g = torch.Generator().manual_seed(cfg.seed * 1000 + ep)
        perm = torch.randperm(len(xt), generator=g)
        total, seen = 0.0, 0
        for i in range(0, len(xt), cfg.batch):
            idx = perm[i : i + cfg.batch]
            xb = xt[idx]
            if cfg.aug_shift > 0 or cfg.aug_deg > 0:
                xb = augment_batch(xb, cfg.aug_shift, cfg.aug_deg, cfg.aug_bright, g)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        sched.step()

        acc = accuracy(model, xv, yv)
        hist["epoch"].append(ep + 1)
        hist["loss"].append(total / seen)
        hist["val_acc"].append(acc)
        hist["lr"].append(opt.param_groups[0]["lr"])
        mark = ""
        if acc > best_acc:
            best_acc, best_ep = acc, ep + 1
            torch.save(
                {"state_dict": model.state_dict(), "snn_config": asdict(cfg.snn)},
                out / "best.pt",
            )
            mark = "  *"
        print(f"  ep{ep + 1:3d}/{cfg.epochs}  loss={total / seen:.4f}  "
              f"val={acc:.4f}  lr={opt.param_groups[0]['lr']:.2e}  "
              f"{time.time() - t0:5.0f}s{mark}", flush=True)

    # 用验证集最优的检查点评测试集，避免用末轮（可能过拟合）的权重
    ckpt = torch.load(out / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_before = accuracy(model, xte, yte)

    # 训练期增强会污染 BN 统计量，部署前必须用未增强数据重校准
    if cfg.recalibrate_bn:
        recalibrate_bn(model, xt)
        model.eval()
    test_acc = accuracy(model, xte, yte)
    rates = model.spike_rates(xte[:2000])
    if cfg.recalibrate_bn:
        torch.save({"state_dict": model.state_dict(), "snn_config": asdict(cfg.snn)},
                   out / "best.pt")

    summary = {
        "name": cfg.name,
        "best_val": best_acc,
        "best_epoch": best_ep,
        "test_acc": test_acc,
        "test_acc_before_recalib": test_before,
        "recalibrated": cfg.recalibrate_bn,
        "spike_rates": rates,
        "params": model.param_count(),
        "amortized_macs": model.amortized_macs(),
        "train_seconds": time.time() - t0,
        "T": cfg.snn.T,
        "surrogate": cfg.snn.surrogate,
    }
    (out / "config.json").write_text(
        json.dumps({"train": {k: v for k, v in asdict(cfg).items() if k != "snn"},
                    "snn": asdict(cfg.snn)}, indent=2, ensure_ascii=False), "utf-8")
    (out / "history.json").write_text(json.dumps(hist, indent=2), "utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")

    print(f"\n  最优 val={best_acc:.4f} (ep{best_ep})   test={test_acc:.4f}  "
          f"用时 {summary['train_seconds']:.0f}s")
    print(f"  发放率 {({k: round(v, 4) for k, v in rates.items()})}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="freebird-lif 训练")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--surrogate", default="fast_sigmoid")
    ap.add_argument("--pool-on", default="membrane")
    ap.add_argument("--val-size", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-recalib", action="store_true", help="关闭 BN 重校准")
    ap.add_argument("--aug-shift", type=float, default=1.0)
    ap.add_argument("--aug-deg", type=float, default=5.0)
    ap.add_argument("--aug-bright", type=float, default=0.1)
    args = ap.parse_args()

    snn = SNNConfig(T=args.T, surrogate=args.surrogate, pool_on=args.pool_on)
    cfg = TrainConfig(
        epochs=args.epochs, batch=args.batch, lr=args.lr, val_size=args.val_size,
        seed=args.seed, snn=snn,
        aug_shift=0.0 if args.no_aug else args.aug_shift,
        aug_deg=0.0 if args.no_aug else args.aug_deg,
        aug_bright=0.0 if args.no_aug else args.aug_bright,
        name=args.name or f"T{args.T}_{args.surrogate}{'_noaug' if args.no_aug else ''}",
        recalibrate_bn=not args.no_recalib,
    )
    train(cfg, args.out)


if __name__ == "__main__":
    main()
