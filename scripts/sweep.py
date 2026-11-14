"""超参扫描：时间步长 T 与替代梯度函数。

产出 benchmarks/sweep.md 与逐配置 runs/sweep/<name>/。

用法:
    python scripts/sweep.py --what T --epochs 10
    python scripts/sweep.py --what surrogate --epochs 10
    python scripts/sweep.py --what T --epochs 10 --Ts 2 4 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freebird_lif.train import TrainConfig, train
from freebird_lif.network import SNNConfig

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "sweep"


def run_one(name: str, snn: SNNConfig, epochs: int, seed: int) -> dict:
    cfg = TrainConfig(epochs=epochs, seed=seed, name=name, snn=snn)
    return train(cfg, OUT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["T", "surrogate", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--Ts", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--surrogates", nargs="+",
                    default=["fast_sigmoid", "atan", "triangle", "rect"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows: list[dict] = []

    if args.what in ("T", "all"):
        print("\n===== 时间步长 T 扫描 =====")
        for T in args.Ts:
            s = run_one(f"T{T}", SNNConfig(T=T), args.epochs, args.seed)
            rows.append({"axis": "T", "value": T, **s})

    if args.what in ("surrogate", "all"):
        print("\n===== 替代梯度对照 (T=8) =====")
        for fn in args.surrogates:
            s = run_one(f"sur_{fn}", SNNConfig(T=8, surrogate=fn), args.epochs, args.seed)
            rows.append({"axis": "surrogate", "value": fn, **s})

    lines = [
        "# 超参扫描结果",
        "",
        f"每配置 {args.epochs} epoch，seed={args.seed}，val=5000，官方测试集隔离。",
        "",
        "| 轴 | 取值 | 最优 val | test | 训练秒 | 发放率 lif1/lif2/lif3 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        rt = r["spike_rates"]
        lines.append(
            f"| {r['axis']} | {r['value']} | {r['best_val']:.4f} | {r['test_acc']:.4f} "
            f"| {r['train_seconds']:.0f} | {rt['lif1']:.3f}/{rt['lif2']:.3f}/{rt['lif3']:.3f} |"
        )
    md = "\n".join(lines) + "\n"
    (ROOT / "benchmarks").mkdir(exist_ok=True)
    (ROOT / "benchmarks" / f"sweep_{args.what}.md").write_text(md, "utf-8")
    (ROOT / "benchmarks" / f"sweep_{args.what}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), "utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
