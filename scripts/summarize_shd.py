"""把 train_shd.py 产出的多轮网格结果汇总成表。

一次调优会留下 r1..rN 多个 json。人工比对容易看漏，且不同轮次的配置差异
（epoch、验证方式、是否循环）必须一起显示，否则会把不可比的两行并列。

用法:
    python scripts/summarize_shd.py                # 全部轮次
    python scripts/summarize_shd.py --min-epoch 60 # 只看训练充分的
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"


def load() -> list[dict]:
    rows = []
    for p in sorted(BENCH.glob("train_shd_*.json")):
        arm = p.stem.replace("train_shd_", "")
        for r in json.loads(p.read_text("utf-8")):
            r["arm"] = arm
            rows.append(r)
    return rows


def fmt(r: dict) -> str:
    rec = "R" if r.get("recurrent") else "-"
    tau = (f"het{r.get('tau_lo', 1):g}-{r.get('tau_hi', 20):g}"
           if r.get("tau_mode") == "het" else f"{r.get('tau_ms', 2):g}")
    aug = []
    if r.get("aug_chan"):
        aug.append(f"c{r['aug_chan']}")
    if r.get("aug_time"):
        aug.append(f"t{r['aug_time']}")
    if r.get("aug_drop"):
        aug.append(f"d{r['aug_drop']:g}")
    reg = []
    if r.get("dropout"):
        reg.append(f"do{r['dropout']:g}")
    if r.get("in_dropout"):
        reg.append(f"ido{r['in_dropout']:g}")
    if r.get("wd", 1e-4) != 1e-4:
        reg.append(f"wd{r['wd']:g}")
    return (f"| {r['tag']:<22} | {r['arm']:<3} | {rec} | {r['n_layers']}×{r['hidden']} | "
            f"{tau:<9} | {r.get('norm', 'ln'):<4} | {','.join(reg) or '—':<12} | "
            f"{','.join(aug) or '—':<7} | {r['epochs_run']:>3} | "
            f"{r['val']:.4f} | {r['test']:.4f} | {r['final_train_acc']:.4f} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-epoch", type=int, default=0)
    ap.add_argument("--top", type=int, default=0, help=">0 时只打印前 N 行")
    ap.add_argument("--md", default=None, help="写入 markdown 文件")
    args = ap.parse_args()

    rows = [r for r in load() if r["epochs_run"] >= args.min_epoch]
    rows.sort(key=lambda r: -r["test"])
    if args.top:
        rows = rows[: args.top]

    head = ("| 配方 | 轮 | 循环 | 结构 | τ(帧) | 归一 | 正则 | 增强 | ep | "
            "val | test | train |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head] + [fmt(r) for r in rows]
    out = "\n".join(lines)
    print(out)
    if args.md:
        Path(args.md).write_text(out + "\n", "utf-8")
        print(f"\n→ {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
