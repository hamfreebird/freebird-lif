"""LIF 变体在 SHD 上的对比扫描。

统一架构、统一配方，唯一变量是神经元方程。这是 RQ1 的执行入口。

用法:
    python scripts/sweep_variants.py --epochs 40 --seeds 0
    python scripts/sweep_variants.py --epochs 100 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freebird_lif.shd.data import ROOT, build  # noqa: E402
from freebird_lif.variants import REGISTRY, build as build_neuron  # noqa: E402

OUT = ROOT / "benchmarks"
N_IN, N_HID, N_CLS = 700, 256, 20


class SNN(nn.Module):
    """700 → 256 → 256 → 20，读出层为泄漏积分（不脉冲、不重置）。

    归一化用 LayerNorm 而非 BatchNorm：M0 实测 BN 在此场景下会把 0.025 的
    原始激活放大 40 倍，running stats 稍有偏差就导致 eval 崩塌
    （train 模式 val 41.2% vs eval 9.2%）。
    """

    def __init__(self, variant: str, T: int, thr: float = 0.5,
                 tau_ms: float = 2.0, hidden: int = N_HID):
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(N_IN, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, N_CLS)
        self.n1 = nn.LayerNorm(hidden)
        self.n2 = nn.LayerNorm(hidden)
        # learn_tau 对所有变体统一为 True：否则 LIF 有 0 个神经元参数、
        # 其余变体有 1–5 个，"时间常数可学习"这个好处会被错误地算进变体差异里。
        base = dict(thr=thr, tau_ms=tau_ms, learn_tau=True)
        self.l1 = build_neuron(variant, hidden, **base)
        self.l2 = build_neuron(variant, hidden, **base)
        self.alpha_out = 0.6

    def forward(self, x):  # [B,T,700]
        b, T = x.shape[0], x.shape[1]
        dev = x.device
        s1 = self.l1.init_state(b, self.fc1.out_features, device=dev, dtype=x.dtype)
        s2 = self.l2.init_state(b, self.fc2.out_features, device=dev, dtype=x.dtype)
        u3 = torch.zeros(b, N_CLS, device=dev, dtype=x.dtype)
        acc = 0.0
        for t in range(T):
            sp1, s1 = self.l1.step(self.n1(self.fc1(x[:, t])), s1)
            sp2, s2 = self.l2.step(self.n2(self.fc2(sp1)), s2)
            u3 = self.alpha_out * u3 + (1 - self.alpha_out) * self.fc3(sp2)
            acc = acc + u3
        return acc / T


@torch.no_grad()
def evaluate(model, x, y, bs=512):
    model.eval()
    c = 0
    dev = next(model.parameters()).device
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i : i + bs]).float().to(dev)
        c += (model(xb).argmax(1).cpu() == torch.from_numpy(y[i : i + bs])).sum().item()
    model.train()
    return c / len(x)


def train_one(variant: str, d: dict, epochs: int, seed: int, lr: float,
              bs: int, thr: float, dev) -> dict:
    torch.manual_seed(seed)
    model = SNN(variant, d["meta"]["T"], thr=thr).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    xn, yn = d["x_train"], d["y_train"]
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(xn), generator=g)
        for i in range(0, len(xn), bs):
            idx = perm[i : i + bs].numpy()
            xb = torch.from_numpy(xn[idx]).float().to(dev, non_blocking=True)
            yb = torch.from_numpy(yn[idx]).to(dev, non_blocking=True)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    return {
        "variant": variant, "seed": seed,
        "val": evaluate(model, d["x_val"], d["y_val"]),
        "test": evaluate(model, d["x_test"], d["y_test"]),
        "params": sum(p.numel() for p in model.parameters()),
        "neuron_params": sum(p.numel() for n in (model.l1, model.l2)
                             for p in n.parameters() if p.requires_grad),
        "n_state": model.l1.n_state, "ops": model.l1.op_count(),
        "seconds": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(REGISTRY))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--tag", default="screen")
    args = ap.parse_args()

    dev = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    print(f"device={dev}  T={args.T}  epochs={args.epochs}  seeds={args.seeds}  thr={args.thr}")
    d = build(ROOT / "data" / "shd", T=args.T)

    rows = []
    for v in args.variants:
        for s in args.seeds:
            r = train_one(v, d, args.epochs, s, args.lr, args.batch, args.thr, dev)
            rows.append(r)
            print(f"  {v:7s} seed={s}  val={r['val']:.4f}  test={r['test']:.4f}  "
                  f"({r['seconds']:.0f}s, 神经元参数 {r['neuron_params']})", flush=True)

    print(f"\n{'变体':8s}{'val':>8s}{'test':>8s}{'σ':>7s}{'状态':>5s}{'ops':>5s}{'神经元参数':>10s}")
    summary = []
    for v in args.variants:
        rs = [r for r in rows if r["variant"] == v]
        if not rs:
            continue
        vals = [r["test"] for r in rs]
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        m = statistics.fmean(vals)
        summary.append({"variant": v, "mean": m, "std": sd, "n": len(vals),
                        "n_state": rs[0]["n_state"], "ops": rs[0]["ops"],
                        "neuron_params": rs[0]["neuron_params"],
                        "val": statistics.fmean([r["val"] for r in rs])})
        print(f"{v:8s}{statistics.fmean([r['val'] for r in rs]):>8.4f}{m:>8.4f}"
              f"{sd:>7.4f}{rs[0]['n_state']:>5d}{rs[0]['ops']:>5.0f}{rs[0]['neuron_params']:>10d}")

    if len(args.seeds) > 1:
        print("\n注：n=2264 时测试集 95% 置信区间半宽约 1.0–1.4 个百分点，"
              "differences 小于该值不可区分。")

    OUT.mkdir(exist_ok=True)
    (OUT / f"variants_{args.tag}.json").write_text(
        json.dumps({"config": vars(args), "rows": rows, "summary": summary},
                   indent=2, ensure_ascii=False), "utf-8")
    print(f"\n→ {OUT / f'variants_{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
