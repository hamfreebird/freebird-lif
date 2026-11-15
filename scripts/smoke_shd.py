"""M0 冒烟验证：SHD → LIF SNN → XPU 训练，端到端跑通。

目的不是刷精度，而是证明整条链路可用：
数据分箱正确、替代梯度能回传、XPU 上不报错、损失确实下降。
若精度停在 5%（随机水平）附近，说明数据或标签有错，后续全部白做。

用法: python scripts/smoke_shd.py [--T 100] [--epochs 5] [--batch 512]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freebird_lif.neuron import LIF  # noqa: E402
from freebird_lif.shd.data import ROOT, build  # noqa: E402


class SmokeSNN(nn.Module):
    """700 → 256 → 256 → 20，LIF + 泄漏积分读出。"""

    def __init__(self, T: int, hidden: int = 256, bn: bool = True, thr: float = 1.0):
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(700, hidden, bias=not bn)
        self.fc2 = nn.Linear(hidden, hidden, bias=not bn)
        self.fc3 = nn.Linear(hidden, 20)
        # BN 是必需的：首层预激活 std 约 0.28，远低于阈值 1.0，
        # 无 BN 时 LIF 几乎不发放，深层拿不到输入，梯度虽在但学不动。
        self.bn1 = nn.BatchNorm1d(hidden) if bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(hidden) if bn else nn.Identity()
        kw = dict(tau=2.0, dt=1.0, thr=thr, alpha=2.0, method="exponential")
        self.l1 = LIF(**kw)
        self.l2 = LIF(**kw)
        self.out = LIF(spiking=False, **kw)

    def forward(self, x):  # x: [B,T,700]
        b = x.shape[0]
        u1, _ = self.l1.init_state(b, 256, device=x.device)
        u2, _ = self.l2.init_state(b, 256, device=x.device)
        u3, _ = self.out.init_state(b, 20, device=x.device)
        acc = 0.0
        for t in range(self.T):
            s1, u1, _ = self.l1(self.bn1(self.fc1(x[:, t])), u1, None)
            s2, u2, _ = self.l2(self.bn2(self.fc2(s1)), u2, None)
            o, u3, _ = self.out(self.fc3(s2), u3, None)
            acc = acc + o
        return acc / self.T


@torch.no_grad()
def evaluate(model, x, y, bs=512):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i : i + bs]).float().to(next(model.parameters()).device)
        logits = model(xb)
        correct += (logits.argmax(1).cpu() == torch.from_numpy(y[i : i + bs])).sum().item()
    model.train()
    return correct / len(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no-bn", action="store_true")
    ap.add_argument("--thr", type=float, default=1.0)
    args = ap.parse_args()

    dev = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    print(f"device = {dev}")

    d = build(ROOT / "data" / "shd", T=args.T)
    xn, yn = d["x_train"], d["y_train"]
    xv, yv = d["x_val"], d["y_val"]
    xt, yt = d["x_test"], d["y_test"]
    print(f"train {xn.shape}  val {xv.shape}  test {xt.shape}")

    model = SmokeSNN(args.T, bn=not args.no_bn, thr=args.thr).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params = {n_par:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    g = torch.Generator().manual_seed(0)

    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(xn), generator=g)
        total = 0.0
        for i in range(0, len(xn), args.batch):
            idx = perm[i : i + args.batch]
            xb = torch.from_numpy(xn[idx.numpy()]).float().to(dev, non_blocking=True)
            yb = torch.from_numpy(yn[idx.numpy()]).to(dev, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        acc = evaluate(model, xv, yv)
        print(f"  ep{ep + 1:2d}/{args.epochs}  loss={total / len(xn):.4f}  "
              f"val={acc:.4f}  {time.time() - t0:5.1f}s", flush=True)

    test_acc = evaluate(model, xt, yt)
    print(f"\n  test = {test_acc:.4f}   （随机水平 0.05，文献 FF-LIF 约 0.87）")

    ok = test_acc > 0.25 and acc > 0.20
    print("  冒烟判定:", "通过" if ok else "未通过——检查数据或梯度链路")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
