"""XPU vs CPU 微基准。M0 的判定闸门之一。

测三件事：
1. 设备是否可见、属性如何
2. 裸 matmul 的有效吞吐（判断硬件上限）
3. 类 SNN 负载的端到端耗时（判断真实训练前的期望值）

判据：类 SNN 负载上 XPU 相对 CPU 加速 ≥ 2×，否则按 CPU 规划。

用法: python scripts/bench_xpu.py [--T 100]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn


def timed(fn, n: int, warmup: int = 3, sync=None) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
        if sync:
            sync()
    ts = []
    for _ in range(n):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return statistics.fmean(ts) * 1e3, ts[len(ts) // 2] * 1e3


def device_report() -> tuple[bool, torch.device]:
    print("=" * 62)
    print("torch", torch.__version__)
    ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    print("torch.xpu.is_available():", ok)
    if not ok:
        return False, torch.device("cpu")
    print("device count:", torch.xpu.device_count())
    for i in range(torch.xpu.device_count()):
        p = torch.xpu.get_device_properties(i)
        print(f"  [{i}] {p}")
    return True, torch.device("xpu")


def matmul_bench(dev: torch.device, sync) -> dict:
    """1024³ 与 4096×1024×1024 两组，覆盖小/大矩阵。"""
    out = {}
    for n, m, k in ((1024, 1024, 1024), (4096, 1024, 1024)):
        a = torch.randn(n, k, device=dev, dtype=torch.float32)
        b = torch.randn(k, m, device=dev, dtype=torch.float32)
        mean_ms, _ = timed(lambda: a @ b, 10, sync=sync)
        tflops = 2 * n * m * k / (mean_ms / 1e3) / 1e12
        out[f"{n}x{k}x{m}"] = {"ms": mean_ms, "tflops": tflops}
        print(f"  matmul {n}x{k}x{m}: {mean_ms:.2f} ms  → {tflops:.2f} TFLOPS")
    return out


class SNNLike(nn.Module):
    """代表性负载：3 层全连接 + 逐时间步的脉冲非线性。

    不实现真实神经元，只复现 SNN 训练的开销结构——
    T 步串行、每步多次小算子，这才是决定实际速度的地方。
    """

    def __init__(self, n_in=700, hidden=256, n_out=20):
        super().__init__()
        self.l1 = nn.Linear(n_in, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, n_out)

    def forward(self, x):  # x: [B, T, n_in]
        u1 = u2 = None
        acc = 0.0
        for t in range(x.shape[1]):
            h = torch.sigmoid(self.l1(x[:, t]))
            u1 = h if u1 is None else 0.7 * u1 + 0.3 * h
            s1 = (u1 > 0.5).float()
            h2 = torch.sigmoid(self.l2(s1))
            u2 = h2 if u2 is None else 0.7 * u2 + 0.3 * h2
            s2 = (u2 > 0.5).float()
            acc = acc + self.l3(s2)
        return acc


def snn_bench(dev: torch.device, sync, T: int, batches: list[int]) -> dict:
    """类 SNN 负载的批大小扫描。

    必须扫批而不是测单点：SNN 的 T 步串行展开使每次迭代产生数百次小算子调用，
    GPU 在批很小时完全是启动开销受限，算力大量闲置。单点测量会得出
    "GPU 无收益"的错误结论。
    """
    out = {}
    model = SNNLike().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for b in batches:
        x = torch.rand(b, T, 700, device=dev)
        y = torch.randint(0, 20, (b,), device=dev)

        def step():
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        mean_ms, _ = timed(step, 5, warmup=2, sync=sync)
        out[b] = {"ms_per_step": mean_ms, "us_per_sample": mean_ms / b * 1e3}
        print(f"  SNN-like B={b:<5d} T={T}: {mean_ms:7.1f} ms/step  "
              f"→ {mean_ms / b * 1e3:6.0f} µs/样本")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--batches", type=int, nargs="+", default=[128, 512, 1024, 2048])
    args = ap.parse_args()

    has_xpu, xdev = device_report()
    devices = [torch.device("cpu")] + ([xdev] if has_xpu else [])

    results = {}
    for dev in devices:
        sync = (lambda: torch.xpu.synchronize()) if dev.type == "xpu" else None
        print(f"\n--- {dev.type.upper()} ---")
        results[dev.type] = {
            "matmul": matmul_bench(dev, sync),
            "snn": snn_bench(dev, sync, args.T, args.batches),
        }

    print("\n" + "=" * 62)
    if has_xpu and "cpu" in results and "xpu" in results:
        mm_c = results["cpu"]["matmul"]["4096x1024x1024"]["tflops"]
        mm_x = results["xpu"]["matmul"]["4096x1024x1024"]["tflops"]
        print(f"大 matmul: CPU {mm_c:.2f} → XPU {mm_x:.2f} TFLOPS  "
              f"({mm_x / mm_c:.1f}×)")
        print(f"\n{'B':>6} {'CPU ms':>10} {'XPU ms':>10} {'加速':>8}")
        best = 0.0
        for b in args.batches:
            c = results["cpu"]["snn"][b]["ms_per_step"]
            x = results["xpu"]["snn"][b]["ms_per_step"]
            sp = c / x
            best = max(best, sp)
            print(f"{b:>6} {c:>10.1f} {x:>10.1f} {sp:>7.2f}×")
        gate_b = [b for b in args.batches
                  if results["cpu"]["snn"][b]["ms_per_step"] /
                     results["xpu"]["snn"][b]["ms_per_step"] >= 2.0]
        print(f"\n类 SNN 负载最大加速 {best:.2f}×（闸门 ≥ 2.0×）")
        if gate_b:
            print(f"达标批大小: {gate_b}  建议取最大者做训练")
            print("判定: 通过")
            return 0
        print("判定: 未达标，应按 CPU 规划")
        return 1
    print("XPU 不可用，M0 闸门未通过；按 CPU 路径规划 RQ1。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
