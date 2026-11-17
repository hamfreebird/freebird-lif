"""RQ-A 的四个诊断仪器。

只报精度无法区分"循环没学到东西"与"循环学到了但没用上"。这四个仪器分别回答：

1. `--mem`   有效记忆长度：把输入截断到前 K 帧，看精度-K 曲线。
             若循环真的延长了记忆，小 K 处精度应明显更高。**这是最直接的判据。**
2. `--ac`    活动自相关：各层脉冲在时间上的自相关衰减常数。循环应更长。
3. `--grad`  梯度沿时间分布：BPTT 梯度在 t=1..T 上的范数，看是否在早期步消失。
4. `--self`  自持性：先喂输入若干帧再切断，看活动能否维持——决定系统是否处于
             边缘稳定（edge-of-chaos）。

实现约束
--------
**不复制模型的前向逻辑。** 用 forward hook 抓 `model.neurons[i]` 的输出，
这样诊断与被诊断的模型永远一致——阶段一的教训是两份实现必然漂移。

用法：
    python scripts/diagnose_rnn.py --ckpt runs/h0_strong_g12_s0.pt --all
    python scripts/diagnose_rnn.py --compare runs/h0_strong_g1_s0.pt runs/h0_strong_g12_s0.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from freebird_lif.shd.data import build  # noqa: E402
from train_shd import SHDNet, Cfg, predict  # noqa: E402

OUT = ROOT / "benchmarks"


# --------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------

def load(ckpt: Path, dev: torch.device) -> tuple[SHDNet, dict]:
    blob = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = Cfg(**blob["config"])
    m = SHDNet(cfg)
    m.load_state_dict(blob["state_dict"])
    m.to(dev).eval()
    return m, blob


class SpikeTap:
    """抓各层每个时间步的脉冲，并保留梯度。

    **不能用 forward hook。** 模型里神经元是 `ne.step(cur, state)` 这样的
    方法调用，不经过 `nn.Module.__call__`，forward hook 根本不会触发
    （第一版就栽在这里，`steps` 全空、stack 报 "non-empty TensorList"）。
    改为包装实例上的 `step`，退出时恢复。
    """

    def __init__(self, model: SHDNet, retain: bool = True):
        self.steps: list[list[torch.Tensor]] = [[] for _ in model.neurons]
        self._orig = []
        self._inst = []
        for i, ne in enumerate(model.neurons):
            self._orig.append(ne.step)
            self._inst.append(ne)
            ne.step = self._mk(i, ne.step, retain)

    def _mk(self, i: int, orig, retain: bool):
        sink = self.steps[i]

        def step(x, state):
            s, st = orig(x, state)
            if retain and s.requires_grad:
                s.retain_grad()
            sink.append(s)
            return s, st

        return step

    def close(self):
        """恢复原方法。必须调用，否则模型被永久改写。"""
        for ne, orig in zip(self._inst, self._orig):
            ne.step = orig

    def stack(self, layer: int) -> torch.Tensor:
        """[B, T, N]。"""
        return torch.stack(self.steps[layer], dim=1)


@torch.no_grad()
def run_with_tap(model: SHDNet, x: torch.Tensor) -> torch.Tensor:
    tap = SpikeTap(model)
    try:
        model(x)
        return torch.stack([tap.stack(i).detach() for i in range(len(model.neurons))])
    finally:
        tap.close()


# --------------------------------------------------------------------------
# 仪器 1：有效记忆长度
# --------------------------------------------------------------------------

@torch.no_grad()
def memory_curve(model: SHDNet, x: np.ndarray, y: np.ndarray,
                 ks=(20, 40, 60, 80, 100)) -> dict:
    """把第 K 帧之后清零，测精度。

    清零而不是缩短张量长度：模型的 T 由输入形状决定，缩短会改变动力学
    （也改变输出积分的分母），使不同 K 之间不可比。
    """
    dev = next(model.parameters()).device
    T = x.shape[1]
    out = {}
    for K in ks:
        if K > T:
            continue
        xt = x.copy()
        xt[:, K:] = 0
        acc = float((predict(model, xt) == y).mean())
        out[str(K)] = acc
    return out


# --------------------------------------------------------------------------
# 仪器 2：活动自相关
# --------------------------------------------------------------------------

def autocorr_time(spikes: torch.Tensor, max_lag: int = 20) -> dict:
    """各层脉冲的时间自相关。

    对每个 (样本, 神经元) 序列去均值后算 lag 相关，再对 (样本,神经元) 求平均。
    返回每层的 r(lag) 与积分自相关时间 tau_int = 1 + 2*Σ_{lag≥1} r(lag)。
    """
    out = {}
    B, T, N = spikes.shape
    z = spikes - spikes.mean(dim=1, keepdim=True)
    denom = (z * z).sum(dim=1).clamp_min(1e-8)          # [B,N]
    lags = range(1, min(max_lag, T - 1) + 1)
    rs = []
    for lag in lags:
        num = (z[:, :-lag] * z[:, lag:]).sum(dim=1)     # [B,N]
        rs.append(float((num / denom).mean()))
    tau = 1.0 + 2.0 * sum(max(r, 0.0) for r in rs)
    out["r"] = rs
    out["tau_int"] = tau
    out["rate"] = float(spikes.mean())
    return out


# --------------------------------------------------------------------------
# 仪器 3：梯度沿时间分布
# --------------------------------------------------------------------------

def grad_over_time(model: SHDNet, x: torch.Tensor, y: torch.Tensor) -> dict:
    """||dL/ds_t|| 在 t 上的分布（各层平均）。"""
    model.train()          # BN 不用；这里是为了与训练时的图一致
    tap = SpikeTap(model)
    try:
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        out = {}
        for i in range(len(model.neurons)):
            # 必须读**每个时间步原始张量**的 .grad。`stack` 会造一个新张量，
            # 它的 .grad 永远是 None——第一版就是这么写的，报了一串
            # "not a leaf Tensor" 的警告却静默返回 None。
            per_t = []
            for st in tap.steps[i]:
                if st.grad is None:
                    per_t = None
                    break
                per_t.append(float(st.grad.flatten(1).norm(dim=-1).mean()))
            out[f"layer{i}"] = per_t
        out["loss"] = loss.item()
        return out
    finally:
        tap.close()
        model.zero_grad(set_to_none=True)
        model.eval()


# --------------------------------------------------------------------------
# 仪器 4：自持性（切断输入后活动是否延续）
# --------------------------------------------------------------------------

@torch.no_grad()
def self_sustain(model: SHDNet, x: np.ndarray, warm: int = 30) -> dict:
    """先喂 warm 帧真实输入，其余帧清零，比较前后段的发放率。

    比值 >>1 说明活动在无输入时被循环自己维持——这是系统处于边缘稳定的迹象；
    比值 <<1 说明活动随输入一起消失。
    """
    xw = x.copy()
    xw[:, warm:] = 0
    dev = next(model.parameters()).device
    sp = run_with_tap(model, torch.from_numpy(xw).float().to(dev))  # [L,B,T,N]
    out = {}
    for i in range(sp.shape[0]):
        r_warm = float(sp[i][:, :warm].mean())
        r_after = float(sp[i][:, warm:].mean())
        out[f"layer{i}"] = {
            "rate_warm": r_warm, "rate_after": r_after,
            "ratio": r_after / max(r_warm, 1e-9),
        }
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def diagnose(ckpt: Path, d: dict, dev: torch.device, which: set[str],
             n_test: int = 2000) -> dict:
    model, blob = load(ckpt, dev)
    cfg = model.cfg
    x = d["x_test"][:n_test]
    y = d["y_test"][:n_test]

    res = {
        "ckpt": ckpt.name,
        "config": {k: v for k, v in asdict(cfg).items() if k != "tag"},
        "test_acc_full": float((predict(model, x) == y).mean()),
    }
    if "mem" in which:
        res["memory"] = memory_curve(model, x, y)
    if "ac" in which:
        sp = run_with_tap(model, torch.from_numpy(x[:512]).float().to(dev))
        res["autocorr"] = {f"layer{i}": autocorr_time(sp[i]) for i in range(sp.shape[0])}
    if "grad" in which:
        xb = torch.from_numpy(x[:128]).float().to(dev)
        yb = torch.from_numpy(y[:128]).to(dev)
        res["grad_time"] = grad_over_time(model, xb, yb)
    if "self" in which:
        res["self_sustain"] = self_sustain(model, x[:512])
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True,
                    help="一个或多个 runs/<tag>_s<seed>.pt")
    ap.add_argument("--all", action="store_true", help="四个仪器全跑")
    ap.add_argument("--mem", action="store_true")
    ap.add_argument("--ac", action="store_true")
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--self", action="store_true")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--out", default="rnn_diag.json")
    args = ap.parse_args()

    which = {k for k in ("mem", "ac", "grad", "self") if getattr(args, k)}
    if args.all or not which:
        which = {"mem", "ac", "grad", "self"}

    dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    d = build(ROOT / "data" / "shd", T=100)

    rows = []
    for c in args.ckpt:
        r = diagnose(c, d, dev, which, args.n_test)
        rows.append(r)
        print(f"\n=== {r['ckpt']}  test={r['test_acc_full']:.4f} "
              f"gain={r['config']['rec_gain']} tmp={r['config']['tau_mode']} ===")
        if "memory" in r:
            print("  记忆曲线 K→acc: " +
                  "  ".join(f"{k}:{v:.3f}" for k, v in r["memory"].items()))
        if "autocorr" in r:
            print("  自相关 tau_int: " + "  ".join(
                f"L{i}={v['tau_int']:.1f}(rate {v['rate']:.3f})"
                for i, v in r["autocorr"].items()))
        if "self_sustain" in r:
            print("  自持比 after/warm: " + "  ".join(
                f"L{i}={v['ratio']:.2f}" for i, v in r["self_sustain"].items()))
        if "grad_time" in r:
            for k, v in r["grad_time"].items():
                if isinstance(v, list):
                    print(f"  {k} 梯度: t=0 {v[0]:.2e} → t=T {v[-1]:.2e}  "
                          f"min {min(v):.2e} @t={int(np.argmin(v))}")

    OUT.mkdir(exist_ok=True)
    path = OUT / args.out
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), "utf-8")
    print(f"\n→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
