"""阶段五的因果仪器：C2（干预式调制指数）与 C3（跨尺度预测增量）。

设计原则与阶段四的仪器一致：**先在已知答案的玩具系统上标定，再碰网络**。
如果玩具系统上给不出设定值，网络上的数字就没有意义。

C2 协议（干预）：
    对每个 Δ ∈ {0,5,10,20,40}：
      U0  = 无任何扰动                    （基线）
      Up  = 只在 t0+Δ 给低层探测脉冲        （对照响应）
      Ut  = 只在 t0 扰动高层               （高层扰动自身的效应）
      Utp = 两者都有                        （测试响应）
    R_ctrl = mean|Up − U0|,  R_test = mean|Utp − Ut|
    M(Δ) = R_test / R_ctrl − 1
    用「探测脉冲的增量效应」而不是原始活动，才能把「高层改变了低层对输入的响应」
    与「高层自己往低层灌了电流」分开。

C3 协议（预测）：
    在 t0 取 [h_low, h_top]，预测该次试验的探测响应 R。
    基线只用 h_low；完整模型加 h_top。ΔR² 是跨尺度信息的增量。

用法：
    python scripts/causal_v5.py self-test
    python scripts/causal_v5.py net --ckpt runs/v4_atlif_het_s0.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_shd import Cfg, SHDNet  # noqa: E402


# ---------------------------------------------------------------------------
# 玩具系统（已知答案）
# ---------------------------------------------------------------------------

def toy_c2(gamma: float = 0.3, coupled: bool = True, T: int = 120, t0: int = 20,
           deltas=(0, 5, 10, 20, 40), probe_amp: float = 1.0,
           top_amp: float = 4.0, seed: int = 0) -> dict:
    """两单元系统：A 通过乘性增益调制 B 对探测脉冲的响应。

    a[t] = 0.95·a[t-1]                 （A 的慢衰减）
    g[t] = tanh(a[t])
    u[t] = 0.90·u[t-1] + probe[t]·(1 + γ·g_eff[t-1])

    coupled=True  → g_eff 来自被扰动的 a（存在因果）
    coupled=False → g_eff 来自一条独立同分布的 a'（无因果）

    解析期望：R_ctrl = 1 + γ·g(t0+Δ)，R_test = 1 + γ·g'(t0+Δ)，
    所以 M ≈ γ·(g' − g)/(1 + γ·g)。
    """
    a = np.zeros(T)
    a[0] = 0.5
    a_ind = np.zeros(T)
    a_ind[0] = 0.5
    rng = np.random.default_rng(seed)

    def run(perturb_top: bool, probe_at: int | None) -> np.ndarray:
        aa = a.copy()
        bb = a_ind.copy()
        g = np.tanh(aa if coupled else bb)
        u = np.zeros(T)
        for t in range(1, T):
            aa[t] = 0.95 * aa[t - 1]
            bb[t] = 0.95 * bb[t - 1] + rng.normal(0, 1e-9)
            # 扰动必须写在状态更新**之后**：第一版写在循环前，
            # 结果在 t=t0 被 aa[t]=0.95*aa[t-1] 直接覆盖，M 恒为 0。
            if perturb_top and t == t0:
                aa[t] += top_amp
            g[t] = np.tanh(aa[t] if coupled else bb[t])
            drive = probe_amp if probe_at is not None and t == probe_at else 0.0
            u[t] = 0.90 * u[t - 1] + drive * (1.0 + gamma * g[t - 1])
        return u

    out = {}
    for d in deltas:
        u0 = run(False, None)
        up = run(False, t0 + d)
        ut = run(True, None)
        utp = run(True, t0 + d)
        r_ctrl = float(np.abs(up - u0).max())
        r_test = float(np.abs(utp - ut).max())
        out[int(d)] = {"M": r_test / max(r_ctrl, 1e-9) - 1.0,
                       "r_ctrl": r_ctrl, "r_test": r_test}
    return out


def toy_c3(gamma: float = 0.3, coupled: bool = True, n_trials: int = 400,
           delta: int = 10, T: int = 60, t0: int = 10, seed: int = 0) -> dict:
    """预测「探测响应」，而不是预测原始信号。

    第一版让 h 预测 y[t]，结果 ΔR² 只有 0.008——因为慢高层注入到 y 的量
    早已被 y[t-1] 吸收，h 成了冗余信息。这不是实现错误，是**指标定义错了**：
    跨尺度信息体现在「低层对输入的响应被调制了多少」，不在信号本身。

    这一版与网络上的 C3 同构：
      - 每次试验的高层状态 h(t0) 不同；
      - 探测脉冲在 t0+Δ，其响应 R = 1 + γ·tanh(h(t0+Δ))；
      - 用 [u(t0)] 预测 R（基线），用 [u(t0), h(t0)] 预测 R（完整）。
    coupled=False 时 γ=0，h 与 R 无关，ΔR² 应≈0。
    """
    rng = np.random.default_rng(seed)
    H, U, R = [], [], []
    for _ in range(n_trials):
        h0 = rng.normal(0, 1.0)
        h = h0 * 0.95 ** np.arange(T)
        u0 = rng.normal(0, 1.0)
        u = u0 * 0.9 ** np.arange(T)
        gain = 1.0 + (gamma * np.tanh(h[t0 + delta]) if coupled else 0.0)
        # 加一点测量噪声：否则 coupled=False 时 R 恒为 1，方差为 0，
        # R² 直接除零。噪声幅度 0.02 远小于耦合效应（0.3），不掩盖信号。
        H.append(h0); U.append(u0); R.append(gain + rng.normal(0, 0.02))
    H, U, R = np.asarray(H), np.asarray(U), np.asarray(R)

    def fit(X):
        cut = int(len(R) * 0.7)
        A = np.hstack([X[:cut], np.ones((cut, 1))])
        w = np.linalg.lstsq(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ R[:cut], rcond=None)[0]
        pred = np.hstack([X[cut:], np.ones((len(R) - cut, 1))]) @ w
        var = ((R[cut:] - R[cut:].mean()) ** 2).sum()
        if var < 1e-12:
            return float("nan")
        return 1 - ((R[cut:] - pred) ** 2).sum() / var

    r2b = fit(U[:, None])
    r2f = fit(np.column_stack([U, H]))
    return {"r2_base": float(r2b), "r2_full": float(r2f), "delta_r2": float(r2f - r2b)}


def self_test() -> dict:
    """M0 闸门：指标必须在已知答案上给出设定值。"""
    coupled = toy_c2(gamma=0.3, coupled=True)
    uncoupled = toy_c2(gamma=0.3, coupled=False)
    m_c = max(abs(v["M"]) for v in coupled.values())
    m_u = max(abs(v["M"]) for v in uncoupled.values())

    c3_pos = toy_c3(gamma=0.3, coupled=True)
    c3_null = toy_c3(gamma=0.3, coupled=False)

    checks = {
        "C2 有因果时 |M| ≥ 0.10": m_c >= 0.10,
        "C2 无因果时 |M| ≤ 0.02": m_u <= 0.02,
        "C3 有耦合时 ΔR² ≥ 0.20": c3_pos["delta_r2"] >= 0.20,
        "C3 无耦合时 ΔR² ≤ 0.02": abs(c3_null["delta_r2"]) <= 0.02,
    }
    return {"coupled_M": {k: round(v["M"], 4) for k, v in coupled.items()},
            "uncoupled_M": {k: round(v["M"], 4) for k, v in uncoupled.items()},
            "max_M_coupled": round(m_c, 4), "max_M_uncoupled": round(m_u, 4),
            "c3_coupled": c3_pos, "c3_null": c3_null,
            "checks": checks, "pass": all(checks.values())}


# ---------------------------------------------------------------------------
# 网络适配器（M3 用；M0 只保证能跑通）
# ---------------------------------------------------------------------------

class Tap:
    def __init__(self, model: SHDNet):
        self.out = [[] for _ in model.neurons]
        self._orig = []
        for i, ne in enumerate(model.neurons):
            self._orig.append(ne.step)
            ne.step = self._mk(i, ne, ne.step)

    def _mk(self, i, ne, orig):
        def step(x, state):
            s, st = orig(x, state)
            self.out[i].append(st[ne.soma_index].detach())
            return s, st
        return step

    def close(self, model):
        for ne, o in zip(model.neurons, self._orig):
            ne.step = o


@torch.no_grad()
def _run(model, x, events):
    model.perturb = events
    tap = Tap(model)
    try:
        model(x)
    finally:
        tap.close(model)
    return [torch.stack(tap.out[i], 1) for i in range(len(model.neurons))]


def net_modulation(model: SHDNet, x: torch.Tensor, t0: int = 20,
                   deltas=(0, 5, 10, 20, 40), probe_amp: float = 5.0,
                   top_amp: float = 10.0, frac: float = 0.10,
                   seed: int = 0) -> dict:
    """在真实网络上跑 C2。探测加在第 0 层，扰动加在最高层。"""
    N = model.cfg.hidden
    top = model.cfg.n_layers - 1
    g = torch.Generator().manual_seed(seed)
    mask_probe = torch.zeros(N)
    mask_probe[torch.randperm(N, generator=g)[: max(1, int(N * frac))]] = 1.0
    mask_top = torch.zeros(N)
    mask_top[torch.randperm(N, generator=g)[: max(1, int(N * frac))]] = 1.0

    out = {}
    for d in deltas:
        t_probe = t0 + d
        ev_probe = {"t": t_probe, "layer": 0, "mask": mask_probe, "amp": probe_amp}
        ev_top = {"t": t0, "layer": top, "mask": mask_top, "amp": top_amp}
        u0 = _run(model, x, None)[0]
        up = _run(model, x, [ev_probe])[0]
        ut = _run(model, x, [ev_top])[0]
        utp = _run(model, x, [ev_top, ev_probe])[0]
        w = slice(t_probe, t_probe + 10)
        r_ctrl = float((up[:, w] - u0[:, w]).abs().mean())
        r_test = float((utp[:, w] - ut[:, w]).abs().mean())
        out[int(d)] = {"M": r_test / max(r_ctrl, 1e-9) - 1.0,
                       "r_ctrl": r_ctrl, "r_test": r_test}
    return out


def shuffled_topdown(model: SHDNet, seed: int = 0):
    """把自顶向下权重原地打乱（同分布），返回备份以便恢复。"""
    import copy
    backup = copy.deepcopy(model.state_dict())
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if "td" in name and p.dim() == 2:
            flat = p.detach().reshape(-1)
            p.data = flat[torch.randperm(flat.numel(), generator=g)].reshape(p.shape)
    return backup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="self-test",
                    choices=["self-test", "net"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.command == "self-test":
        r = self_test()
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0 if r["pass"] else 1

    if not args.ckpt:
        print("net 模式需要 --ckpt")
        return 2
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SHDNet(Cfg(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    from instruments_v4 import shd
    x_np, _ = shd(n=args.batch)
    x = torch.from_numpy(x_np).float()
    r = {"ckpt": args.ckpt, "modulation": net_modulation(model, x)}
    print(json.dumps(r, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
