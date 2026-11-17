"""H1 核实（含一次自我纠正）：循环耦合到底有多强。

最初的 H1 假设
--------------
PLAN 里写的是："循环项被 (1-α) 缩放，而文献写法不缩放，所以 rec_gain=1
在两种写法下相差 1/(1-α)"。

**这个推理是错的，M0 实测推翻了它。** 原因是本项目把循环电流并入神经元的
输入 `cur`，而 `W_in·x` 也走同一个 `cur`：

    cur = W_in·x + W_rec·s
    u   = αu + (1-α)·cur

两者**同乘** `(1-α)`，所以 `W_rec/W_in` 的比值与 `(1-α)` 无关。再逐神经元
缩放循环项并不能"恢复文献等价"，它只是**单方面把循环耦合放大** `1/(1-α)` 倍。

那真正的现象是什么
------------------
实测（`rec_gain=1`、默认 `nn.Linear` 初始化、定稿配方的 in_scale=5）：
**循环电流的幅值只有前馈的 3.7%**。也就是说阶段二所有标着 "recurrent=True"
的实验，其循环通路实际上几乎不存在。这才是需要解释和修正的东西。

本脚本量化三件事
----------------
1. 定稿配方下 `1/(1-α)` 的真实分布（异构 τ 导致层内跨 25 倍）
2. `rec_scale` 对"循环/前馈"比值的实测影响（这是它唯一的实际作用）
3. `rec_gain` 扫描：要让循环与前馈同量级（比值 ≈1）需要多大增益
   —— 这是 M1 干预实验的标定起点，不能凭猜

用法：python scripts/diagnose_h1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_shd import Cfg, SHDNet  # noqa: E402

OUT = ROOT / "benchmarks" / "h1_rec_scale.json"


def scale_stats(model) -> dict:
    """逐层 1/(1-α) 与对应 τ 的分布。"""
    out = {}
    for i, ne in enumerate(model.neurons):
        sc = ne.input_scale().detach()
        tau = -1.0 / torch.log(ne.decay.detach())
        out[f"layer{i}"] = {
            "scale_min": float(sc.min()), "scale_median": float(sc.median()),
            "scale_max": float(sc.max()),
            "span": float(sc.max() / sc.min()),
            "tau_min": float(tau.min()), "tau_max": float(tau.max()),
        }
    return out


@torch.no_grad()
def drive_ratio(rec_gain: float, rec_scale: bool, in_scale: float,
                hidden: int = 384, layers: int = 2, batch: int = 32,
                T: int = 40, seed: int = 0, tau_hi: float = 40.0,
                rec_init: str = "default") -> dict:
    """实测"循环电流 / 前馈电流"的幅值比。

    逐时间步、逐层比较 `|W_rec·s|` 与 `|W_in·x|` 的均值。第 0 步没有 s_prev，
    故每次运行有 (T-1)×L 个测量点。
    """
    torch.manual_seed(seed)
    cfg = Cfg(recurrent=True, rec_scale=rec_scale, rec_gain=rec_gain,
              rec_init=rec_init,
              hidden=hidden, n_layers=layers, tau_mode="het", tau_lo=1.0,
              tau_hi=tau_hi, norm="none", in_scale=in_scale, frames=T)
    m = SHDNet(cfg).eval()

    x = torch.randn(batch, T, 700)
    h = x * cfg.in_scale
    ff, rc = [], []
    states = [ne.init_state(batch, hidden) for ne in m.neurons]
    prev = [None] * layers
    rec_sc = [ne.input_scale().detach() for ne in m.neurons] if rec_scale else None
    for t in range(T):
        a = h[:, t]
        for i, (fc, nrm, ne) in enumerate(zip(m.fcs, m.norms, m.neurons)):
            base = nrm(fc(a))
            if prev[i] is not None:
                r = m.recs[i](prev[i])
                if rec_sc is not None:
                    r = r * rec_sc[i]
                ff.append(float(base.abs().mean()))
                rc.append(float(r.abs().mean()))
                cur = base + r
            else:
                cur = base
            s, states[i] = ne.step(cur, states[i])
            prev[i] = s
            a = s
    ratios = [r / max(f, 1e-9) for f, r in zip(ff, rc)]
    return {
        "rec_gain": rec_gain, "rec_scale": rec_scale, "in_scale": in_scale,
        "feedforward_abs_mean": sum(ff) / len(ff),
        "recurrent_abs_mean": sum(rc) / len(rc),
        "ratio_mean": sum(ratios) / len(ratios),
        "ratio_median": sorted(ratios)[len(ratios) // 2],
        "n": len(ratios),
    }


def calibrate_for_init(rec_init: str, tau_hi: float = 40.0, in_scale: float = 5.0,
                       lo: float = 1.0, hi: float = 120.0) -> dict:
    """给定初始化方式，二分出配对增益。

    E/I 平衡把行和归零后，循环电流的均值分量消失、只剩涨落，达到同样
    幅值需要更大的增益——必须单独标定，不能沿用 default 的 12。
    """
    f = lambda g: drive_ratio(rec_gain=g, rec_scale=False, in_scale=in_scale,
                              tau_hi=tau_hi, rec_init=rec_init)["ratio_mean"]
    for _ in range(8):
        mid = (lo + hi) / 2
        if f(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return {"rec_init": rec_init, "gain_lo": lo, "gain_hi": hi,
            "ratio_lo": f(lo), "ratio_hi": f(hi)}


def calibrate_for_tau(tau_hi: float, in_scale: float = 5.0,
                      lo: float = 1.0, hi: float = 60.0) -> dict:
    """给定 τ 上界，二分出"循环与前馈同量级"所需的 rec_gain。

    **必须按 τ 范围重新标定。** 配对增益不是常数：τ 收窄会改变发放率，
    进而改变 s 的密度和循环电流的幅值。M1 用的 gain=12 是在 τ∈[1,40] 下
    测出来的，直接搬到 τ∈[1,5] 上没有依据。
    """
    f = lambda g: drive_ratio(rec_gain=g, rec_scale=False, in_scale=in_scale,
                              tau_hi=tau_hi)["ratio_mean"]
    for _ in range(7):
        mid = (lo + hi) / 2
        if f(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return {"tau_hi": tau_hi, "gain_lo": lo, "gain_hi": hi,
            "ratio_lo": f(lo), "ratio_hi": f(hi)}


def main() -> int:
    print("=" * 72)
    print("H1 核实：循环耦合强度的实测（含对 PLAN 原假设的纠正）")
    print("=" * 72)

    base = drive_ratio(rec_gain=1.0, rec_scale=False, in_scale=5.0)
    m = SHDNet(Cfg(recurrent=True, rec_scale=False, hidden=384, n_layers=2,
                   tau_mode="het", tau_lo=1.0, tau_hi=40.0, norm="none",
                   in_scale=5.0))
    st = scale_stats(m)

    print("\n[1] 逐神经元 1/(1-α)（定稿配方：层内 τ 1→40 帧，层间 ×2）")
    for k, v in st.items():
        print(f"  {k}: τ {v['tau_min']:.1f}–{v['tau_max']:.1f} 帧  "
              f"1/(1-α) {v['scale_min']:.2f}–{v['scale_max']:.2f}  "
              f"（层内跨 {v['span']:.1f}×）")

    print("\n[2] 循环 / 前馈 电流幅值比（rec_gain=1, in_scale=5, 默认初始化）")
    print(f"  {base['ratio_mean']:.4f}（中位 {base['ratio_median']:.4f}，"
          f"{base['n']} 个测量点）")
    print("  → 循环通路在初始化时几乎不存在。阶段二标着 recurrent=True 的实验，")
    print("    实际循环强度只有前馈的 3.7%，这是它'没有收益'最可能的直接原因。")

    print("\n[3] rec_gain 扫描：要让循环与前馈同量级需要多大增益")
    print(f"  {'in_scale':>9} {'rec_scale':>10} {'rec_gain':>9} {'循环/前馈':>10}")
    rows = []
    for in_scale in (1.0, 5.0):
        for rec_scale in (False, True):
            for g in (1.0, 2.0, 5.0, 10.0):
                r = drive_ratio(rec_gain=g, rec_scale=rec_scale, in_scale=in_scale)
                rows.append(r)
                print(f"  {in_scale:>9.1f} {str(rec_scale):>10} {g:>9.1f} "
                      f"{r['ratio_mean']:>10.4f}")

    # 比值对 rec_gain **不是**线性的：增益一大，发放率被推高、s 变稠密，
    # 循环电流自己把自己顶起来（正反馈）。所以必须实测标定，不能用外推。
    print("\n  注意：增益 >2 后比值已明显超线性（rec_scale=True 时 1→2 使比值涨 22×），"
          "\n        因为发放率上升使 s 变稠密、形成正反馈。故下面用实测二分而非外推。")

    def calibrate(in_scale: float, lo: float = 1.0, hi: float = 40.0) -> dict:
        """二分找到 ratio≈1 的 rec_gain。返回区间与两端实测值。"""
        f = lambda g: drive_ratio(rec_gain=g, rec_scale=False, in_scale=in_scale)["ratio_mean"]
        for _ in range(6):
            mid = (lo + hi) / 2
            if f(mid) < 1.0:
                lo = mid
            else:
                hi = mid
        return {"in_scale": in_scale, "gain_lo": lo, "gain_hi": hi,
                "ratio_lo": f(lo), "ratio_hi": f(hi)}

    target = {}
    print("\n  实测标定（二分，目标 ratio=1）：")
    for in_scale in (1.0, 5.0):
        c = calibrate(in_scale)
        target[str(in_scale)] = c
        print(f"    in_scale={in_scale}: rec_gain ∈ [{c['gain_lo']:.2f}, {c['gain_hi']:.2f}]"
              f"  →  ratio {c['ratio_lo']:.2f} / {c['ratio_hi']:.2f}")

    print("\n[4] 按 τ 范围标定配对增益（M2 的 τ 收窄实验要用）")
    print(f"  {'tau_hi':>8} {'gain_lo':>9} {'gain_hi':>9} {'ratio_lo':>10} {'ratio_hi':>10}")
    tau_cal = {}
    for th in (5.0, 10.0, 40.0):
        c = calibrate_for_tau(th)
        tau_cal[str(th)] = c
        print(f"  {th:>8.0f} {c['gain_lo']:>9.2f} {c['gain_hi']:>9.2f} "
              f"{c['ratio_lo']:>10.3f} {c['ratio_hi']:>10.3f}")

    print("\n[5] 按初始化方式标定配对增益（H2 的 E/I 平衡）")
    init_cal = {}
    for init in ("default", "ei_balanced"):
        c = calibrate_for_init(init)
        init_cal[init] = c
        print(f"  {init:12s} gain ∈ [{c['gain_lo']:.1f}, {c['gain_hi']:.1f}]  "
              f"ratio {c['ratio_lo']:.2f} / {c['ratio_hi']:.2f}")

    res = {
        "init_calibration": init_cal,
        "tau_calibration": tau_cal,
        "correction": (
            "原 H1 说'循环项被 (1-α) 缩放而文献不缩放'，是错的：W_in 与 W_rec "
            "同乘 (1-α)，比值与 (1-α) 无关。rec_scale 的实际作用是单方面放大"
            "循环耦合，不是恢复文献等价。"),
        "input_scale_stats": st,
        "baseline_ratio": base,
        "sweep": rows,
        "gain_for_parity": target,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
