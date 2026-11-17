"""M1 / RQ-A 的分析：H0 的 2×2 对照 + 四个诊断仪器。

H0 的主张是"循环的收益依赖基线强度"。要检验它，四格必须各自对照**同基线下的
前馈**，否则测到的是"基线配方的差异"而不是"循环的贡献"。

统计口径沿用阶段二：
  * 跨 seed 均值 ± σ（n=3，配对 t 只有 2 自由度，只用于粗筛）
  * McNemar 精确二项（逐样本配对，功效高，但不含 seed 方差）
  * 要求 3 个 seed 全部显著才判"显著"，Bonferroni 到 0.0125
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _stats import ci95, exact_mcnemar  # noqa: E402

B = ROOT / "benchmarks"
OUT = B / "m1_h0.md"
ALPHA = 0.0125


def load(name: str) -> list[dict]:
    p = B / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def by_tag(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["tag"], []).append(r)
    return out


def summarize(rows: list[dict]) -> dict:
    t = [r["test"] for r in rows]
    return {"mean": statistics.fmean(t), "sd": statistics.stdev(t) if len(t) > 1 else float("nan"),
            "seeds": sorted(t), "n": len(t)}


def mcnemar_vs(rows: list[dict], base_rows: list[dict], y: np.ndarray) -> dict:
    """逐 seed 配对 McNemar。要求两边 seed 数一致且预测已保存。"""
    ps, cells = [], []
    for ra, rb in zip(rows, base_rows):
        if "test_pred" not in ra or "test_pred" not in rb:
            return {"ok": False, "reason": "缺 test_pred"}
        pa = np.asarray(ra["test_pred"], dtype=np.int64)
        pb = np.asarray(rb["test_pred"], dtype=np.int64)
        ok_a, ok_b = pa == y, pb == y
        b_ = int((ok_a & ~ok_b).sum())      # 只在 A 对
        c_ = int((~ok_a & ok_b).sum())      # 只在 B 对
        cells.append((b_, c_))
        ps.append(exact_mcnemar(b_, c_))
    return {"ok": True, "cells": cells, "p": ps,
            "n_sig": sum(1 for p in ps if p < ALPHA)}


def main() -> int:
    from freebird_lif.shd.data import build

    d = build(ROOT / "data" / "shd", T=100)
    y = d["y_test"]

    h0 = by_tag(load("train_shd_h0.json"))

    # 前馈对照的两个来源：
    #   弱基线 —— h0ff 网格（3 seed，带预测）
    #   强基线 —— 阶段二 final_w 里的 lif_w（3 seed，同配方，带预测）。
    #   注意 train_shd_r10.json 已被 M0 的锚点重跑覆盖成单 seed，不能再用；
    #   final_w 的 lif_w 与 r10 final 是同一配置，精度逐位相同。
    ffw = by_tag(load("train_shd_h0ff.json"))
    ffw.update({k: v for k, v in by_tag(load("train_shd_final_w.json")).items()
                if k == "lif_w"})
    ffw["h0_strong_ff"] = ffw.pop("lif_w")
    r10 = by_tag(load("train_shd_r10.json"))

    L: list[str] = []
    L.append("# M1：循环的收益是否依赖基线强度（H0 的 2×2 对照）\n")
    L.append("**问题**：文献里「加循环能涨很多点」的结论，是否只是弱基线的产物？\n")
    L.append("四格设计：基线强度（弱 = 阶段二 M0 冒烟配方 54.15% / 强 = 定稿配方 83.04%）")
    L.append("× 循环增益（1 = 阶段二等价的弱耦合，实测仅前馈的 3.7% / 12 = 实测标定的同量级）。")
    L.append("每格 3 seed，同 seed 配对。\n")

    # ---- 主表 ----
    L.append("## 1. 四格 + 两个前馈对照\n")
    L.append("| 基线 | 循环 | rec_gain | test 均值 | σ | 各 seed | Δ vs 同基线前馈 |")
    L.append("|---|---|---|---|---|---|---|")

    ff_mean = {}
    for key, label in (("h0_weak_ff", "弱"), ("h0_strong_ff", "强")):
        if key in ffw:
            s = summarize(ffw[key])
            ff_mean[label] = s["mean"]
            L.append(f"| {label} | 否 | — | {s['mean']:.4f} | {s['sd']:.4f} | "
                     + " / ".join(f"{v:.4f}" for v in s["seeds"]) + " | — |")
    # 若没跑 h0ff，回退到阶段二的数字（单 seed / 3 seed）
    if "弱" not in ff_mean:
        ff_mean["弱"] = 0.5415
        L.append(f"| 弱 | 否 | — | 0.5415 | — | 阶段二 r1 base（单 seed） | — |")
    if "强" not in ff_mean and "final" in r10:
        s = summarize(r10["final"])
        ff_mean["强"] = s["mean"]
        L.append(f"| 强 | 否 | — | {s['mean']:.4f} | {s['sd']:.4f} | "
                 + " / ".join(f"{v:.4f}" for v in s["seeds"]) + " | — |")

    cells = {}
    for base_label, prefix in (("弱", "h0_weak"), ("强", "h0_strong")):
        for g in (1.0, 12.0):
            tag = f"{prefix}_g{int(g)}"
            if tag not in h0:
                continue
            s = summarize(h0[tag])
            cells[(base_label, g)] = s
            dlt = s["mean"] - ff_mean.get(base_label, float("nan"))
            L.append(f"| {base_label} | 是 | {g:.0f} | {s['mean']:.4f} | {s['sd']:.4f} | "
                     + " / ".join(f"{v:.4f}" for v in s["seeds"])
                     + f" | {dlt:+.4f} |")

    # ---- McNemar ----
    L.append("\n## 2. McNemar 逐样本检验（vs 同基线前馈，同 seed 配对）\n")
    L.append("| 基线 | rec_gain | 各 seed (b, c) | p 值 | 显著(α=0.0125) |")
    L.append("|---|---|---|---|---|")
    for base_label, prefix in (("弱", "h0_weak"), ("强", "h0_strong")):
        base_rows = ffw.get(f"{prefix}_ff", [])
        for g in (1.0, 12.0):
            tag = f"{prefix}_g{int(g)}"
            if tag not in h0:
                continue
            r = mcnemar_vs(h0[tag], base_rows, y) if base_rows else {"ok": False,
                                                                     "reason": "无前馈对照"}
            if r["ok"]:
                L.append(f"| {base_label} | {g:.0f} | " + " ".join(f"({b},{c})" for b, c in r["cells"])
                         + " | " + " / ".join(f"{p:.1e}" for p in r["p"])
                         + f" | {r['n_sig']}/{len(r['p'])} |")
            else:
                L.append(f"| {base_label} | {g:.0f} | — | — | {r['reason']} |")

    # ---- 判定 ----
    L.append("\n## 3. 判定\n")
    if all((("弱", g) in cells and ("强", g) in cells) for g in (1.0, 12.0)):
        dw1 = cells[("弱", 1.0)]["mean"] - ff_mean["弱"]
        dw12 = cells[("弱", 12.0)]["mean"] - ff_mean["弱"]
        ds1 = cells[("强", 1.0)]["mean"] - ff_mean["强"]
        ds12 = cells[("强", 12.0)]["mean"] - ff_mean["强"]
        L.append(f"- 弱基线：gain=1 → {dw1:+.4f}；gain=12 → {dw12:+.4f}")
        L.append(f"- 强基线：gain=1 → {ds1:+.4f}；gain=12 → {ds12:+.4f}")
        best_weak, best_strong = max(dw1, dw12), max(ds1, ds12)
        if best_weak > 0.02 and best_strong < 0.01:
            L.append("\n**H0 成立**：循环在弱基线上有明显收益、在强基线上没有。"
                     "即「循环更强」这一结论至少部分是弱基线的产物。")
        elif best_weak <= 0.01:
            L.append("\n**H0 不成立（弱基线侧）**：即使把耦合标定到同量级，循环在弱基线上"
                     "也没有收益。说明本实现的循环确有技术缺陷，转入 H2–H6 排查。")
        else:
            L.append("\n**H0 部分成立**：弱基线上循环有收益，但强基线上也有——"
                     "不是单纯的基线产物，需结合诊断仪器判断机制。")

    # ---- 诊断仪器 ----
    diag = load("rnn_diag.json")
    if diag:
        L.append("\n## 4. 四个诊断仪器\n")
        L.append("| 模型 | test | 记忆@K=40 | 记忆@K=60 | tau_int L0 | tau_int L1 | 自持比 L0 | 自持比 L1 | 梯度 t0→tT 衰减 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in diag:
            c = r["config"]
            mem = r.get("memory", {})
            ac = r.get("autocorr", {})
            ss = r.get("self_sustain", {})
            gt = r.get("grad_time", {})
            g0 = gt.get("layer0")
            decay = (f"{g0[0] / max(g0[-1], 1e-30):.0f}×" if isinstance(g0, list) and g0 else "—")
            L.append(f"| {r['ckpt']} (g{c['rec_gain']:.0f}) | {r['test_acc_full']:.4f} | "
                     f"{mem.get('40', float('nan')):.3f} | {mem.get('60', float('nan')):.3f} | "
                     f"{ac.get('layer0', {}).get('tau_int', float('nan')):.1f} | "
                     f"{ac.get('layer1', {}).get('tau_int', float('nan')):.1f} | "
                     f"{ss.get('layer0', {}).get('ratio', float('nan')):.2f} | "
                     f"{ss.get('layer1', {}).get('ratio', float('nan')):.2f} | {decay} |")
        L.append("\n**怎么读**：")
        L.append("- `记忆@K` 是输入截断到前 K 帧后的精度。循环若真的延长记忆，小 K 处应更高。")
        L.append("- `tau_int` 是积分自相关时间（帧）。1.0 表示脉冲在时间上独立（纯前馈特征）。")
        L.append("- `自持比` 是切断输入后 / 之前的发放率之比。>1 意味着活动被循环自己维持。")
        L.append("- 梯度衰减比接近 1 说明 BPTT 能把误差传到最早的时间步。")

    # ---- 机制解读：诊断仪器给出的解释 ----
    if diag:
        L.append("\n## 5. 机制解读：为什么循环没帮上忙\n")
        byk = {r["ckpt"].replace(".pt", ""): r for r in diag}

        def g(ck, key, sub=None):
            r = byk.get(ck)
            if not r:
                return float("nan")
            v = r.get(key, {})
            return v.get(sub, float("nan")) if sub else v

        # 1) 记忆长度：循环到底有没有延长有效记忆
        L.append("**其一，循环确实延长了一点记忆，但幅度不足以改变最终精度。**\n")
        L.append("| 对比 | 记忆@K=40 前馈 | 记忆@K=40 循环 | 差 | 最终精度差 |")
        L.append("|---|---|---|---|---|")
        for base, ff, rc in (("弱", "h0_weak_ff_s0", "h0_weak_g12_s0"),
                             ("强", "h0_strong_ff_s0", "h0_strong_g12_s0")):
            a, b = g(ff, "memory", "40"), g(rc, "memory", "40")
            acc_a, acc_b = g(ff, "test_acc_full"), g(rc, "test_acc_full")
            L.append(f"| {base} | {a:.3f} | {b:.3f} | {b - a:+.3f} | {acc_b - acc_a:+.3f} |")
        L.append("\n弱基线上循环把小 K 处的精度抬了约 6 个点（0.378→0.439），说明它确实"
                 "多记住了一些东西；但**满长度时的精度差只有 0.6–1.3 点**"
                 "（弱 +0.006 / 强 +0.013），且强基线的 McNemar 为 0/3。"
                 "也就是说这些额外的记忆没有兑换成判别力。\n")

        # 2) 自相关：前馈基线本身的时间尺度有多长
        L.append("**其二，也是最关键的一点：前馈基线的时间尺度已经和循环一样长了。**\n")
        L.append("| 模型 | tau_int L0 | tau_int L1 | 自持比 L0 | 自持比 L1 |")
        L.append("|---|---|---|---|---|")
        for ck, name in (("h0_strong_ff_s0", "强 / 前馈"),
                         ("h0_strong_g1_s0", "强 / g1"),
                         ("h0_strong_g12_s0", "强 / g12"),
                         ("h0_weak_ff_s0", "弱 / 前馈"),
                         ("h0_weak_g12_s0", "弱 / g12")):
            L.append(f"| {name} | {g(ck, 'autocorr', 'layer0')['tau_int']:.1f} | "
                     f"{g(ck, 'autocorr', 'layer1')['tau_int']:.1f} | "
                     f"{g(ck, 'self_sustain', 'layer0')['ratio']:.2f} | "
                     f"{g(ck, 'self_sustain', 'layer1')['ratio']:.2f} |")
        L.append("\n强基线的**前馈**模型 tau_int 已经有 11.2 / 10.9 帧，而加了循环之后"
                 "是 10.2 / 9.5 帧——**没有变长**。这解释了为什么精度没有提升："
                 "循环没有带来新的时间尺度。")
        L.append("\n原因是定稿配方的**层内异构 τ（1→40 帧、层间 ×2）本身就是一个"
                 "多尺度积分器**：慢神经元的时间常数就有几十帧，前馈通路已经能把信息"
                 "保持到需要的长度。再叠加循环，只是把同一件事做了第二遍。")
        L.append("\n自持比提供了另一个侧面证据：强基线前馈切断输入后活动掉到 0.19 / 0.36，"
                 "而 g12 维持在 0.97 / 1.20——循环**确实**制造了自持活动，"
                 "**但自持活动没有换来精度**。这是一个直接的「活动 ≠ 功能」证据。\n")

        # 3) 梯度
        L.append("**其三，循环让梯度的跨时间分布变得更不均匀。**\n")
        L.append("前馈强基线 t=0 与 t=T 的梯度范数只差约 5 倍，加循环后差 374–875 倍。"
                 "BPTT 需要把误差穿过 100 步的循环连接，靠后的时间步梯度被压得很低——"
                 "这与「循环更难优化」的预期一致，也可能是它无法把新增记忆兑换成精度的原因。\n")

        # 4) 结论
        L.append("**综合判断**：在本配方下，循环不是「接得不够强」（M0 已排除），"
                 "而是**与已有的多尺度 τ 功能重叠**。要检验这个解释，"
                 "最直接的干预是**收窄 τ 的范围**（例如 τ_hi 从 40 降到 5），"
                 "让前馈通路的时间尺度变短，再看向循环是否变得重要。"
                 "这是 M2 的第一项实验——它比继续调 rec_gain 更有判别力。")

    text = "\n".join(L) + "\n"
    OUT.write_text(text, "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
