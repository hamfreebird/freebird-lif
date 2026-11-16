"""M3 分析：五个 LIF 变体在 SHD 上的对比。

三个层次，缺一不可：

1. **均值 ± 标准差**（跨 seed）。文献的通行做法，但 n=3 时配对 t 检验只有
   2 个自由度，只能分辨 1.5 个百分点以上的差异。

2. **McNemar 逐样本检验**。测试集是固定的 2264 条，比较两个模型在同一批样本
   上的对错分歧，功力远高于 3 seed 的 t 检验。局限是它只回答"在这批样本上
   是否不同"，不包含 seed 方差。

3. **按说话人分解**。SHD 测试集 81% 的样本来自训练时未见的两位说话人
   （4、5 号），其余 10 位已见。如果某变体只在已见说话人上更好，那它改善的是
   拟合而不是泛化——这个区分对"该用哪个神经元"的结论至关重要。

用法:
    python scripts/analyze_m3.py
    python scripts/analyze_m3.py --json benchmarks/train_shd_m3.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

from _stats import T975, ci95, exact_mcnemar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "benchmarks" / "train_shd_m3.json"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "m3_variants.md"))
    args = ap.parse_args()

    rows = json.loads(Path(args.json).read_text("utf-8"))
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)

    d = build_data()
    spk = d["speaker_test"]
    y = d["y_test"].astype(np.int64)
    unseen = np.isin(spk, [4, 5])
    order = [v for v in ("lif", "alif", "adlif", "glif", "mclif") if v in by]
    if "lif" not in by:
        order = list(by)

    L = []
    L.append("# M3：五个 LIF 变体在 SHD 上的对比\n")
    L.append(f"配方：2×384 前馈，无归一化，层内异构 τ（1→40 帧，层间 ×2），"
             f"频道平移 ±35，120 epoch，全量 7156 条训练，取末轮权重。")
    L.append(f"测试集 {len(y)} 条，其中 {int(unseen.sum())} 条（"
             f"{unseen.mean():.1%}）来自训练时未见的两位说话人。\n")

    # ---- 主表 -------------------------------------------------------------
    L.append("## 1. 精度与成本\n")
    L.append("| 变体 | test 均值 | σ | 各 seed | Δ vs LIF | 浮点状态/神经元 | "
             "ops/步 | 总参数 |")
    L.append("|---|---|---|---|---|---|---|---|")
    acc = {v: [r["test"] for r in by[v]] for v in order}
    base = acc["lif"]
    for v in order:
        rs = by[v]
        m = statistics.fmean(acc[v])
        sd = statistics.stdev(acc[v]) if len(acc[v]) > 1 else float("nan")
        dm = statistics.fmean([a - b for a, b in zip(acc[v], base)])
        seeds = " / ".join(f"{a:.4f}" for a in acc[v])
        L.append(f"| **{v}** | **{m:.4f}** | {sd:.4f} | {seeds} | {dm:+.4f} | "
                 f"{rs[0].get('float_state', '—')} | "
                 f"{rs[0].get('op_count', 0):.0f} | {rs[0].get('params', 0):,} |")

    # ---- 配对差异的置信区间 ------------------------------------------------
    L.append("\n### 配对差异（vs LIF，同 seed 相减）\n")
    L.append("| 变体 | 各 seed 差值 | 均值 | 95% CI | 同号 |")
    L.append("|---|---|---|---|---|")
    for v in order:
        if v == "lif":
            continue
        diffs = [a - b for a, b in zip(acc[v], base)]
        n = len(diffs)
        m = statistics.fmean(diffs)
        if n > 1:
            se = statistics.stdev(diffs) / math.sqrt(n)
            half = T975.get(n - 1, 2.0) * se
            ci = f"[{m - half:+.4f}, {m + half:+.4f}]"
        else:
            ci = "n=1，无法估计"
        same = "是" if (all(x > 0 for x in diffs) or all(x < 0 for x in diffs)) else "否"
        L.append(f"| {v} | {' / '.join(f'{x:+.4f}' for x in diffs)} | {m:+.4f} | {ci} | {same} |")
    L.append(f"\n注：n={len(base)} 个 seed，配对 t 的临界值 t(0.975,{len(base)-1})="
             f"{T975.get(len(base)-1, 2.0)}。**置信区间跨过 0 的差异应视为未区分。**")

    # ---- McNemar ----------------------------------------------------------
    if not all("test_pred" in r for v in order for r in by[v]):
        L.append("\n## 2. McNemar 逐样本检验\n\n本次结果缺少逐样本预测，跳过。"
                 "（`test_pred` 是 2026-11-16 之后加入的字段。）")
        text = "\n".join(L) + "\n"
        Path(args.out).write_text(text, "utf-8")
        print(text)
        return 0
    L.append("\n## 2. McNemar 逐样本检验（vs LIF）\n")
    L.append("同一测试集上逐样本比较对错。b = 变体对/LIF 错，c = 变体错/LIF 对。")
    L.append("4 个变体各检验一次，Bonferroni 校正后 α = 0.05/4 = 0.0125。\n")
    L.append("| 变体 | seed | b | c | b−c | p（精确二项） | 显著(α=0.0125) |")
    L.append("|---|---|---|---|---|---|---|")
    sig_count = {}
    for v in order:
        if v == "lif":
            continue
        sig_count[v] = 0
        for k, rs in enumerate(by[v]):
            pr = np.array(rs["test_pred"], dtype=np.int64)
            pl = np.array(by["lif"][k]["test_pred"], dtype=np.int64)
            ok_a, ok_b = (pr == y), (pl == y)
            b = int((ok_a & ~ok_b).sum())
            c = int((~ok_a & ok_b).sum())
            p = exact_mcnemar(b, c)
            if p < 0.0125:
                sig_count[v] += 1
            L.append(f"| {v} | {rs['seed']} | {b} | {c} | {b - c:+d} | "
                     f"{p:.2e} | {'**是**' if p < 0.0125 else '否'} |")
    L.append("")
    for v, k in sig_count.items():
        L.append(f"- {v}：{k}/{len(by[v])} 个 seed 上相对 LIF 显著不同")

    # ---- 按说话人分解 ------------------------------------------------------
    L.append("\n## 3. 按说话人分解（未见 vs 已见）\n")
    L.append("未见 = 4、5 号说话人（训练集中完全不存在）；已见 = 其余 10 位。\n")
    L.append("| 变体 | 未见均值 | 已见均值 | 未见−已见 |")
    L.append("|---|---|---|---|")
    for v in order:
        un, sn = [], []
        for rs in by[v]:
            pr = np.array(rs["test_pred"], dtype=np.int64)
            ok = pr == y
            un.append(float(ok[unseen].mean()))
            sn.append(float(ok[~unseen].mean()))
        L.append(f"| {v} | {statistics.fmean(un):.4f} | {statistics.fmean(sn):.4f} | "
                 f"{statistics.fmean(un) - statistics.fmean(sn):+.4f} |")
    L.append("\n读法：**未见−已见 越接近 0，说明模型越不依赖说话人特异的特征。** "
             "如果某变体只在\"已见\"上提升，它改善的是拟合而不是泛化。")

    # ---- 训练过程 ----------------------------------------------------------
    L.append("\n## 4. 训练轨迹（末轮训练精度）\n")
    L.append("| 变体 | 各 seed 末轮 train | Δ vs LIF |")
    L.append("|---|---|---|")
    tr = {v: [r["final_train_acc"] for r in by[v]] for v in order}
    for v in order:
        dtr = statistics.fmean(tr[v]) - statistics.fmean(tr["lif"])
        L.append(f"| {v} | {' / '.join(f'{x:.4f}' for x in tr[v])} | {dtr:+.4f} |")

    # ---- 与弱基线筛查的对照 ------------------------------------------------
    L.append("\n## 5. 与弱基线（54%）筛查的对照\n")
    L.append("M1.5 曾在 M0 的冒烟配方（LayerNorm、τ=2、无增强、40 epoch、2×256）上"
             "做过一次 1 seed 筛查。当时基线只有 54.2%。\n")
    L.append("| 变体 | Δ @ 54% 基线 | Δ @ 83% 基线 | 结果 |")
    L.append("|---|---|---|---|")
    weak = {"adlif": 0.141, "glif": 0.081, "mclif": 0.042, "alif": 0.004}
    for v in order:
        if v == "lif":
            continue
        if v not in weak:
            continue
        diffs = [a - b for a, b in zip(acc[v], base)]
        d = statistics.fmean(diffs)
        w = weak[v]
        lo, hi = ci95(diffs)
        if lo <= 0.0 <= hi:
            verdict = "不可区分（CI 跨 0）"
        elif d > 0:
            verdict = f"存活，幅度缩小 {w / d:.1f}×"
        else:
            verdict = "**显著更差**"
        L.append(f"| {v} | {w:+.3f} | {d:+.4f} | {verdict} |")
    L.append("\n**四个变体里只有 AdLIF 在 83% 基线上表现出正向优势。** "
             "ALIF 始终不可区分；GLIF 的点估计翻负、总体仍不可区分，但按说话人拆开后"
             "是「已见 +1.65 / 未见 −1.85」；MC-LIF 显著更差。")
    L.append("\n机制：54% 的基线离「能不能训起来」很近，此时自适应电流、门控、"
             "树突分支的主要作用是**让优化更容易**，而不是提供更强的计算。"
             "基线抬到 83% 后优化不再是瓶颈，这些机制就只剩成本。")
    L.append("\n这直接说明：**在弱基线上做变体比较会系统性高估新机制的价值。** "
             "如果按 M1.5 的排序下结论，会得出「GLIF 和 MC-LIF 明显优于 LIF」的"
             "错误结论，而它们实际更差。")

    text = "\n".join(L) + "\n"
    Path(args.out).write_text(text, "utf-8")
    print(text)
    print(f"→ {args.out}")

    (ROOT / "benchmarks" / "m3_summary.json").write_text(json.dumps({
        "n_seeds": len(base),
        "n_test": int(len(y)),
        "n_unseen": int(unseen.sum()),
        "acc": acc,
        "mean": {v: statistics.fmean(a) for v, a in acc.items()},
        "std": {v: (statistics.stdev(a) if len(a) > 1 else None) for v, a in acc.items()},
        "train": tr,
    }, indent=2, ensure_ascii=False), "utf-8")
    return 0


def build_data():
    from freebird_lif.shd.data import ROOT as DROOT, build
    return build(DROOT / "data" / "shd", T=100)


if __name__ == "__main__":
    raise SystemExit(main())
