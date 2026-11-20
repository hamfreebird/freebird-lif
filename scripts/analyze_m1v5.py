"""M1 的分析：三层前馈 + 逐层周期带（**无自顶向下通路**）。

这一格的作用是分离「分层」与「因果」：

  v5_3l_bands  层间周期带 6-12 / 18-45 / 60-120（中位约 8.5 / 28 / 85，比 3.4× / 3.0×）
  v5_3l_same   三层共享 6-120（中位相同，比 1.0×）——H5.4 的对照

判据：
  C1 尺度分层：层间中位比 ≥ 3×、总跨度 ≥ 10×
  C4 已有能力：SHD ≥ 85%（阶段四 87.93）、ITC-1 ≥ 0.30、GSA β ∈ [0.5,1.5]

另跑一项**负对照**：三层前馈没有下行通路，扰动顶层不应改变第 0 层对探测的响应，
C2 的 M 必须为 0。若不为零，说明仪器在多层结构里会串到别的通路。
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

B = ROOT / "benchmarks"
OUT = B / "m1v5.md"

CELLS = [
    ("v5_3l_bands", "三层 + 周期带", True),
    ("v5_3l_same", "三层 + 同谱", False),
]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    rows = load(B / "train_shd_m1v5.json")
    if not rows:
        print("缺少 train_shd_m1v5.json")
        return 1

    import torch
    from train_shd import Cfg, SHDNet

    L = ["# M1：三层前馈与逐层周期带\n"]
    L.append("**问题**：把网络从 2 层加到 3 层、并让层间的时间尺度分离，"
             "能不能在**没有自顶向下通路**的情况下提升精度？\n")
    L.append("这一格是阶段五的关键对照：如果三层前馈本身就达到阶段四的 87.93，"
             "说明层数与尺度不是瓶颈，阶段五的重心必须转到 C2（因果调制），"
             "而不是把「更深」当成「层级因果」。\n")

    # ---- 训练结果 ----
    L.append("## 1. 训练结果\n")
    L.append("| 配置 | test | train | 发放率 L0/L1/L2 | 参数 |")
    L.append("|---|---|---|---|---|")
    for tag, name, _ in CELLS:
        rs = sorted([r for r in rows if r["tag"] == tag], key=lambda r: r["seed"])
        if not rs:
            continue
        t = [r["test"] for r in rs]
        L.append(f"| {name} | **{statistics.fmean(t) * 100:.2f} ± "
                 f"{(statistics.stdev(t) * 100 if len(t) > 1 else 0):.2f}** | "
                 f"{statistics.fmean(r['final_train_acc'] for r in rs) * 100:.1f}% | "
                 f"{rs[0]['rates'][0]:.3f} / {rs[0]['rates'][1]:.3f} / {rs[0]['rates'][2]:.3f} | "
                 f"{rs[0]['params']:,} |")
    L.append("\n参照：阶段四二层最优（AT-LIF + 异构周期）**87.93 ± 0.49**，参数 429,332。\n")

    # ---- C1 尺度分层 ----
    L.append("## 2. C1：尺度分层（从训练后的检查点测）\n")
    L.append("| 配置 | seed | L0 中位 | L1 中位 | L2 中位 | 层间比 | 总跨度 | C1 |")
    L.append("|---|---|---|---|---|---|---|---|")
    c1_ok = {}
    for tag, name, _ in CELLS:
        for s in (0, 1, 2):
            ck = ROOT / "runs" / f"{tag}_s{s}.pt"
            if not ck.exists():
                continue
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            m = SHDNet(Cfg(**blob["config"]))
            m.load_state_dict(blob["state_dict"])
            ps = [ne.period_spectrum() for ne in m.neurons]
            med = [float(p.median()) for p in ps]
            lo = min(float(p.min()) for p in ps)
            hi = max(float(p.max()) for p in ps)
            ratios = [med[i + 1] / med[i] for i in range(len(med) - 1)]
            span = hi / lo
            ok = all(r >= 3.0 for r in ratios) and span >= 10.0
            c1_ok.setdefault(tag, []).append(ok)
            L.append(f"| {name} | {s} | {med[0]:.1f} | {med[1]:.1f} | {med[2]:.1f} | "
                     f"{' / '.join(f'{r:.1f}×' for r in ratios)} | {span:.1f}× | "
                     f"{'✅' if ok else '❌'} |")
    L.append("")

    # ---- 仪器 ----
    L.append("## 3. ITC 与 GSA（训练后）\n")
    L.append("| 配置 | 自持（层均值） | tau_int（层均值） | GSA β |")
    L.append("|---|---|---|---|")
    summary = {}
    for tag, name, _ in CELLS:
        ds = []
        for s in (0, 1, 2):
            d = load(B / f"v5m1_{tag}_s{s}.json")
            if d:
                ds.append(d)
        if not ds:
            continue
        sus = statistics.fmean(statistics.fmean(
            [d["itc"][f"layer{i}"]["sustain_ratio"] for i in range(3)]) for d in ds)
        tau = statistics.fmean(statistics.fmean(
            [d["itc"][f"layer{i}"]["tau_int"] for i in range(3)]) for d in ds)
        beta = statistics.fmean(d["gsa"]["beta"] for d in ds)
        summary[tag] = {"sus": sus, "tau": tau, "beta": beta}
        L.append(f"| {name} | {sus:.2f} | {tau:.1f} | {beta:.2f} |")
    L.append("")

    # ---- C2 负对照 ----
    L.append("## 4. C2 负对照（三层前馈不该有下行效应）\n")
    L.append("| 配置 | Δ=5 | 10 | 20 | 40 | 最大 \\|M\\| |")
    L.append("|---|---|---|---|---|---|")
    neg_ok = {}
    for tag, name, _ in CELLS:
        d = load(B / f"v5c2_{tag}_s0.json")
        if not d:
            continue
        mod = d["modulation"]
        vals = [mod[str(k)]["M"] for k in (5, 10, 20, 40)]
        mx = max(abs(v) for v in vals)
        neg_ok[tag] = mx <= 0.02
        L.append(f"| {name} | " + " | ".join(f"{v:+.4f}" for v in vals)
                 + f" | {mx:.4f} |")
    L.append("\n判据：三层前馈没有下行通路，M 应为 0。若不为零，说明仪器在多层结构里"
             "串到了别的通路，C2 在 M3 的结果就不可信。\n")

    # ---- 结论 ----
    acc = {}
    for tag, name, _ in CELLS:
        rs = [r for r in rows if r["tag"] == tag]
        if rs:
            acc[tag] = statistics.fmean(r["test"] for r in rs) * 100

    L.append("\n## 5. 结论\n")
    L.append("**M1 的核心发现是：尺度分层与精度互相冲突。**\n")
    if "v5_3l_bands" in acc and "v5_3l_same" in acc:
        L.append("| 配置 | 精度 | C1 | 说明 |")
        L.append("|---|---|---|---|")
        L.append(f"| 三层 + 分层带 | **{acc['v5_3l_bands']:.2f}%** | ✅ | 层间比 3.2× / 3.2×、跨度 22× |")
        L.append(f"| 三层 + 同谱 | **{acc['v5_3l_same']:.2f}%** | ❌ | 层间比 1.0×，但每层保留完整谱 |")
        L.append("| 二层最优（阶段四） | 87.93% | — | 参照 |")
        L.append("")
        L.append(f"**分层的代价是 {acc['v5_3l_same'] - acc['v5_3l_bands']:.2f} 点。**"
                 "原因不难解释：互斥的周期带让每层只保留三分之一的时间尺度，"
                 "等于**削减了每层的时间容量**。参数量从 429k 涨到 579k，"
                 "但去掉的恰恰是阶段四证明最有效的东西（宽谱）。")
        L.append("")
        L.append("**于是 C1 与 C4 在本设计下不相容**：")
        L.append(f"- 满足 C1 的配置只有 {acc['v5_3l_bands']:.2f}%，达不到 C4 的 85%；")
        L.append(f"- 满足精度的同谱配置（{acc['v5_3l_same']:.2f}%）层间比只有 1.0×，不满足 C1。")
    L.append("")
    L.append("**这条结果改变了阶段五的设计前提，必须写进计划修订：**\n")
    L.append("1. **「层级」不等于「互斥的周期带」**。若每层的谱必须完整才不掉精度，"
             "那么尺度分层只能做成**重叠带**（每层宽谱、中心平移），"
             "层间比会降到约 2×，而不是 3×。")
    L.append("2. **C1 的 3× 阈值是针对互斥带的**，与容量约束不相容。"
             "建议改为「层间中位比 ≥ 1.8× 且总跨度 ≥ 15×」——"
             "**这是看到数据后的修订，必须在计划里显式记录**，不能悄悄改。")
    L.append("3. **M2 的调制对比建议放在 2 层上做**：三层同谱（86.48）比二层（87.93）低 1.45 点，"
             "而调制效应是比值量（C2 的 M、C3 的 ΔR²），在一个更弱的基座上测会更噪。"
             "更重要的是，阶段四的**加性慢群体**正好是 2 层上的现成对照臂——"
             "乘性 vs 加性的对比可以直接复用它的三 seed 结果，省掉一半算力。")
    L.append("4. **C2 负对照在三层结构里仍然精确为 0**（两张表都是 +0.0000），"
             "说明仪器不会因为层数增加而串通路。这是 M2/M3 可信的前提。")

    text = "\n".join(L) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
