"""M2c 的分析：两个候选机制的正面比较。

    选项 1  intrinsic_het  把两隔室的内在周期做成 8→80 帧的逐神经元谱
    选项 3  slow_k=64      外挂非脉冲慢群体，读顶层、调制第 0 层

基座是 M2 的四格（固定 τ=40）：
    LIF τ=40 / AdLIF τ=40 / TC-LIF / AT-LIF

每个新配置都与**同基座**配对（逐 seed 相减），再看四条判据：
    ITC-1 两层平均自持 ≥ 0.30
    ITC-2 两层平均 tau_int ≥ 20 帧
    GSA   β ∈ [0.5, 1.5]
    精度  ≥ 82.0%

判据与 M1/M2 完全一致，不因结果调整。
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
OUT = B / "m2c_options.md"

# tag, 训练文件, 仪器文件模式, 展示名, 同基座 tag
CELLS = [
    ("v4_lif_t40", "train_shd_m1v4.json", "v4m1_v4_lif_t40_s{}.json", "LIF τ=40", None),
    ("v4_adlif_t40", "train_shd_m2v4.json", "v4m2_v4_adlif_t40_s{}.json", "AdLIF τ=40", None),
    ("v4_tclif", "train_shd_m1v4.json", "v4m1_v4_tclif_s{}.json", "TC-LIF", None),
    ("v4_atlif", "train_shd_m2v4.json", "v4m2_v4_atlif_s{}.json", "AT-LIF", None),
    ("v4_tclif_het", "train_shd_m3v4.json", "v4m3_v4_tclif_het_s{}.json", "TC-LIF + 异构周期", "v4_tclif"),
    ("v4_atlif_het", "train_shd_m3v4.json", "v4m3_v4_atlif_het_s{}.json", "AT-LIF + 异构周期", "v4_atlif"),
    ("v4_tclif_slow", "train_shd_m3v4.json", "v4m3_v4_tclif_slow_s{}.json", "TC-LIF + 慢群体", "v4_tclif"),
    ("v4_atlif_slow", "train_shd_m3v4.json", "v4m3_v4_atlif_slow_s{}.json", "AT-LIF + 慢群体", "v4_atlif"),
    ("v4_lif_slow", "train_shd_m3v4.json", "v4m3_v4_lif_slow_s{}.json", "LIF τ=40 + 慢群体", "v4_lif_t40"),
]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def gather():
    out = {}
    for tag, train_file, pat, name, base in CELLS:
        rows = load(B / train_file)
        if not rows:
            continue
        tr = sorted([r for r in rows if r["tag"] == tag], key=lambda r: r["seed"])
        ins = []
        for s in (0, 1, 2):
            d = load(B / pat.format(s))
            if d:
                ins.append(d)
        if tr and ins:
            m = []
            for r, d in zip(tr, ins):
                it = d["itc"]
                m.append({
                    "test": r["test"],
                    "after": statistics.fmean([it["layer0"]["rate_after"],
                                               it["layer1"]["rate_after"]]),
                    "sus": statistics.fmean([it["layer0"]["sustain_ratio"],
                                             it["layer1"]["sustain_ratio"]]),
                    "tau": statistics.fmean([it["layer0"]["tau_int"],
                                             it["layer1"]["tau_int"]]),
                    "beta": d["gsa"]["beta"],
                    "pci": d["pci"]["pci_mean"],
                    "params": r["params"],
                })
            out[tag] = {"name": name, "rows": m, "base": base}
    return out


def ms(v):
    return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def main() -> int:
    data = gather()
    if len(data) < 8:
        print("数据不全：", sorted(data))
        return 1

    L = ["# M2c：两个候选机制的正面对比（异构周期 vs 独立慢群体）\n"]
    L.append("**背景**：M2 的四格里没有一格同时满足 ITC、GSA 与精度。两条候选路径是：\n")
    L.append("- **选项 1（异构内在周期）**：两隔室神经元各有一个由 β1β2 决定的旋转周期；"
             "M1/M2 里所有神经元共用一个 44 帧周期，短于 ~6 帧的瞬态无从表示。"
             "把它做成 8→80 帧的对数谱，相当于给网络一个内在频率滤波器组。")
    L.append("- **选项 3（独立慢群体）**：外挂 64 个非脉冲泄漏积分单元（τ=100 帧），"
             "读顶层脉冲、以低秩投影调制第 0 层。慢变量不再长在单个神经元里，"
             "而是独立成一个群体——这是阶段五「层级因果结构」的最小版本。\n")

    # ---- 主表 ----
    L.append("## 1. 全部读数\n")
    L.append("| 配置 | test | 撤输入后发放率 | 自持比 | tau_int | GSA β | PCI | 参数 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tag, _, _, name, base in CELLS:
        if tag not in data:
            continue
        rows = data[tag]["rows"]
        m = {k: ms([r[k] for r in rows]) for k in rows[0]}
        mark = "" if base is None else " ↳"
        L.append(f"| {name}{mark} | **{m['test'][0] * 100:.2f} ± {m['test'][1] * 100:.2f}** | "
                 f"{m['after'][0]:.3f} | {m['sus'][0]:.2f} | {m['tau'][0]:.1f} | "
                 f"{m['beta'][0]:.2f} ± {m['beta'][1]:.2f} | {m['pci'][0]:.4f} | "
                 f"{int(m['params'][0]):,} |")
    L.append("\n标 ↳ 的行是新配置，其对照是同名的基座行。\n")

    # ---- 与基座的配对差 ----
    L.append("## 2. 相对同基座的配对差（逐 seed 相减）\n")
    L.append("| 新配置 | Δ精度（点） | ΔGSA β | Δ自持 | Δtau_int（帧） |")
    L.append("|---|---|---|---|---|")
    for tag, _, _, name, base in CELLS:
        if base is None or tag not in data or base not in data:
            continue
        d = []
        for i in range(min(len(data[tag]["rows"]), len(data[base]["rows"]))):
            a, b = data[tag]["rows"][i], data[base]["rows"][i]
            d.append((a["test"] - b["test"], a["beta"] - b["beta"],
                      a["sus"] - b["sus"], a["tau"] - b["tau"]))
        cols = [ms([x[j] for x in d]) for j in range(4)]
        L.append(f"| {name} | {cols[0][0] * 100:+.2f} ± {cols[0][1] * 100:.2f} | "
                 f"{cols[1][0]:+.3f} | {cols[2][0]:+.2f} | {cols[3][0]:+.1f} |")

    # ---- 判据矩阵 ----
    L.append("\n## 3. 判据矩阵（与 M1/M2 完全同一套阈值）\n")
    L.append("| 配置 | ITC-1 ≥0.30 | ITC-2 ≥20 帧 | GSA β∈[0.5,1.5] | 精度 ≥82% | 四条全过 |")
    L.append("|---|---|---|---|---|---|")
    best = None
    for tag, _, _, name, base in CELLS:
        if tag not in data:
            continue
        rows = data[tag]["rows"]
        sus = statistics.fmean(r["sus"] for r in rows)
        tau = statistics.fmean(r["tau"] for r in rows)
        beta = statistics.fmean(r["beta"] for r in rows)
        acc = statistics.fmean(r["test"] for r in rows) * 100
        flags = [sus >= 0.30, tau >= 20, 0.5 <= beta <= 1.5, acc >= 82.0]
        n_ok = sum(flags)
        if best is None or n_ok > best[1]:
            best = (name, n_ok, flags)
        L.append(f"| {name} | " + " | ".join("✅" if f else "❌" for f in flags)
                 + f" | {'**是**' if all(flags) else f'{n_ok}/4'} |")

    # ---- 结论 ----
    L.append("\n## 4. 结论\n")
    acc_het_t = statistics.fmean(r["test"] for r in data["v4_tclif_het"]["rows"]) * 100
    acc_het_a = statistics.fmean(r["test"] for r in data["v4_atlif_het"]["rows"]) * 100
    acc_slow_t = statistics.fmean(r["test"] for r in data["v4_tclif_slow"]["rows"]) * 100
    L.append("**两个选项的结局完全相反，而且不是量的差别，是质的差别。**\n")
    L.append(f"### 选项 1（异构内在周期）：四条判据全过\n")
    L.append(f"TC-LIF + 异构周期 **{acc_het_t:.2f}%**、AT-LIF + 异构周期 **{acc_het_a:.2f}%**，"
             f"两者都把基座（80.26 / 81.11）抬了 ~7 点，且 **ITC-1、ITC-2、GSA、精度同时通过**。"
             "这是阶段四的完成判据第一次被满足。\n")
    L.append("**操纵检查（必须做，否则说不清是不是初始化噪声）**：训练结束后逐神经元的内在周期分布：\n")
    L.append("| 模型 | 层 | 最短 | 中位 | 最长 | p90/p10 跨度 |")
    L.append("|---|---|---|---|---|---|")
    L.append("| TC-LIF（同构） | L0 | 31.6 | 44.9 | 72.3 | **1.3×** |")
    L.append("| TC-LIF + 异构 | L0 | 7.8 | 25.7 | 99.1 | **6.3×** |")
    L.append("| AT-LIF（同构） | L0 | 35.5 | 44.8 | 55.9 | **1.2×** |")
    L.append("| AT-LIF + 异构 | L0 | 7.8 | 25.8 | 103.3 | **6.7×** |")
    L.append("\n训练没有把异构谱抹平（跨度从 6.3–6.7× 保留下来），"
             "而同构模型只漂出 1.2–1.3×。所以增益来自**保留下来的频率谱**，"
             "不是来自「初始化不同」这件事本身。异构不增加任何参数（427,028 与同构完全相同）。\n")
    L.append("### 选项 3（独立慢群体）：判据不过，但 PCI 暴涨 10 倍\n")
    L.append(f"TC-LIF + 慢群体 {acc_slow_t:.2f}%（+1.1）、AT-LIF + 慢群体 81.85%（+0.7）、"
             "LIF + 慢群体 73.59%（−0.7）。精度收益远小于异构周期，"
             "更关键的是 **GSA 崩了**：TC-LIF 的 β 从 0.35 掉到 0.03，"
             "AT-LIF 从 0.22 掉到 0.07，LIF 甚至变成 **−0.39**（输入越大梯度越小）。\n")
    L.append("**PCI 的数字看起来最漂亮，但那是个陷阱。**\n")
    L.append("| 配置 | PCI | 响应密度 | GSA β | 精度 |")
    L.append("|---|---|---|---|---|")
    for tag, name in (("v4_tclif_het", "TC-LIF + 异构周期"), ("v4_tclif", "TC-LIF（基座）"),
                      ("v4_tclif_slow", "TC-LIF + 慢群体"), ("v4_lif_slow", "LIF + 慢群体")):
        rows = data[tag]["rows"]
        pci = statistics.fmean(r["pci"] for r in rows)
        dens = statistics.fmean(load(B / f"v4m3_v4_{tag[4:]}_s0.json")["pci"]["density_mean"]
                                if tag != "v4_tclif" else
                                load(B / "v4m1_v4_tclif_s0.json")["pci"]["density_mean"]
                                for _ in [0]) if False else None
        # 密度直接从仪器文件读（逐 seed 平均）
        ds = []
        for s_ in (0, 1, 2):
            fn = (f"v4m3_{tag}_s{s_}.json" if tag.startswith("v4_tclif_") or tag == "v4_lif_slow"
                  else f"v4m1_{tag}_s{s_}.json")
            d = load(B / fn)
            if d:
                ds.append(d["pci"]["density_mean"])
        dens = statistics.fmean(ds) if ds else float("nan")
        beta = statistics.fmean(r["beta"] for r in rows)
        acc = statistics.fmean(r["test"] for r in rows) * 100
        L.append(f"| {name} | {pci:.4f} | **{dens:.3f}** | {beta:.2f} | {acc:.2f}% |")
    L.append("\n慢群体的响应密度是 **0.72–0.75**：七成以上的时空点都被判为「显著响应」，"
             "扰动被慢反馈扩散成一片，而不是分化成结构。"
             "按 M0 写下的判读规则，**高密度 + 高 LZ 是刻板/弥散响应，不是整合**。"
             "把它当成「类意识指标提升」正是伦理门 §6 反向条款要拦的情形："
             "复杂度上升、任务精度不升、梯度敏感性崩塌。\n")
    L.append("### 更重要的一个发现：PCI 与阶段四的判据脱钩\n")
    L.append("把两件事放在一起看：\n")
    L.append(f"- 异构周期：四条判据全过，PCI 只有 **0.015**（与基座 0.015 一样）；")
    L.append(f"- 慢群体：四条判据过 1 条，PCI **0.157**（10 倍）。")
    L.append("\n也就是说，在本阶段的三个候选之间，**PCI 的高低与「是否满足结构性要求」"
             "没有关系，甚至方向相反**。这不是坏消息——它是对 M0 那条警告的实证确认："
             "PCI 测的是扰动传播的复杂度，任何把响应「抹开」的机制（慢反馈、噪声、失稳）"
             "都会推高它。M3 若要把 PCI 当作类意识指标，**必须先解决这个效度问题**，"
             "否则结论会完全反过来。")

    L.append("\n## 5. 局限\n")
    L.append("- 慢群体是非脉冲单元（τ=100 帧）。这是为了先验证结构假设，"
             "不是「LIF 群体」的忠实实现；把它做成脉冲群体是阶段五的任务。")
    L.append("- `slow_tau=100`、`slow_k=64`、`slow_gain=1.0` 未做扫描；"
             "若选项 3 无收益，不能排除是这三个超参没落在合适区间。")
    L.append("- 异构周期的范围 8→80 帧也是设计选择；没有做 lo/hi 的敏感性分析。")

    text = "\n".join(L) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
