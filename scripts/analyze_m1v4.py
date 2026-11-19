"""M1 的分析：TC-LIF 单点能不能在 SHD 上同时保住精度、ITC 与 GSA。

三个配置：
    v4_tclif     TC-LIF，固定 τ（`tau_ms` 对它无效，见 variants/twocomp.py）
    v4_lif_t40   固定 τ=40 的 LIF——最接近 TC-LIF 静息周期（初始 44.4 帧）的单 τ 对照
    v4_lif_t2    固定 τ=2 的 LIF——短记忆下界

另列阶段三的**异构 τ LIF**（83.04 ± 0.63）作为「配方基准」，但它与两隔室不是同一配方，
只作参照，不用于归因——这是 M0 已经写明的口径。

判据（PLAN-v4 §1.3，M0 冻结）：
    ITC：两层平均自持比 ≥ 0.30；两层平均 tau_int ≥ 20 帧
    GSA：β ∈ [0.5, 1.5] 且 g(4)/g(1) ≥ 1.5
    精度：≥ 82.0%（相对前馈 83.04 损失 ≤ 1 点）
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
OUT = B / "m1v4.md"

TAGS = [("v4_tclif", "TC-LIF（固定 τ）"),
        ("v4_lif_t40", "LIF τ=40"),
        ("v4_lif_t2", "LIF τ=2")]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    rows = load(B / "train_shd_m1v4.json")
    if not rows:
        print("缺少 train_shd_m1v4.json")
        return 1
    by = {}
    for r in rows:
        by.setdefault(r["tag"], []).append(r)

    L = ["# M1：TC-LIF 单点（SHD，3 seed）\n"]
    L.append("**问题**：TC-LIF 能带来内在时间连续性（M0 已证：随机初始化下 tau_int 39–42 帧），"
             "但训练之后：(a) 精度要付多少代价？(b) ITC 还在吗？(c) GSA 保得住吗？\n")
    L.append("**配对基线**：固定 τ 的 LIF。因为 `tau_ms` 对 TC-LIF 无效，"
             "阶段三的异构 τ 配方在两隔室变体上不适用；拿 83.04 直接比会把"
             "「配方差异」错记成「神经元差异」。异构 τ 的 LIF 只作参照列。\n")

    # ---- 训练结果 ----
    L.append("## 1. 训练结果\n")
    L.append("| 配置 | test | train | 发放率 L0/L1 | 参数 | 单次耗时 |")
    L.append("|---|---|---|---|---|---|")
    for tag, name in TAGS:
        if tag not in by:
            continue
        rs = by[tag]
        t = [r["test"] for r in rs]
        L.append(f"| {name} | **{statistics.fmean(t) * 100:.2f} ± "
                 f"{(statistics.stdev(t) * 100 if len(t) > 1 else 0):.2f}** | "
                 f"{statistics.fmean(r['final_train_acc'] for r in rs) * 100:.1f}% | "
                 f"{rs[0]['rates'][0]:.3f} / {rs[0]['rates'][1]:.3f} | "
                 f"{rs[0]['params']:,} | {statistics.fmean(r['seconds'] for r in rs) / 60:.1f} min |")
    L.append("\n参照：阶段三异构 τ LIF **83.04 ± 0.63**（配方基准，不参与归因）；"
             "异构 τ AdLIF **86.07 ± 0.68**。\n")

    # ---- 仪器读数 ----
    inst = {}
    for tag, _ in TAGS:
        for s in (0, 1, 2):
            d = load(B / f"v4m1_{tag}_s{s}.json")
            if d:
                inst.setdefault(tag, []).append(d)

    if inst:
        L.append("## 2. 仪器读数（训练后权重，3 seed 各自测）\n")
        L.append("| 配置 | 自持比 L0/L1 | 平均自持 | tau_int L0/L1 | 平均 tau | GSA β | PCI |")
        L.append("|---|---|---|---|---|---|---|")
        summary = {}
        for tag, name in TAGS:
            if tag not in inst:
                continue
            ds = inst[tag]
            sus = [statistics.fmean([d["itc"]["layer0"]["sustain_ratio"],
                                     d["itc"]["layer1"]["sustain_ratio"]]) for d in ds]
            taus = [statistics.fmean([d["itc"]["layer0"]["tau_int"],
                                      d["itc"]["layer1"]["tau_int"]]) for d in ds]
            betas = [d["gsa"]["beta"] for d in ds]
            pcis = [d["pci"]["pci_mean"] for d in ds]
            summary[tag] = {"sus": statistics.fmean(sus),
                            "tau": statistics.fmean(taus),
                            "beta": statistics.fmean(betas),
                            "pci": statistics.fmean(pcis)}
            d0 = ds[0]
            L.append(f"| {name} | "
                     f"{d0['itc']['layer0']['sustain_ratio']:.2f} / "
                     f"{d0['itc']['layer1']['sustain_ratio']:.2f} | "
                     f"**{statistics.fmean(sus):.2f}** | "
                     f"{d0['itc']['layer0']['tau_int']:.1f} / "
                     f"{d0['itc']['layer1']['tau_int']:.1f} | "
                     f"**{statistics.fmean(taus):.1f}** | "
                     f"{statistics.fmean(betas):.2f} | {statistics.fmean(pcis):.3f} |")
        L.append("\n（逐层数值取 seed 0；均值列是 3 seed × 2 层的平均。）\n")

        # ---- 判据 ----
        L.append("## 3. 判据核对\n")
        L.append("| 判据 | 阈值 | " + " | ".join(n for _, n in TAGS if _ in summary) + " |")
        L.append("|---|---|" + "---|" * sum(1 for t, _ in TAGS if t in summary))
        def cell(v, ok):
            return ("✅ " if ok else "❌ ") + v
        L.append("| ITC-1 自持比（两层平均） | ≥ 0.30 | " + " | ".join(
            cell(f"{summary[t]['sus']:.2f}", summary[t]["sus"] >= 0.30)
            for t, _ in TAGS if t in summary) + " |")
        L.append("| ITC-2 内在时间尺度 | ≥ 20 帧 | " + " | ".join(
            cell(f"{summary[t]['tau']:.1f}", summary[t]["tau"] >= 20)
            for t, _ in TAGS if t in summary) + " |")
        L.append("| GSA β | ∈ [0.5, 1.5] | " + " | ".join(
            cell(f"{summary[t]['beta']:.2f}", 0.5 <= summary[t]["beta"] <= 1.5)
            for t, _ in TAGS if t in summary) + " |")
        L.append("| 精度 | ≥ 82.0% | " + " | ".join(
            cell(f"{statistics.fmean(r['test'] for r in by[t]) * 100:.2f}%",
                 statistics.fmean(r["test"] for r in by[t]) * 100 >= 82.0)
            for t, _ in TAGS if t in summary) + " |")

        # ---- 结论 ----
        L.append("\n## 4. 结论\n")
        if "v4_tclif" in summary:
            s = summary["v4_tclif"]
            acc = statistics.fmean(r["test"] for r in by["v4_tclif"]) * 100
            after = statistics.fmean(
                d["itc"]["layer0"]["rate_after"] for d in inst["v4_tclif"])
            beta0 = statistics.fmean(d["gsa"]["beta"] for d in inst["v4_tclif"])

            L.append(f"**一句话**：TC-LIF 拿到了「无输入仍活动」，代价是精度、梯度敏感性和"
                     f"一个接近护栏上限的发放率。\n")
            L.append("| 判据 | 结果 | 读数 |")
            L.append("|---|---|---|")
            itc1 = s["sus"] >= 0.30
            itc2 = s["tau"] >= 20
            gsa_ok = 0.5 <= s["beta"] <= 1.5
            acc_ok = acc >= 82.0
            L.append(f"| ITC-1 自持 | {'✅ 达成' if itc1 else '❌'} | 两层平均 {s['sus']:.2f}；"
                     f"撤输入后发放率 **{after:.3f}**（护栏上限 0.60） |")
            L.append(f"| ITC-2 时间尺度 | {'✅' if itc2 else '❌ 未达'} | 平均 {s['tau']:.1f} 帧"
                     f"（阈值 20；层 0 是 31.3/17.5/16.4，层 1 只有 13） |")
            L.append(f"| GSA | {'✅' if gsa_ok else '❌ 未达'} | β = {s['beta']:.2f}"
                     f"（阈值 [0.5, 1.5]） |")
            L.append(f"| 精度 | {'✅' if acc_ok else '❌ 未达'} | {acc:.2f}%（阈值 82.0） |")
            L.append("")
            L.append("**必须记录的一次自我修正（M0 → M1）**：M0 在**随机初始化**的网络上报出 "
                     "TC-LIF 的 β = 0.82（五个网络里最高），据此推翻了「TC-LIF 会失去 GSA」"
                     "这条前提。M1 训练后重测，β 掉到 **0.35**。"
                     "也就是说 M0 的结论只在初始化时成立——**训练把网络推到了饱和侧**"
                     f"（撤输入后发放率 {after:.3f}，接近 0.6 的护栏）。"
                     "H4.2 的原始动机（用适应电流的增益控制把 GSA 救回来）因此重新成立，"
                     "M2 的 2×2 有了明确的目标。")
            L.append("")
            if "v4_lif_t40" in summary:
                acc40 = statistics.fmean(r["test"] for r in by["v4_lif_t40"]) * 100
                L.append(f"**两条对照给出的信息**：")
                L.append(f"- **单一大 τ 不是两隔室的等价物**：LIF τ=40 只有 {acc40:.2f}%，"
                         f"比 TC-LIF 低 {acc - acc40:.1f} 点，且撤输入后立刻归零"
                         f"（自持比 {summary['v4_lif_t40']['sus']:.2f}、tau "
                         f"{summary['v4_lif_t40']['tau']:.1f} 帧）。"
                         "所以两隔室带来的不是「更长的记忆窗口」，"
                         "而是**有活动的持续状态**——这是 M3 匹配对照要正式检验的命题。")
                L.append(f"- **相对异构 τ LIF（83.04）仍差 {83.04 - acc:.1f} 点**："
                         "TC-LIF 没有逐神经元的时间尺度谱，缺的是「快」的那一端。"
                         "这不是两隔室的必然缺陷——把 β 做成逐神经元异构即可，"
                         "属于 M2 之后的一个明确候选（见 §5）。")
            L.append("")
            L.append("**PCI 的反向结果**：TC-LIF 的 PCI 0.015 是三者最低（LIF τ=2 是 0.036），"
                     "而它的响应密度最高（0.099）。高密度 + 低复杂度 = **刻板响应**："
                     "扰动被全局振荡吞掉，没有分化。"
                     "这直接否证了「有内在连续性 → 类意识指标更高」的朴素推断，"
                     "也说明 M3 必须把「匹配对照」和「密度」一起报，不能只看 PCI 数值。")

            L.append("\n## 5. 对 M2 的直接输入\n")
            L.append("1. **主攻 GSA**：TC-LIF 训练后 β=0.35、撤输入发放率 0.43，处在饱和侧。"
                     "适应电流的作用是压低持续发放——这正是 M2 2×2 要检验的机制。"
                     "预注册：若 AT-LIF 的 β ≥ 0.5 且发放率落到 0.25–0.35，则协同成立。")
            L.append("2. **精度缺口另有来源**：2.8 点来自缺「快时间尺度」。"
                     "在 M2 之后加一个 3 水平因子（β 同构 / β 异构），比在 M2 里混进来更干净。")
            L.append("3. **ITC-2 的阈值要复核**：层 0 的 tau_int 是 31.3/17.5/16.4，"
                     "层 1 只有 13。两层平均 17.6 卡在 20 的门槛下。"
                     "若 M2 的 AT-LIF 也在 15–20，就不是架构问题而是阈值定得不符合这个网络的时间尺度"
                     "——**但阈值不能因为结果不理想就改**，要先看 M2 的数据再决定。")

    text = "\n".join(L) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
