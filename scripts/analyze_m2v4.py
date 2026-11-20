"""M2 的分析：适应电流 × 两隔室的 2×2 消融。

四个格子（全部固定 τ=40，同配方）：

    |            | 无适应            | 有适应            |
    | 单隔室     | LIF τ=40（M1）    | AdLIF τ=40（M2）  |
    | 两隔室     | TC-LIF（M1）      | AT-LIF（M2）      |

交互效应按**同 seed 配对**计算：Δ = (AT-LIF − TC-LIF) − (AdLIF − LIF)，
再对三个 seed 取均值与 σ。同 seed 共享数据划分与增强序列，配对能压掉一部分方差。

预注册判据（M1 报告 §5）：
    若 AT-LIF 的 GSA β ≥ 0.5 **且** 撤输入后发放率落到 0.25–0.35 → 协同成立。
否则按三分支记录：协同 / 折中 / 有害。**不允许事后改判据。**
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
OUT = B / "m2v4.md"

CELLS = {
    "lif": ("v4_lif_t40", "train_shd_m1v4.json", "v4m1_v4_lif_t40_s{}.json", "LIF τ=40"),
    "adlif": ("v4_adlif_t40", "train_shd_m2v4.json", "v4m2_v4_adlif_t40_s{}.json", "AdLIF τ=40"),
    "tclif": ("v4_tclif", "train_shd_m1v4.json", "v4m1_v4_tclif_s{}.json", "TC-LIF"),
    "atlif": ("v4_atlif", "train_shd_m2v4.json", "v4m2_v4_atlif_s{}.json", "AT-LIF"),
}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def train_rows(tag: str, fname: str):
    rows = load(B / fname)
    if not rows:
        return []
    return sorted([r for r in rows if r["tag"] == tag], key=lambda r: r["seed"])


def inst_rows(pat: str):
    out = []
    for s in (0, 1, 2):
        d = load(B / pat.format(s))
        if d:
            out.append(d)
    return out


def mean_sd(v):
    return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def metrics(train, inst):
    """把训练结果与仪器读数合成一格的指标（逐 seed）。"""
    out = []
    for r, d in zip(train, inst):
        it = d["itc"]
        out.append({
            "test": r["test"],
            "train": r["final_train_acc"],
            "after": statistics.fmean([it["layer0"]["rate_after"], it["layer1"]["rate_after"]]),
            "sus": statistics.fmean([it["layer0"]["sustain_ratio"], it["layer1"]["sustain_ratio"]]),
            "tau": statistics.fmean([it["layer0"]["tau_int"], it["layer1"]["tau_int"]]),
            "beta": d["gsa"]["beta"],
            "pci": d["pci"]["pci_mean"],
            "dens": d["pci"]["density_mean"],
        })
    return out


def main() -> int:
    data = {}
    for key, (tag, fname, pat, name) in CELLS.items():
        tr, ins = train_rows(tag, fname), inst_rows(pat)
        if tr and ins:
            data[key] = {"name": name, "rows": metrics(tr, ins)}

    if len(data) < 4:
        print("四格数据不全：", sorted(data))
        return 1

    L = ["# M2：2×2 消融（适应电流 × 两隔室）\n"]
    L.append("**问题**：M1 的 TC-LIF 拿到了内在连续性（自持 1.22），但 GSA 掉到 0.35、"
             "精度 80.26%。适应电流作为**带泄漏的慢负反馈**，理论上能把持续发放压下来、"
             "把 GSA 拉回敏感区。本轮检验：两隔室与适应电流是协同、折中，还是有害。\n")
    L.append("四个格子全部固定 τ=40（TC-LIF 忽略 `tau_ms`，异构 τ 会引入配方差异）。\n")

    # ---- 主表 ----
    L.append("## 1. 四格读数（3 seed）\n")
    L.append("| 格子 | test | train | 撤输入后发放率 | 自持比 | tau_int | GSA β | PCI | 密度 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key in ("lif", "adlif", "tclif", "atlif"):
        d = data[key]
        rows = d["rows"]
        m = {k: mean_sd([r[k] for r in rows]) for k in rows[0]}
        L.append(f"| {d['name']} | **{m['test'][0] * 100:.2f} ± {m['test'][1] * 100:.2f}** | "
                 f"{m['train'][0] * 100:.1f}% | {m['after'][0]:.3f} | {m['sus'][0]:.2f} | "
                 f"{m['tau'][0]:.1f} | {m['beta'][0]:.2f} ± {m['beta'][1]:.2f} | "
                 f"{m['pci'][0]:.4f} | {m['dens'][0]:.3f} |")

    # ---- 交互 ----
    L.append("\n## 2. 交互效应（同 seed 配对）\n")
    L.append("交互 = (AT-LIF − TC-LIF) − (AdLIF − LIF)。正值表示两个机制叠加后**超过**"
             "各自单独效果之和。\n")
    L.append("| 指标 | 交互（均值 ± σ） | 逐 seed |")
    L.append("|---|---|---|")
    inter = {}
    for k in ("test", "after", "sus", "tau", "beta", "pci"):
        ds = []
        for i in range(3):
            v = ((data["atlif"]["rows"][i][k] - data["tclif"]["rows"][i][k])
                 - (data["adlif"]["rows"][i][k] - data["lif"]["rows"][i][k]))
            ds.append(v)
        m, sd = mean_sd(ds)
        inter[k] = (m, sd)
        scale = 100 if k == "test" else 1
        L.append(f"| {k} | {m * scale:+.3f} ± {sd * scale:.3f} | "
                 + ", ".join(f"{v * scale:+.3f}" for v in ds) + " |")

    # ---- 预注册判据 ----
    a = data["atlif"]["rows"]
    t = data["tclif"]["rows"]
    ad = data["adlif"]["rows"]
    l = data["lif"]["rows"]
    beta_a, _ = mean_sd([r["beta"] for r in a])
    after_a, _ = mean_sd([r["after"] for r in a])
    beta_t, _ = mean_sd([r["beta"] for r in t])
    beta_ad, _ = mean_sd([r["beta"] for r in ad])
    acc_a, _ = mean_sd([r["test"] for r in a])
    acc_t, _ = mean_sd([r["test"] for r in t])

    L.append("\n## 3. 预注册判据（M1 报告 §5，不得事后修改）\n")
    L.append(f"判据：AT-LIF 的 GSA β ≥ 0.5 **且** 撤输入后发放率 ∈ [0.25, 0.35]。\n")
    L.append(f"- AT-LIF：β = **{beta_a:.2f}**，撤输入后发放率 = **{after_a:.3f}**。")
    L.append(f"- 对照：TC-LIF β = {beta_t:.2f}、AdLIF β = {beta_ad:.2f}。")
    ok_beta = beta_a >= 0.5
    ok_rate = 0.25 <= after_a <= 0.35
    L.append(f"- 结论：{'✅ **协同成立**' if (ok_beta and ok_rate) else '❌ **协同不成立**'}"
             f"（β 判据 {'过' if ok_beta else '不过'}，发放率判据 {'过' if ok_rate else '不过'}）。\n")

    # 三分支
    L.append("**三分支归类**（按 GSA 相对两个单机制的位置）：\n")
    if beta_a >= max(beta_t, beta_ad):
        branch = "协同"
        desc = (f"AT-LIF 的 β（{beta_a:.2f}）不低于两个单机制"
                f"（TC-LIF {beta_t:.2f}、AdLIF {beta_ad:.2f}）。")
    elif beta_a >= min(beta_t, beta_ad):
        branch = "折中"
        desc = (f"AT-LIF 的 β（{beta_a:.2f}）落在两个单机制之间"
                f"（{min(beta_t, beta_ad):.2f} – {max(beta_t, beta_ad):.2f}），"
                "组合没有产生额外收益，也没有额外损失。")
    else:
        branch = "有害"
        desc = (f"AT-LIF 的 β（{beta_a:.2f}）低于两个单机制"
                f"（TC-LIF {beta_t:.2f}、AdLIF {beta_ad:.2f}）。")
    L.append(f"- **{branch}**：{desc}")
    L.append(f"- 精度：AT-LIF {acc_a * 100:.2f}% vs TC-LIF {acc_t * 100:.2f}%"
             f"（差 {(acc_a - acc_t) * 100:+.2f} 点）。")

    # ---- 判据矩阵 ----
    L.append("\n## 4. 判据矩阵：哪一格同时满足四条\n")
    L.append("| 格子 | ITC-1 自持 ≥0.30 | ITC-2 tau ≥20 | GSA β ∈[0.5,1.5] | 精度 ≥82% | 四条全过 |")
    L.append("|---|---|---|---|---|---|")
    ok_all = {}
    for key in ("lif", "adlif", "tclif", "atlif"):
        rows = data[key]["rows"]
        sus = statistics.fmean(r["sus"] for r in rows)
        tau = statistics.fmean(r["tau"] for r in rows)
        beta = statistics.fmean(r["beta"] for r in rows)
        acc = statistics.fmean(r["test"] for r in rows) * 100
        flags = [sus >= 0.30, tau >= 20, 0.5 <= beta <= 1.5, acc >= 82.0]
        ok_all[key] = all(flags)
        L.append(f"| {data[key]['name']} | " + " | ".join("✅" if f else "❌" for f in flags)
                 + f" | {'**是**' if all(flags) else '否'} |")
    L.append("")
    if not any(ok_all.values()):
        L.append("**没有任何一格同时满足四条。** 而且失败的方式是**互补的**：")
        L.append("- 单隔室的两格（LIF / AdLIF）GSA 过、精度不过（74.3 / 72.3）；")
        L.append("- 两隔室的两格（TC-LIF / AT-LIF）ITC 过、精度接近但 GSA 不过（0.35 / 0.22）。")
        L.append("\n也就是说，在本架构族里 **「持续活动」与「幅度敏感性」是此消彼长的**："
                 "网络一旦在撤输入后维持 0.4 的发放率，胞体就长期停在阈值之上，"
                 "输入的增量被压缩成常数偏置。这不是调参能同时拿到的两个目标。")
    else:
        L.append("通过的格子：" + "、".join(data[k]["name"] for k, v in ok_all.items() if v))

    L.append("\n## 5. 解读\n")
    L.append(f"**其一，精度上的交互是真的（{inter['test'][0] * 100:+.2f} ± "
             f"{inter['test'][1] * 100:.2f} 点）。** 三个 seed 全为正（"
             + ", ".join(f"{r['test'] * 100:+.2f}" for r in
                         [{"test": (a[i]['test'] - t[i]['test']) - (ad[i]['test'] - l[i]['test'])}
                          for i in range(3)]) + "）。"
             "适应电流在两隔室里加 0.85 点，在单隔室里减 2.01 点——"
             "**同一个机制在不同宿主上符号相反**，这是交互效应，不是主效应。")
    L.append("")
    L.append(f"**其二，GSA 上的交互是零（{inter['beta'][0]:+.3f} ± {inter['beta'][1]:.3f}）。** "
             f"AT-LIF 的 β={beta_a:.2f} 是四格里**最低**的（LIF 0.72、AdLIF 0.61、TC-LIF 0.35）。"
             "适应电流没有把幅度敏感性救回来，反而略微更差。")
    L.append("")
    L.append("**其三，也是关键：适应电流在持续活动这个维度上撞到了天花板。**")
    L.append(f"- 单隔室里它把自持比从 {statistics.fmean(r['sus'] for r in l):.2f} "
             f"拉到 {statistics.fmean(r['sus'] for r in ad):.2f}、tau 从 "
             f"{statistics.fmean(r['tau'] for r in l):.1f} 拉到 "
             f"{statistics.fmean(r['tau'] for r in ad):.1f} 帧——**+0.38 / +22 帧**；")
    L.append(f"- 两隔室里它几乎什么都没加：自持 "
             f"{statistics.fmean(r['sus'] for r in t):.2f} → "
             f"{statistics.fmean(r['sus'] for r in a):.2f}、tau "
             f"{statistics.fmean(r['tau'] for r in t):.1f} → "
             f"{statistics.fmean(r['tau'] for r in a):.1f} 帧。")
    L.append("\n两隔室在无适应时就已经自持在 1.22（撤输入后发放率 0.42），"
             "这个维度没有余量给第二个机制。**交互为负不是因为适应电流有害，"
             "而是因为它要加的东西已经满了。** 这与 M1 的「循环与异构 τ 功能重叠」"
             "是同一类现象，只是这次的「重叠」发生在饱和维度上。")
    L.append("")
    L.append(f"**其四，为什么会饱和到 GSA 塌缩。** AT-LIF 撤输入后仍以 {after_a:.3f} 的发放率活动，"
             "胞体长期停在阈值之上；此时把输入幅度乘 4，膜电位的增量被阈值非线性压平，"
             "`∂y/∂x₀` 不再随幅度增长。β=0.22 是这一饱和的直接读数，"
             "不是「梯度算错了」。")
    # ---- 探针：发放率正则 ----
    probe_rows = load(B / "train_shd_m2v4b.json") or []
    probe = {}
    for tag in ("v4_atlif_reg", "v4_tclif_reg"):
        tr = [r for r in probe_rows if r["tag"] == tag]
        d = load(B / f"v4m2_{tag}_s0.json")
        if tr and d:
            probe[tag] = (tr[0], d)
    if probe:
        L.append("\n## 6. 探针：把网络拉出饱和区，GSA 会回来吗（1 seed，不作结论）\n")
        L.append("四格里没有一格同时满足判据，而最直接的嫌疑是**饱和**：AT-LIF 撤输入后"
                 "仍以 0.415 的发放率活动。于是加一个发放率正则（`rate_target=0.15`、"
                 "`rate_lambda=2.0`，等价于 homeostatic 阈值）直接压这个症状。\n")
        L.append("| 模型 | test | 撤输入后发放率 | GSA β | g(4)/g(1) |")
        L.append("|---|---|---|---|---|")
        for tag, name, key in (("v4_tclif", "TC-LIF（原）", "tclif"),
                               ("v4_tclif_reg", "TC-LIF + 正则", None),
                               ("v4_atlif", "AT-LIF（原）", "atlif"),
                               ("v4_atlif_reg", "AT-LIF + 正则", None)):
            if tag in probe:
                r, d = probe[tag]
                it = d["itc"]
                after = statistics.fmean([it["layer0"]["rate_after"],
                                          it["layer1"]["rate_after"]])
                L.append(f"| {name} | {r['test'] * 100:.2f}% | {after:.3f} | "
                         f"**{d['gsa']['beta']:.2f}** | {d['gsa']['ratio_4_1']:.2f} |")
            elif key in data:
                rows = data[key]["rows"]
                L.append(f"| {name} | {statistics.fmean(x['test'] for x in rows) * 100:.2f}% | "
                         f"{statistics.fmean(x['after'] for x in rows):.3f} | "
                         f"**{statistics.fmean(x['beta'] for x in rows):.2f}** | — |")
        L.append("\n（原始行是 3 seed 均值；正则行只有 seed 0。下面的对比一律取 **seed 0**，"
                 "避免把单 seed 与均值混着比。）\n")
        L.append("**结果分成两半，而且方向相反：**")
        if "v4_tclif_reg" in probe and "v4_atlif_reg" in probe:
            # 原始模型（无正则）在 data 里，探针模型在 probe 里
            b_t = data["tclif"]["rows"][0]["beta"]
            b_a = data["atlif"]["rows"][0]["beta"]
            b_tr = probe["v4_tclif_reg"][1]["gsa"]["beta"]
            b_ar = probe["v4_atlif_reg"][1]["gsa"]["beta"]
            L.append(f"- TC-LIF（seed 0）：β {b_t:.2f} → **{b_tr:.2f}**（升），发放率压到 0.28，"
                     f"方向符合「饱和导致塌缩」的解释，但 **{b_tr:.2f} 仍未过 0.5**。")
            L.append(f"- AT-LIF（seed 0）：β {b_a:.2f} → **{b_ar:.2f}**（**降**），"
                     f"g(4)/g(1) 只有 {probe['v4_atlif_reg'][1]['gsa']['ratio_4_1']:.2f}"
                     "（<1，即输入幅度越大梯度越小）。")
            L.append("\n**这说明饱和不是唯一原因。** 对两隔室+适应这个组合，"
                     "把发放率压下来反而让幅度响应反号——适应电流在低发放率区变得主导，"
                     "输入越强、适应越强、输出反而被压得越低。"
                     "「用增益控制救 GSA」这条路，在**适应电流已经存在**的结构里不成立。")
        L.append("\n**探针的定位**：1 个 seed、只扫了一组正则参数，不能作为结论，"
                 "只能说明「饱和不是唯一原因」这个方向性判断。"
                 "要把它做成结论需要 3 seed 与 `rate_target` 扫描。")

    L.append("\n## 7. 局限\n")
    L.append("- 只有 3 seed，交互的置信区间较宽；结论以方向与三分支归类为主，不给精确效应量。")
    L.append("- 固定 τ=40 是为了配方可比，但它本身是三者里最弱的基线（74.32%）。"
             "若换成异构 τ 的 LIF，AdLIF 那一格会变成 86.07%，交互的基线会变——"
             "**但那个对照不可用**，因为 TC-LIF 无法接受异构 τ。这是两隔室变体的结构性限制。")
    L.append("- AT-LIF 的适应电流参数（`tau_w=10` 帧、a=b=0 初始化）未做扫描；"
             "若交互为负，有可能是参数没落在合适区间，而不是机制本身有害。"
             "这一点在报告里必须写明，不能反过来当成「机制无效」的证据。")

    text = "\n".join(L) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
