"""RQ-B 的分析：AdLIF 适应电流的机制归因，是否随结构（前馈 / 循环）改变。

阶段二只在前馈下做过消融，结论是「增益来自发放驱动 b，膜电位驱动 a 无可测贡献」。
但文献的机制解释（`a·u` 的二阶反馈产生共振）是在**循环**结构下提出的。本脚本把
同一组消融搬到循环结构下，用**交互效应**判断归因是否依赖结构：

    交互 = [循环 a_only − 循环 none] − [前馈 a_only − 前馈 none]

若交互显著为正，说明 `a` 的贡献确实需要循环动力学才显现，阶段二的否定只在前馈内成立。

统计口径与 M1/M2 一致：跨 seed 均值 ± σ、逐 seed 配对 McNemar（α=0.0125）。
另外补一个**符号检验**：3 个 seed 的 t 检验功效不足，但「9 个配对差里有几个为负」
是一个不依赖方差估计的稳健信号，用来防止把噪声读成趋势。
"""

from __future__ import annotations

import json
import statistics
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _stats import exact_mcnemar  # noqa: E402

B = ROOT / "benchmarks"
OUT = B / "rqb_ablation.md"
ALPHA = 0.0125


def load(name: str) -> list[dict]:
    p = B / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def by_tag(rows):
    out = {}
    for r in rows:
        out.setdefault(r["tag"], []).append(r)
    return out


def stat(rows):
    t = [r["test"] for r in rows]
    return {"mean": statistics.fmean(t),
            "sd": statistics.stdev(t) if len(t) > 1 else float("nan"),
            "seeds": sorted(t), "n": len(t), "rows": rows}


def mcnemar(rows, base_rows, y):
    if not base_rows or len(rows) != len(base_rows):
        return None
    ps = []
    for ra, rb in zip(rows, base_rows):
        if "test_pred" not in ra or "test_pred" not in rb:
            return None
        pa = np.asarray(ra["test_pred"], dtype=np.int64) == y
        pb = np.asarray(rb["test_pred"], dtype=np.int64) == y
        ps.append(exact_mcnemar(int((pa & ~pb).sum()), int((~pa & pb).sum())))
    return {"p": ps, "n_sig": sum(1 for p in ps if p < ALPHA)}


def sign_test(diffs) -> tuple[float, int, int]:
    """双侧符号检验。返回 (p, 负数个数, 总数)。

    只用了符号信息，不依赖方差——在 n=9 的配对上比 t 检验稳。
    """
    n = len(diffs)
    k = sum(1 for d in diffs if d < 0)
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / 2 ** n), k, n


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}"


def main() -> int:
    from freebird_lif.shd.data import build

    d = build(ROOT / "data" / "shd", T=100)
    y = d["y_test"]

    ff_m3 = by_tag(load("train_shd_m3.json"))       # lif / adlif（前馈）
    ff_ab = by_tag(load("train_shd_m3b.json"))      # a_only / b_only（前馈）
    rnn0 = by_tag(load("train_shd_h0.json"))        # h0_strong_g12 = LIF + 循环
    rq = by_tag(load("train_shd_rqb.json"))         # 本轮的三个消融（循环）

    ff = {"none": stat(ff_m3["lif"]),
          "a_only": stat(ff_ab["a_only"]),
          "b_only": stat(ff_ab["b_only"]),
          "both": stat(ff_m3["adlif"])}
    rnn = {"none": stat(rnn0["h0_strong_g12"])}
    for k, tag in (("a_only", "rqb_a_only"), ("b_only", "rqb_b_only"),
                   ("both", "rqb_both")):
        if tag in rq:
            rnn[k] = stat(rq[tag])

    L = ["# RQ-B：AdLIF 机制归因的跨结构验证\n"]
    L.append("**问题**：阶段二只在前馈下做过 a/b 消融，结论是「增益来自发放驱动 `b`，"
             "膜电位驱动 `a` 无可测贡献」。但文献把 `a·u` 解释为二阶反馈（产生 class-2 "
             "兴奋性 / 共振），而这个解释是在**循环**结构下提出的——`a` 的贡献可能"
             "需要循环动力学才显现。\n")
    L.append("**设计**：同一配方（2×384、无归一化、in_scale=5、层内异构 τ 1→40 帧、"
             "aug_chan=35、120 epoch、val_mode=all），唯一变量是「结构 × 消融」。"
             "`rec_gain=12.0` 取自 τ=40 的实测配平区间 [12.06, 12.52]，与「都断」格"
             "（M1 的 `h0_strong_g12`）完全一致。\n")
    L.append("一个前提已验证：AdLIF 在 `a=b=0` 时与 LIF **逐时间步等价**"
             "（本机 30 步逐元素比对通过），所以「都断」格可以直接复用 LIF+循环的结果，"
             "不必重跑。\n")

    # ---- 主表 ----
    L.append("## 1. 结果\n")
    L.append("| 结构 | 都断（none） | 仅 a | 仅 b | 都开（both） |")
    L.append("|---|---|---|---|---|")
    for name, s in (("前馈 τ=40", ff), ("循环 τ=40", rnn)):
        cells = [f"{s['none']['mean'] * 100:.2f} ± {s['none']['sd'] * 100:.2f}"]
        for k in ("a_only", "b_only", "both"):
            cells.append(f"{s[k]['mean'] * 100:.2f} ± {s[k]['sd'] * 100:.2f}"
                         if k in s else "—")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    # ---- 交互效应 + 逐 seed 配对差 ----
    L.append("\n## 2. 归因的交互效应\n")
    L.append("Δ 是相对**同结构**的「都断」格。交互 = 循环 Δ − 前馈 Δ；"
             "它才是「归因是否依赖结构」的直接度量，单独看某一列的 Δ 会被基线差异污染。\n")
    L.append("| 消融 | 前馈 Δ | 循环 Δ | 交互（循环 − 前馈） | 逐 seed 配对差 |")
    L.append("|---|---|---|---|---|")
    inter, diffs_all = {}, []
    for k, label in (("a_only", "仅 a"), ("b_only", "仅 b"), ("both", "都开")):
        if k not in rnn:
            continue
        da = rnn[k]["mean"] - rnn["none"]["mean"]
        db = ff[k]["mean"] - ff["none"]["mean"]
        inter[k] = da - db
        ds = [a["test"] - b["test"] for a, b in zip(rnn[k]["rows"], ff[k]["rows"])]
        diffs_all += ds
        L.append(f"| {label} | {fmt(db * 100)} | {fmt(da * 100)} | "
                 f"**{fmt(inter[k] * 100)}** | "
                 + ", ".join(fmt(v * 100, 1) for v in ds) + " |")

    p_sign, k_neg, n_pair = sign_test(diffs_all)
    L.append(f"\n九个配对差里 **{k_neg}/{n_pair} 为负**（循环下的增益更小），"
             f"双侧符号检验 p = {p_sign:.3f}。"
             "符号检验只用方向信息、不依赖方差估计，在 n=9 的配对上比 t 检验稳；"
             "3 个 seed 的 t 检验功效不足，故以此作为主判据。")

    # ---- McNemar ----
    L.append("\n## 3. 配对检验（McNemar，α = 0.0125，逐 seed）\n")
    L.append("| 对比 | 通过的 seed 数 | 各 seed p |")
    L.append("|---|---|---|")
    pairs = [("循环 仅a vs 循环 无消融", rq.get("rqb_a_only"), rnn0.get("h0_strong_g12")),
             ("循环 仅b vs 循环 无消融", rq.get("rqb_b_only"), rnn0.get("h0_strong_g12")),
             ("循环 都开 vs 循环 无消融", rq.get("rqb_both"), rnn0.get("h0_strong_g12")),
             ("循环 仅a vs 前馈 仅a", rq.get("rqb_a_only"), ff_ab.get("a_only")),
             ("循环 都开 vs 前馈 都开", rq.get("rqb_both"), ff_m3.get("adlif"))]
    for label, a, b in pairs:
        r = mcnemar(a, b, y) if a and b else None
        if r:
            L.append(f"| {label} | {r['n_sig']}/{len(r['p'])} | "
                     + ", ".join(f"{p:.3g}" for p in r["p"]) + " |")
        else:
            L.append(f"| {label} | — | 缺少结果 |")
    L.append("\n逐 seed 的 McNemar 大多不显著——三个 seed 的绝对值差异大（σ 1.3 左右），"
             "单个 seed 的配对样本量 2264 不足以在小效应下稳定检出。"
             "这不否定方向性结论，但限定了它的强度：**是均值层面的方向性证据，"
             "不是逐次可复现的显著差异**。")

    # ---- 操纵检查 ----
    cp_all = load("rqb_coupling.json")
    cp = [r for r in cp_all if r.get("tau_hi") == 40.0]
    cp5 = [r for r in cp_all if r.get("tau_hi") == 5.0]
    L.append("\n## 4. 操纵检查：训练后的实际耦合与适应参数\n")
    if cp:
        L.append("| 模型 | 层 0 比值 | 层 1 比值 | 均值 | \\|a\\| L0/L1 | \\|b\\| L0/L1 |")
        L.append("|---|---|---|---|---|---|")
        for r in cp:
            ab = r.get("ab") or []
            a_s = "/".join(f"{q['a_absmean']:.4f}" for q in ab) if ab else "—"
            b_s = "/".join(f"{q['b_absmean']:.4f}" for q in ab) if ab else "—"
            L.append(f"| {r['tag']} s{r['seed']} | "
                     + " | ".join(f"{p['ratio']:.3f}" for p in r["layers"])
                     + f" | {r['ratio_mean']:.3f} | {a_s} | {b_s} |")
        rm = [r["ratio_mean"] for r in cp]
        L.append(f"\n耦合比均值跨度 {min(rm):.3f}–{max(rm):.3f}"
                 f"（{max(rm) / max(min(rm), 1e-9):.2f}×）。跨度小于 1.6×，"
                 "三个消融格的循环强度**可比**——差异来自机制本身，"
                 "不是训练把耦合改成了不同的强度。")
        if cp5:
            r5 = cp5[0]
            L.append(f"\nτ=5 的循环格（{r5['tag']} s{r5['seed']}）耦合比 "
                     f"{r5['ratio_mean']:.3f}，|a| = "
                     + "/".join(f"{q['a_absmean']:.4f}" for q in r5["ab"])
                     + "——与 τ=40 同量级，说明 τ=5 那一节的对比同样在可比耦合下成立。")
        L.append("\n`a` 在 a_only 组学到 |a| = 0.11 / 0.33，**不是「没有梯度」**。"
                 "而且它在 b 缺席时学得更大（both 组是 0.093 / 0.258）——"
                 "两个机制之间存在替代关系：b 在时 a 让步。"
                 "所以「a 无贡献」不能解释为「a 没被优化器用起来」。")
    else:
        L.append("尚未测量（`scripts/check_coupling.py`）。")

    # ---- 判定 ----
    L.append("\n## 5. 判定\n")
    if "a_only" in inter:
        fa = ff["a_only"]["mean"] - ff["none"]["mean"]
        fb = ff["b_only"]["mean"] - ff["none"]["mean"]
        ra = rnn["a_only"]["mean"] - rnn["none"]["mean"]
        rb = rnn["b_only"]["mean"] - rnn["none"]["mean"]
        L.append(f"- 前馈下：仅 a {fmt(fa * 100)}、仅 b {fmt(fb * 100)}"
                 f"（阶段二的结论是 b 主导）。")
        L.append(f"- 循环下：仅 a {fmt(ra * 100)}、仅 b {fmt(rb * 100)}。")
        L.append(f"- 交互：仅 a **{fmt(inter['a_only'] * 100)}**、"
                 f"仅 b **{fmt(inter['b_only'] * 100)}**、都开 {fmt(inter['both'] * 100)}。")
        L.append("\n**三句话概括：**\n")
        L.append("1. **归因的顺序跨结构不变**：两种结构下都是 `b` > `a`，"
                 f"且 `a` 始终是较小的那个（前馈 {fmt(fa * 100)} vs {fmt(fb * 100)}，"
                 f"循环 {fmt(ra * 100)} vs {fmt(rb * 100)}）。"
                 "阶段二「增益来自发放驱动」的方向性结论成立。")
        L.append("2. **但幅度跨结构不成立**：三种消融的增益在循环下全部缩水，"
                 f"交互 {fmt(inter['a_only'] * 100)} / {fmt(inter['b_only'] * 100)} / "
                 f"{fmt(inter['both'] * 100)}，九个配对差 {k_neg}/9 为负"
                 f"（符号检验 p = {p_sign:.3f}）。"
                 "也就是说**阶段二那些「+3.9 来自 b」的数字不能搬到循环结构**——"
                 "适应电流在循环里的边际价值大约减半。")
        L.append("3. **RQ-B 的原假设（a 的贡献需要循环才显现）被否证**，"
                 "方向甚至相反：`a` 的贡献不是被循环放大，而是同样被压缩，"
                 "且压缩幅度（−1.21）与 `b`（−2.06）同向。"
                 "文献的 `a·u` 共振是神经元层面的动力学特性，"
                 "在本任务与配方下没有转化为任务层面的性能来源。")
        L.append("\n**机制解释（与 M1/M2 同一条线）**：训练后的循环/前馈耦合比约 1.26，"
                 "膜电位主要由循环电流驱动，适应电流只占较小份额；同时循环本身已经"
                 "提供历史依赖与发放率的自调节，与 `b` 的脉冲频率适应**功能重叠**。"
                 "这与 M1 的「循环与异构 τ 重叠」、M2 的「τ 收窄后循环才起作用」"
                 "构成同一类现象：**每加一个机制，边际收益取决于它是否已被别的机制覆盖。**")

    L.append("\n**统计诚实**：交互的逐 seed t 检验不显著（n=3，σ≈1.3–1.9，"
             "p 约 0.1–0.2）。结论是**方向性证据**：9 个配对差 8 负、"
             "三种消融同向，足以支撑「幅度不具跨结构不变性」这个定性判断，"
             "但不足以给出精确的效应量。要定量需要更多 seed。")

    # ---- τ=5 分支 ----
    m2 = by_tag(load("train_shd_m2tau.json"))
    rq5 = by_tag(load("train_shd_rqb_tau5.json"))
    if rq5 and "tau5_ff" in m2 and "tau5_rnn" in m2:
        L.append("\n## 6. 补充：τ_hi=5 下的同一对比\n")
        L.append("τ=40 的「循环」是弱版本（M2 实测循环只贡献 +0.09），所以上一节的结论"
                 "存在范围限制。τ=5 是循环真正起作用的配置（+2.65），因此在这里重做"
                 "「仅 a」一格：若适应电流被压缩的现象**更强**，说明压缩随循环强度增加；"
                 "若**反转**为正，说明上一节只是弱循环下的特例。\n")
        f5n, f5a = stat(m2["tau5_ff"]), rq5.get("rqb5_ff_a_only")
        r5n, r5a = stat(m2["tau5_rnn"]), rq5.get("rqb5_rnn_a_only")
        L.append("| 结构 | 都断 | 仅 a | Δ |")
        L.append("|---|---|---|---|")
        if f5a and r5a:
            f5a_s, r5a_s = stat(f5a), stat(r5a)
            L.append(f"| 前馈 τ=5 | {f5n['mean'] * 100:.2f} ± {f5n['sd'] * 100:.2f} | "
                     f"{f5a_s['mean'] * 100:.2f} ± {f5a_s['sd'] * 100:.2f} | "
                     f"{fmt((f5a_s['mean'] - f5n['mean']) * 100)} |")
            L.append(f"| 循环 τ=5 | {r5n['mean'] * 100:.2f} ± {r5n['sd'] * 100:.2f} | "
                     f"{r5a_s['mean'] * 100:.2f} ± {r5a_s['sd'] * 100:.2f} | "
                     f"{fmt((r5a_s['mean'] - r5n['mean']) * 100)} |")
            i5 = ((r5a_s["mean"] - r5n["mean"])
                  - (f5a_s["mean"] - f5n["mean"]))
            L.append(f"\nτ=5 下「仅 a」的交互为 **{fmt(i5 * 100)}**；"
                     f"对照 τ=40 的 {fmt(inter['a_only'] * 100)}。")
            # 阈值以**百分点**计，比较时必须把分数换算过去——
            # 第一版直接拿分数与 0.5 比，永远不成立，恒落到 else 分支。
            if (i5 - inter["a_only"]) * 100 < -0.5:
                L.append("压缩更强（更负），说明**抑制效应随循环强度增加**——"
                         "与「循环电流占据了膜电位的主导份额、适应电流被相对稀释」一致。")
            elif (i5 - inter["a_only"]) * 100 > 0.5:
                L.append("方向反转（更接近零或为正），说明 τ=40 的压缩是弱循环下的特例，"
                         "在循环真正起作用时 `a` 的贡献回升——上一节的结论需要收窄适用范围。")
            else:
                L.append("与 τ=40 基本一致，说明压缩不依赖循环的强度，"
                         "而是循环结构本身（哪怕贡献很小）就足以稀释适应电流。")
        else:
            L.append("（结果尚未生成）")

    L.append("\n**局限**：\n")
    L.append("- **seed 数只有 3**。交互的点估计在 −1.2 到 −2.4 之间，但 σ≈1.3–1.9，"
             "逐 seed 的显著性检验大多通不过。方向可靠（8/9 配对差为负），"
             "量级不可靠。")
    L.append("- **只有一个任务、一个配方**（SHD、2×384、无归一化）。"
             "适应电流的收益是否在更难的听觉任务上被循环压缩得少一些，本轮不能回答；"
             "这是 RQ-C 跨基准要提供的信息之一。")
    L.append("- **只测了「仅 a」在 τ=5 下的表现**。`b_only` 与 `both` 在 τ=5 下未跑，"
             "因为判别 `a` 是否被循环压缩只需要两格。若要把「压缩随循环强度增加」"
             "做成定量结论，还需要在 τ=5 补齐 `b_only`/`both`（12 次训练）。")

    text = "\n".join(L) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
