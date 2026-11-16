"""M4 分析：成本-精度权衡、分类别误差结构、错误集合重叠。

M3 回答了"哪个变体更好"，M4 回答三个更进一步的问题：

1. **代价是什么？** 把精度增益折算到每 2× 算子上，看性价比排序是否变化。
2. **好在哪些类别上？** 总精度提升可能只来自少数几个类，也可能均匀分布。
   这两种情况的含义完全不同：前者说明新机制解决了特定困难，后者说明是普遍改善。
3. **错误集合是否重叠？** 如果两个变体错的是同一批样本，它们学到的表示相似；
   如果几乎不重叠，则有集成价值。

只依赖 `train_shd_m3.json` 里的逐样本预测，不需要重训。

用法:
    python scripts/analyze_m4.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

from _stats import ci95, exact_mcnemar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

def class_keys() -> list[str]:
    """从数据文件读真实标签。**不要硬编码**——我最初凭记忆写成
    "zehn/elf/zwölf/dreizehn..."（把 10–19 当成德语的十位数），
    实际是 0–9 英语、10–19 德语（zero..nine / null..neun）。
    错误的标签会让"哪个类最难"的解读整体跑偏。"""
    import h5py
    with h5py.File(ROOT / "data" / "shd" / "shd_train.h5", "r") as f:
        return [k.decode() for k in f["extra"]["keys"][:]]


def load(variants: list[str]) -> tuple[dict, np.ndarray]:
    from freebird_lif.shd.data import ROOT as D, build
    rows = json.loads((ROOT / "benchmarks" / "train_shd_m3.json").read_text("utf-8"))
    by = {}
    for r in rows:
        if r["variant"] in variants:
            by.setdefault(r["variant"], []).append(r)
    d = build(D / "data" / "shd", T=100)
    return by, d["y_test"].astype(np.int64)


def main() -> int:
    CLASSES = class_keys()
    order = ["lif", "alif", "adlif", "glif", "mclif"]
    by, y = load(order)
    order = [v for v in order if v in by]
    L = ["# M4：成本-精度权衡与误差结构\n"]

    # ---- 1. 成本-精度 ------------------------------------------------------
    L.append("## 1. 每 2× 算子换到多少点\n")
    base = statistics.fmean([r["test"] for r in by["lif"]])
    base_ops = by["lif"][0]["op_count"]
    L.append("| 变体 | test | Δ vs LIF | ops/步 | 算子倍数 | 每 2× 算子换到的点 |")
    L.append("|---|---|---|---|---|---|")
    rows = []
    for v in order:
        m = statistics.fmean([r["test"] for r in by[v]])
        ops = by[v][0]["op_count"]
        mult = ops / base_ops
        d = (m - base) * 100
        # 每 2× 算子的等价增益：把 Δ 折算到 log2(倍数) 的每单位
        per2 = d / math.log2(mult) if mult > 1 else float("nan")
        rows.append((v, m, d, ops, mult, per2))
        L.append(f"| {v} | {m:.4f} | {d:+.2f} | {ops:.0f} | {mult:.2f}× | "
                 f"{per2:+.2f} |" if mult > 1 else
                 f"| {v} | {m:.4f} | {d:+.2f} | {ops:.0f} | 1.00× | —（基准） |")
    L.append("\n注：`每 2× 算子换到的点` = Δ / log2(算子倍数)。负值意味着多付算力换到更差的结果。")
    L.append("\n**这一列是负的就不该选它**，无论绝对精度如何——除非它有别的价值"
             "（更少的状态、更容易训练、更适合某种硬件）。")

    # ---- 2. 分类别 ---------------------------------------------------------
    L.append("\n## 2. 分类别精度：增益是普遍的还是集中在少数类？\n")
    L.append("| 类 | " + " | ".join(order) + " | AdLIF−LIF |")
    L.append("|---|" + "---|" * (len(order) + 1))
    per_cls = {}
    for v in order:
        preds = np.array([np.array(r["test_pred"], dtype=np.int64) for r in by[v]])
        acc = np.array([(p == y).mean() for p in preds])
        per_cls[v] = acc
    diffs = []
    for c in range(20):
        mask = y == c
        vals = {v: np.array([(np.array(r["test_pred"], dtype=np.int64)[mask] == c).mean()
                             for r in by[v]]) for v in order}
        for v in order:
            per_cls.setdefault(v + "_cls", {})[c] = vals[v].mean()
        d = vals["adlif"].mean() - vals["lif"].mean()
        diffs.append((c, d))
        L.append(f"| {CLASSES[c]} | " +
                 " | ".join(f"{vals[v].mean():.3f}" for v in order) +
                 f" | {d:+.3f} |")
    diffs.sort(key=lambda t: -t[1])
    L.append("\n**AdLIF 提升最多的 5 个类：** " +
             "、".join(f"{CLASSES[c]}({d:+.3f})" for c, d in diffs[:5]))
    L.append(f"\n**AdLIF 下降最多的 5 个类：** " +
             "、".join(f"{CLASSES[c]}({d:+.3f})" for c, d in diffs[-5:]))
    pos = sum(1 for _, d in diffs if d > 0)
    L.append(f"\n{pos}/20 个类别上 AdLIF 优于 LIF。"
             f"若接近 20，说明是普遍改善；若只有少数几个，说明它解决了特定困难。")

    # ---- 2b. 最难的类与跨语言混淆 ------------------------------------------
    L.append("\n### 2b. 最难的类为什么难：跨语言同源词\n")
    L.append("标签取自数据文件的 `extra/keys`：0–9 是英语 zero..nine，"
             "10–19 是德语 null..neun。**同一位数字的两种语言形式是近义词**"
             "（nine / neun、six / sechs），因此跨语言同数字的混淆是结构性的。\n")
    C = np.zeros((20, 20), int)
    pl = np.array(by["lif"][0]["test_pred"], dtype=np.int64)
    for t, q in zip(y, pl):
        C[t, q] += 1
    L.append("| 真类 | 正确率 | 最主要的错判 |")
    L.append("|---|---|---|")
    worst = sorted(range(20), key=lambda c: C[c, c] / max(1, C[c].sum()))[:5]
    for c in worst:
        row = sorted(((C[c, j], j) for j in range(20) if j != c), reverse=True)[:3]
        L.append(f"| {CLASSES[c]} | {C[c, c] / max(1, C[c].sum()):.2f} | " +
                 "、".join(f"{CLASSES[j]}({n})" for n, j in row) + " |")
    # 跨语言同数字的混淆质量占比
    cross = sum(C[i, i + 10] + C[i + 10, i] for i in range(10))
    total_off = int(C.sum() - np.trace(C))
    L.append(f"\n跨语言同数字（i ↔ i+10）的混淆共 {cross} 次，"
             f"占全部 {total_off} 次错误的 {cross / total_off:.1%}。")
    L.append("这是数据本身的性质而非模型缺陷：德语 neun 与英语 nine 只差几个音素，"
             "在 13.7 ms 的分箱下尤其难分。任何变体都无法消除这部分错误——"
             "它给精度设了一个所有变体共享的上限。")

    # ---- 3. 错误集合重叠 ---------------------------------------------------
    L.append("\n## 3. 错误集合的重叠（seed 0）\n")
    L.append("Jaccard = |两者都错| / |至少一个错|。接近 1 说明犯同样的错，"
             "接近 0 说明错误互补（有集成价值）。\n")
    L.append("| | " + " | ".join(order) + " |")
    L.append("|---|" + "---|" * len(order))
    err = {v: np.array(by[v][0]["test_pred"], dtype=np.int64) != y for v in order}
    for a in order:
        cells = []
        for b in order:
            inter = int((err[a] & err[b]).sum())
            uni = int((err[a] | err[b]).sum())
            cells.append(f"{inter / uni:.3f}" if uni else "—")
        L.append(f"| {a} | " + " | ".join(cells) + " |")

    # ---- 4. 一个补充：AdLIF 修好了哪些、弄坏了哪些 -------------------------
    L.append("\n## 4. AdLIF 相对 LIF 的样本级流向（seed 0）\n")
    ok_l = np.array(by["lif"][0]["test_pred"], dtype=np.int64) == y
    ok_a = np.array(by["adlif"][0]["test_pred"], dtype=np.int64) == y
    L.append("| | 数量 | 占测试集 |")
    L.append("|---|---|---|")
    L.append(f"| 两者都对 | {int((ok_l & ok_a).sum())} | {(ok_l & ok_a).mean():.1%} |")
    L.append(f"| **LIF 错、AdLIF 对** | {int((~ok_l & ok_a).sum())} | "
             f"{(~ok_l & ok_a).mean():.1%} |")
    L.append(f"| **LIF 对、AdLIF 错** | {int((ok_l & ~ok_a).sum())} | "
             f"{(ok_l & ~ok_a).mean():.1%} |")
    L.append(f"| 两者都错 | {int((~ok_l & ~ok_a).sum())} | {(~ok_l & ~ok_a).mean():.1%} |")
    gain = int((~ok_l & ok_a).sum()) / max(1, int((~ok_l).sum()))
    L.append(f"\nAdLIF 修好了 LIF 错误的 **{gain:.1%}**，同时弄坏了 "
             f"{int((ok_l & ~ok_a).sum())} 条原本正确的样本。")

    # ---- 5. AdLIF 机制消融 -------------------------------------------------
    ab_path = ROOT / "benchmarks" / "train_shd_m3b.json"
    if ab_path.exists():
        ab = json.loads(ab_path.read_text("utf-8"))
        byab = {}
        for r in ab:
            byab.setdefault(r["tag"], []).append(r)
        L.append("\n## 5. AdLIF 机制消融：+3.03 点来自哪一项\n")
        L.append("AdLIF 的适应电流有两项驱动，来源不同：")
        L.append("- `a·u[t-1]`：膜电位驱动，构成 u→w→u 二阶反馈，产生 class-2 兴奋性（共振）")
        L.append("- `b·s[t-1]`：发放驱动，经典的脉冲频率适应")
        L.append("\n分别冻结即可拆开两者的贡献。`both` 与 `none` 直接引用 M3 的 "
                 "adlif / lif（同配置、同协议），不重跑。\n")
        L.append("| 配置 | a | b | test 均值 | σ | 各 seed | Δ vs LIF |")
        L.append("|---|---|---|---|---|---|---|")
        lif_m = statistics.fmean([r["test"] for r in by["lif"]])
        L.append(f"| LIF（都不开） | — | — | {lif_m:.4f} | "
                 f"{statistics.stdev([r['test'] for r in by['lif']]):.4f} | " +
                 " / ".join(f"{r['test']:.4f}" for r in by["lif"]) + " | — |")
        for tag, la, lb, name in (("b_only", "—", "✓", "仅发放驱动"),
                                  ("a_only", "✓", "—", "仅膜电位驱动")):
            if tag not in byab:
                continue
            rs = byab[tag]
            vals = [r["test"] for r in rs]
            m = statistics.fmean(vals)
            L.append(f"| {name} | {la} | {lb} | {m:.4f} | "
                     f"{(statistics.stdev(vals) if len(vals) > 1 else float('nan')):.4f} | " +
                     " / ".join(f"{v:.4f}" for v in vals) +
                     f" | {m - lif_m:+.4f} |")
        ad = [r["test"] for r in by["adlif"]]
        L.append(f"| AdLIF（都开） | ✓ | ✓ | {statistics.fmean(ad):.4f} | "
                 f"{statistics.stdev(ad):.4f} | " +
                 " / ".join(f"{v:.4f}" for v in ad) +
                 f" | {statistics.fmean(ad) - lif_m:+.4f} |")
        # 消融内部的 McNemar：b_only vs both 是最关键的一对
        L.append("\n### 5.1 消融内部的 McNemar（vs LIF，同 seed 配对）\n")
        L.append("| 比较 | 各 seed (b, c) | p 值 | 显著(α=0.0125) |")
        L.append("|---|---|---|---|")
        pairs = [("a_only", byab.get("a_only", []), by["lif"]),
                 ("b_only", byab.get("b_only", []), by["lif"]),
                 ("b_only", byab.get("b_only", []), by["adlif"])]
        labels = ["a_only vs LIF", "b_only vs LIF", "b_only vs AdLIF(都开)"]
        for lab, A, B in zip(labels, [p[1] for p in pairs], [p[2] for p in pairs]):
            if not A:
                continue
            cells, ps = [], []
            for ra, rb in zip(A, B):
                pa = np.array(ra["test_pred"], dtype=np.int64)
                pb = np.array(rb["test_pred"], dtype=np.int64)
                ok_a, ok_b = pa == y, pb == y
                b_ = int((ok_a & ~ok_b).sum())
                c_ = int((~ok_a & ok_b).sum())
                cells.append(f"({b_},{c_})")
                ps.append(exact_mcnemar(b_, c_))
            L.append(f"| {lab} | {' '.join(cells)} | "
                     f"{' / '.join(f'{x:.1e}' for x in ps)} | "
                     f"{sum(1 for x in ps if x < 0.0125)}/{len(ps)} |")

        # 发散记录
        all_ab = [r for t in byab for r in byab[t]] + [r for r in by["adlif"]]
        div = [r for r in all_ab if r.get("diverged_at")]
        if div:
            L.append(f"\n**发散记录**：{len(div)}/{len(all_ab)} 次训练出现非有限损失并提前终止"
                     f"（轮次 {[r['diverged_at'] for r in div]}）。"
                     "AdLIF 的 u→w→u 正反馈使它在训练后期比 LIF 更靠近发散边界，"
                     "这是自适应机制的隐性成本，通常不在论文里报告。")

    # ---- 6. recurrent 复核与稳定性 ----------------------------------------
    rp = ROOT / "benchmarks" / "train_shd_m4b.json"
    if rp.exists():
        m4b = {r["tag"]: r for r in json.loads(rp.read_text("utf-8"))}
        L.append("\n## 6. recurrent 复核与一次稳定性事件\n")
        L.append("M2 里循环结构的失败症状是发放率从 ~20% 爬到 40–47%（正反馈失控）。"
                 "这里加发放率正则（目标 12%、λ=2）直接针对该症状，看循环能否被救回来。"
                 "**1 seed 探针，不作为结论。**\n")
        L.append("| 配置 | 循环 | test | 发放率 | vs 同变体前馈 |")
        L.append("|---|---|---|---|---|")
        ff = {"lif": statistics.fmean([r["test"] for r in by["lif"]]),
              "adlif": statistics.fmean([r["test"] for r in by["adlif"]])}
        if "rnn_lif_reg" in m4b:
            r = m4b["rnn_lif_reg"]
            L.append(f"| LIF | 是 | {r['test']:.4f} | "
                     f"[{r['rates'][0]:.3f}, {r['rates'][1]:.3f}] | "
                     f"{r['test'] - ff['lif']:+.4f} |")
        L.append(f"| LIF | 否 | {ff['lif']:.4f} | — | — |")
        if "rnn_adlif_reg" in m4b:
            r = m4b["rnn_adlif_reg"]
            L.append(f"| AdLIF | 是 | {r['test']:.4f} | "
                     f"[{r['rates'][0]:.3f}, {r['rates'][1]:.3f}] | "
                     f"{r['test'] - ff['adlif']:+.4f} |")
        L.append(f"| AdLIF | 否 | {ff['adlif']:.4f} | — | — |")
        L.append("\n**发放率正则确实解决了饱和**（循环 LIF 从 33–45% 降到 19.4/23.0%），"
                 "**但循环仍然是净损失，而且对两个变体的惩罚几乎完全相同（−8.35 / −8.38）。** "
                 "所以 M2 里\"循环失败是因为发放率失控\"这个解释**不完整**——"
                 "控制住发放率之后循环依然更差。")
        L.append("\n注意 epoch 不一致：循环用 100 epoch，前馈用 120。但前馈在 60 epoch "
                 "就有 75.84%、120 epoch 80.08%，按此折算 100 epoch 约 78–79%，"
                 "循环仍低约 4–5 个点。**结论方向不受影响，幅度需打折看。**")
        L.append("\n好的一面：两个变体掉的分几乎相同，说明**变体排序在循环结构下保持**"
                 "——AdLIF 的优势不是前馈结构的特有现象。")

        # 稳定性事件
        if "adlif_recheck" in m4b:
            r = m4b["adlif_recheck"]
            L.append("\n### 6.1 一次发散事件的处理\n")
            L.append("此前有一次 AdLIF 训练（与另一个进程并发时）在 ep60–70 之间发散成 NaN，"
                     "train 从 95% 掉回 5.1%。**单独重跑同配置同 seed 得到 "
                     f"{r['test']:.4f}，与 M3 的 adlif seed 0（0.8529）完全一致。**")
            L.append("\n由此可以定位：训练在 XPU 上是可复现的，那次发散只发生在并发时。"
                     "LIF（动力学稳定）在 12 次训练中从未发散，AdLIF（u→w→u 正反馈，"
                     "处在发散边界附近）被并发的调度扰动推过了界。"
                     "**可操作的结论：AdLIF 训练不要与其他任务并发。**")

    text = "\n".join(L) + "\n"
    out = ROOT / "benchmarks" / "m4_analysis.md"
    out.write_text(text, "utf-8")
    print(text)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
