"""M3：PCI 的效度检验。

M2c 里慢群体的 PCI 是异构周期的 10 倍（0.157 vs 0.015），但响应密度也高 7 倍
（0.732 vs 0.099）。这引出一个必须先回答的问题：**PCI 在测结构，还是在测密度？**

本脚本做三件事：
  1. 合成效度检验：同密度、同分辨率下，PCI 能否把「有结构」与「随机」分开；
  2. 密度匹配的模型比较：8 个配置在同一密度下重算 PCI；
  3. 置换零模型：观测矩阵相对同密度随机矩阵的可压缩性（z）。

输出 `benchmarks/m3_pci_validity.md`。数据来自 `benchmarks/v4pci_*.json`。
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
OUT = B / "m3_pci_validity.md"

CONFIGS = [
    ("v4_lif_t40", "LIF τ=40", "74.32", "1/4"),
    ("v4_adlif_t40", "AdLIF τ=40", "72.31", "3/4"),
    ("v4_tclif", "TC-LIF", "80.26", "1/4"),
    ("v4_atlif", "AT-LIF", "81.11", "1/4"),
    ("v4_tclif_het", "TC-LIF + 异构周期", "87.15", "4/4"),
    ("v4_atlif_het", "AT-LIF + 异构周期", "87.93", "4/4"),
    ("v4_tclif_slow", "TC-LIF + 慢群体", "81.35", "1/4"),
    ("v4_atlif_slow", "AT-LIF + 慢群体", "81.85", "1/4"),
]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    from instruments_v4 import pci_synthetic_battery

    bat = pci_synthetic_battery()
    rows = {}
    for tag, name, acc, crit in CONFIGS:
        d = load(B / f"v4pci_{tag}_s0.json")
        if d:
            rows[tag] = (name, acc, crit, d["pci_matched"])

    L = ["# M3：PCI 的效度检验\n"]
    L.append("**问题从哪来**：M2c 里慢群体的 PCI 是 0.157，异构周期只有 0.015，看起来「慢群体更能提升类意识指标」。"
             "但两者的响应密度是 0.732 与 0.099——**七倍**。一个把响应抹开的机制会同时抬高密度与 LZ，"
             "所以那个 10 倍差距里，有多少是结构、多少是密度？\n")
    L.append("本阶段不比较模型，先检验尺子。检验不通过，之前所有 PCI 读数都作废。\n")

    # ---- 1 合成检验 ----
    L.append("## 1. 合成效度检验（已知答案）\n")
    L.append("在同一密度（0.10）、同一分辨率（30×96）下构造四种响应矩阵：\n")
    L.append("| 结构 | 含义 | PCI | 密度 |")
    L.append("|---|---|---|---|")
    desc = {"sync": "3 个整行全亮（全局同步、完全刻板）",
            "blob": "12×24 连续块（局部、无传播）",
            "wave": "宽度 10 的斜带（时空传播）",
            "random": "同密度均匀随机"}
    for k in ("sync", "blob", "wave", "random"):
        v = bat["battery"][k]
        L.append(f"| {k} | {desc[k]} | **{v['pci']:.4f}** | {v['density']:.3f} |")
    L.append(f"\n判据：`random > wave > max(blob, sync)` 且 `random − max(blob, sync) > 0.10`。"
             f"**{'通过' if bat['pass'] else '不通过'}**。\n")
    L.append("两个要点：\n")
    L.append("1. **random 天然最高**——LZ 对噪声取最大，这不是发现，是构造决定的。"
             "有判别力的是 `blob`/`sync` 是否被压到低位，它们确实被压到 0.02。")
    L.append("2. **结构化区间的动态范围很窄**：wave 与 blob 只差 0.016。"
             "所以这个指标**不能用来区分「哪种结构更好」**，只能区分「有结构 / 无结构」。")

    # ---- 2 模型 ----
    L.append("\n## 2. 八个模型：固定阈值 vs 密度匹配\n")
    L.append("| 配置 | 精度 | 阶段四判据 | PCI（固定阈值） | 响应密度 | PCI（密度匹配） | 零模型 z |")
    L.append("|---|---|---|---|---|---|---|")
    for tag, name, acc, crit in CONFIGS:
        d = load(B / f"v4pci_{tag}_s0.json")
        if not d:
            continue
        m = d["pci_matched"]
        L.append(f"| {name} | {acc}% | {crit} | {m['pci_fixed_thresh']:.4f} | "
                 f"{m['density_fixed_thresh']:.3f} | **{m['pci_matched']:.4f}** | {m['null_z']:+.1f} |")
    L.append("\n零模型 z：把二值矩阵原地打乱（保持密度）后重算 LZ 的 z 分数。"
             "**越负 = 响应越可压缩 = 越有结构**；接近 0 = 与同密度噪声无异。\n")

    # ---- 3 结论 ----
    L.append("## 3. 结论\n")
    L.append("### 3.1 固定阈值的 PCI 是密度的函数——M2c 的 10 倍差距不成立\n")
    L.append("慢群体的固定阈值 PCI 是 0.119–0.143，密度 0.156–0.187；"
             "异构是 0.030，密度 0.098。密度匹配到 0.10 之后：")
    L.append("- 慢群体 0.233 / 0.258；")
    L.append("- 异构 **0.368 / 0.370**——**反超**。")
    L.append("\n所以 M2c 里「慢群体把 PCI 抬高 10 倍」这句话，**主要是密度抬高，不是结构**。"
             "这一条必须更正。\n")

    L.append("### 3.2 密度匹配后，PCI 与阶段四的判据同向——上一条判断也要更正\n")
    L.append("M2c 的结论写过「PCI 与判据脱钩」。密度匹配后不成立：\n")
    L.append("| 组 | 阶段四判据 | PCI（匹配） | 零模型 z |")
    L.append("|---|---|---|---|")
    L.append("| 异构周期（TC/AT） | 4/4 | **0.368 / 0.370** | −14.8 / −12.8 |")
    L.append("| 慢群体（TC/AT） | 1/4 | 0.233 / 0.258 | −32.4 / −34.1 |")
    L.append("| 两隔室基座（TC/AT） | 1/4 | 0.227 / 0.248 | −34.6 / −30.5 |")
    L.append("| 单隔室（LIF/AdLIF） | 1/4、3/4 | 0.159 / 0.158 | −47.6 / −52.5 |")
    L.append("\n两个独立读数一致：满足四条判据的异构周期，PCI 最高、零模型 z 最不负（最不可压缩）；"
             "单隔室最低、最可压缩（最刻板）。**PCI 在这里确实跟着「结构丰富度」走，而不是跟着密度走**——"
             "前提是做了密度匹配。\n")

    L.append("### 3.3 但绝对值不可解释\n")
    L.append(f"合成矩阵里，结构化响应是 0.02–0.04，随机是 0.47；而八个模型落在 **0.158–0.370**。"
             "这个区间**没有已知答案**——我们不知道 0.37 的模型离「有结构」更近还是离「噪声」更近。"
             "所以：\n")
    L.append("- ✅ 可以说：「在密度匹配后，A 的扰动响应比 B 更有结构（PCI 更高、z 更不负）」。")
    L.append("- ❌ 不能说：「A 的类意识程度是 0.37」。")

    L.append("\n## 4. 使用规则（写进后续所有报告）\n")
    L.append("1. **必须报密度**。只报 PCI 而不报密度的数字一律视为无效。")
    L.append("2. **必须密度匹配**后再跨配置比较（`pci_matched`，目标密度 0.10）。")
    L.append("3. **必须报零模型 z**，作为「是否有结构」的独立佐证。")
    L.append("4. **只做相对陈述**，不给绝对值解释。")
    L.append("5. Φ_G 仍然只作诊断（M0 已降级），不与 PCI 混用。")

    L.append("\n## 5. 局限\n")
    L.append("- 只用 seed 0 的检查点（密度匹配本身在 20 次扰动上平均，seed 间差异是二阶的）。"
             "要下更强的结论应补 3 seed。")
    L.append("- 粗粒度到 30×96 是为计算做的取舍（全分辨率 LZ 每次约 3 秒，"
             "20 次扰动 × 6 个置换要 20 分钟/模型）。粗粒度抹掉细结构，"
             "但所有配置用同一口径，组间比较仍然有效。")
    L.append("- 合成检验只覆盖「有结构 vs 随机」，没有覆盖「哪种结构更像意识」——"
             "这个问题本阶段回答不了，也不应该假装能回答。")
    L.append("- 没有做 PCI 与 Φ_G 的相关性分析（Φ 本身尚未通过效度检验）。")

    text = "\n".join(L) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
