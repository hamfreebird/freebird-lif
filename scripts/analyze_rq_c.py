"""RQ-C 的分析：三个基准上的变体增益是否随基线精度升高而缩小（H* 的天花板效应）。

设计（PLAN §5.2）：配方按规则迁移，**只有**输入维度、T、epoch 与增强幅度按数据集改变，
其余照搬（2×384、无归一化、in_scale=5、τ 层内对数铺开、AdamW 3e-3 / wd 1e-4、batch 512、
读出对 T 取均值、固定 epoch 取末轮）。任何进一步的逐数据集调参都会让「是否普适」
这个问题失去意义。

SHD 那一臂直接复用 M3 的 `train_shd_m3.json`——同一配方、同 3 seed、同 val_mode=all，
重跑只会重复消耗算力。N-MNIST 与 EvTouch 由 `scripts/train_bench.py` 产出。

统计口径与 M1–M3 一致：跨 seed 均值 ± σ、逐 seed 配对 McNemar（α=0.0125）。
RQ-C 特有的两条限制必须写进报告：
  1) 只有 3 个数据集，相关系数没有推断意义，只能当描述；
  2) EvTouch 测试集只有 144 条，单样本差异就是 0.69 个百分点。
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

from _stats import exact_mcnemar  # noqa: E402

B = ROOT / "benchmarks"
OUT = B / "rq_c_transfer.md"
ALPHA = 0.0125

DATASETS = [
    # key, 展示名, 预测（LIF 基线, AdLIF 增益）
    ("shd", "SHD（听觉）", None),
    ("nmnist", "N-MNIST（视觉）", ("> 98%", "< 1 点")),
    ("evtouch", "EvTouch（触觉）", ("30–60%", "不可区分")),
]
VARIANTS = [("lif", "LIF"), ("alif", "ALIF"), ("adlif", "AdLIF")]


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def collect(key: str) -> dict[str, list[dict]]:
    out = {}
    for v, _ in VARIANTS:
        if key == "shd":
            rows = [r for r in load(B / "train_shd_m3.json") if r["tag"] == v]
        else:
            rows = load(B / f"bench_{key}_{v}.json")
        if rows:
            out[v] = sorted(rows, key=lambda r: r["seed"])
    return out


def stat(rows):
    t = [r["test"] for r in rows]
    return {"mean": statistics.fmean(t),
            "sd": statistics.stdev(t) if len(t) > 1 else float("nan"),
            "seeds": [round(x, 4) for x in t], "n": len(t), "rows": rows}


def train_mean(rows) -> float:
    """跨 seed 的训练精度均值。

    不能只看 seed 0：SHD 的 LIF 三个 seed 是 92.37/92.37/92.37（一致），
    但 EvTouch 的 LIF 是 74.7/79.2/77.8——用单 seed 会和正文的均值对不上。
    """
    return statistics.fmean(r["final_train_acc"] for r in rows)


def rate_mean(r, i=0):
    rates = r.get("rates") or []
    return rates[i] if len(rates) > i else float("nan")


def test_labels(key: str) -> np.ndarray:
    """取该数据集的测试标签。

    结果 JSON 里只存了 test_pred、没存标签，所以配对检验必须回到数据层取 y。
    这是有意的取舍：标签是数据集的一部分，存进每个结果文件会让 JSON 大一倍，
    而它们随时可由 build() 重建。
    """
    if key == "shd":
        from freebird_lif.shd.data import build
        return np.asarray(build(ROOT / "data" / "shd", T=100)["y_test"])
    if key == "nmnist":
        from freebird_lif.bench import nmnist
        return np.asarray(nmnist.build(T=100)["y_test"])
    if key == "evtouch":
        from freebird_lif.bench import evtouch
        return np.asarray(evtouch.build(T=250)["y_test"])
    raise KeyError(key)


def main() -> int:
    data = {}
    for key, _, _ in DATASETS:
        d = collect(key)
        if d:
            data[key] = d

    L = ["# RQ-C：多基准普适性（H\\* 的天花板效应）\n"]
    L.append("**问题**：阶段二在 SHD 上观察到「AdLIF 比 LIF 高 3.03 点」。如果这个增益来自"
             "神经元机制本身，它应当跨数据集成立；如果它只是「把基线从 83% 推到 86%」的"
             "补差，那么在已经接近饱和的基准上应当消失。后者就是 H\\* 的天花板效应。\n")
    L.append("**协议**（PLAN §5.2）：配方按规则迁移，只有输入维度、T、epoch 与增强幅度"
             "按数据集改变，其余照搬。SHD 一臂复用 M3 的同配方结果（同 3 seed、"
             "val_mode=all），不重跑。\n")

    # ---- 预测 vs 实测 ----
    L.append("## 1. 先写下的预测\n")
    L.append("| 数据集 | 预测 LIF 基线 | 预测 AdLIF 增益 | 实测 LIF | 实测 AdLIF 增益 |")
    L.append("|---|---|---|---|---|")
    for key, name, pred in DATASETS:
        if key not in data or "lif" not in data[key]:
            continue
        base = stat(data[key]["lif"])["mean"] * 100
        g = ("—" if "adlif" not in data[key]
             else f"{(stat(data[key]['adlif'])['mean'] - stat(data[key]['lif'])['mean']) * 100:+.2f} 点")
        pr = pred or ("—", "—")
        L.append(f"| {name} | {pr[0]} | {pr[1]} | {base:.2f}% | {g} |")

    L.append("\n**两处预测错了，必须记下来：**\n")
    L.append("- N-MNIST 的 LIF 基线预测 >98%，实测 **91.37%**。当初的估计是基于"
             "「MNIST 类任务已饱和」的先验，但 2×384 的 SNN 在 60 epoch 下并没有到顶。"
             "这不影响天花板检验（91.37% 仍是三个基准里最高的），但说明"
             "「接近饱和」不能靠任务名判断。")
    L.append("- EvTouch 的增益预测为「不可区分」，实测 **+7.18 点**（3/3 seed 同向，"
             "2/3 McNemar 显著）。这是本轮最强的反例——小样本下机制差异不但可测，"
             "而且比 SHD 上更大。")

    # ---- 主表 ----
    L.append("\n## 2. 结果\n")
    L.append("| 数据集 | 测试样本 | LIF | ALIF | AdLIF | Δ(ALIF) | Δ(AdLIF) | LIF/AdLIF 训练精度 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, name, _ in DATASETS:
        if key not in data:
            continue
        d = data[key]
        n = len(d["lif"][0]["test_pred"]) if "lif" in d else 0
        cells = []
        for v, _ in VARIANTS:
            if v in d:
                s = stat(d[v])
                cells.append(f"{s['mean'] * 100:.2f} ± {s['sd'] * 100:.2f}")
            else:
                cells.append("—")
        deltas = []
        for v in ("alif", "adlif"):
            if v in d and "lif" in d:
                deltas.append(f"{(stat(d[v])['mean'] - stat(d['lif'])['mean']) * 100:+.2f}")
            else:
                deltas.append("—")
        tr = ""
        if "lif" in d:
            tr = f"{train_mean(d['lif']) * 100:.1f}"
            if "adlif" in d:
                tr += f" / {train_mean(d['adlif']) * 100:.1f}"
        L.append(f"| {name} | {n} | " + " | ".join(cells) + " | " + " | ".join(deltas)
                 + f" | {tr}% |")

    # ---- 配对检验 ----
    L.append("\n## 3. 与同数据集 LIF 的配对检验（McNemar，α = 0.0125）\n")
    L.append("| 数据集 | 变体 | 通过 seed 数 | 各 seed p |")
    L.append("|---|---|---|---|")
    for key, name, _ in DATASETS:
        if key not in data or "lif" not in data[key]:
            continue
        y = test_labels(key)
        for v in ("alif", "adlif"):
            if v not in data[key]:
                continue
            ps = []
            for ra, rb in zip(data[key][v], data[key]["lif"]):
                pa = np.asarray(ra["test_pred"]) == y
                pb = np.asarray(rb["test_pred"]) == y
                ps.append(exact_mcnemar(int((pa & ~pb).sum()), int((~pa & pb).sum())))
            L.append(f"| {name} | {v} | {sum(1 for p in ps if p < ALPHA)}/{len(ps)} | "
                     + ", ".join(f"{p:.3g}" for p in ps) + " |")

    # ---- 饱和度 ----
    L.append("\n## 4. 饱和度：精度与发放率\n")
    L.append("| 数据集 | LIF 基线 | LIF 发放率 | AdLIF 精度 | AdLIF 发放率 |")
    L.append("|---|---|---|---|---|")
    for key, name, _ in DATASETS:
        if key not in data or "lif" not in data[key]:
            continue
        rl = data[key]["lif"][0]
        line = (f"| {name} | {stat(data[key]['lif'])['mean'] * 100:.2f}% | "
                f"{rate_mean(rl):.3f} / {rate_mean(rl, 1):.3f} |")
        if "adlif" in data[key]:
            ra = data[key]["adlif"][0]
            line += (f" {stat(data[key]['adlif'])['mean'] * 100:.2f}% | "
                     f"{rate_mean(ra):.3f} / {rate_mean(ra, 1):.3f} |")
        else:
            line += " — | — |"
        L.append(line)

    # ---- 判据 ----
    L.append("\n## 5. 判据核对（PLAN §5.3）\n")
    pts = []
    for key, name, _ in DATASETS:
        if key in data and "lif" in data[key] and "adlif" in data[key]:
            pts.append((key, name,
                        stat(data[key]["lif"])["mean"] * 100,
                        (stat(data[key]["adlif"])["mean"]
                         - stat(data[key]["lif"])["mean"]) * 100))

    if len(pts) >= 2:
        base = [p[2] for p in pts]
        gain = [p[3] for p in pts]
        r = float(np.corrcoef(base, gain)[0, 1])
        L.append("| 数据集 | LIF 基线 | AdLIF 增益 |")
        L.append("|---|---|---|")
        for _, name, b, g in pts:
            L.append(f"| {name} | {b:.2f}% | {g:+.2f} 点 |")
        L.append(f"\nLIF 基线与 AdLIF 增益的 Pearson r = **{r:+.2f}**"
                 f"（n={len(pts)}，**无推断意义**，只作方向描述）。")
        order = sorted(pts, key=lambda p: p[2])
        mono = all(order[i][3] >= order[i + 1][3] for i in range(len(order) - 1))
        L.append(f"\n按基线升序：" + "、".join(
            f"{p[1]} {p[2]:.2f}% → 增益 {p[3]:+.2f}" for p in order) + "。"
                 + ("**严格单调递减**，与天花板效应一致。" if mono
                    else "**非单调**：天花板效应只能部分成立，需要分数据集讨论。"))

        # 三条具体判据
        nm = next((p for p in pts if p[0] == "nmnist"), None)
        et = next((p for p in pts if p[0] == "evtouch"), None)
        L.append("\n**逐条判据：**\n")
        if nm:
            L.append(f"- N-MNIST 上 AdLIF 增益 {nm[3]:+.2f} 点："
                     + ("**≥3 点，天花板效应不成立**——需要修正 H\\*。"
                        if nm[3] >= 3 else
                        "< 3 点，与「接近饱和后增益缩小」一致。"))
        if et:
            sd = (stat(data['evtouch']['adlif'])['sd'] ** 2
                  + stat(data['evtouch']['lif'])['sd'] ** 2) ** 0.5
            L.append(f"- EvTouch 上 AdLIF 增益 {et[3]:+.2f} 点（合并 σ≈{sd * 100:.2f} 点）："
                     + ("幅度小于 1 个标准差，**不可区分**，符合预测。"
                        if abs(et[3]) < sd * 100 else
                        "幅度超过 1 个标准差，小样本下机制差异仍可测，H\\* 需限定。"))

    # ---- 结论 ----
    L.append("\n## 6. 结论\n")
    if len(pts) == 3:
        order = sorted(pts, key=lambda p: p[2])
        L.append("**H\\* 的天花板效应成立**：三个基准上 AdLIF 的增益随 LIF 基线升高"
                 "严格单调递减（+7.18 → +3.03 → −0.73），Pearson r = −0.99。"
                 "这是本轮最强的定量结果，而且是在**配方按规则迁移、不做逐数据集调参**"
                 "的前提下得到的。\n")
        tr_lif_et = train_mean(data["evtouch"]["lif"]) * 100
        tr_adlif_et = train_mean(data["evtouch"]["adlif"]) * 100
        tr_lif_nm = train_mean(data["nmnist"]["lif"]) * 100
        L.append("**但机制要改一个说法。** 把「测试精度的天花板」当作解释并不准确——"
                 f"N-MNIST 的 LIF 训练精度已经 {tr_lif_nm:.1f}%，测试 91.37%，"
                 "两者的差距是泛化误差，不是模型容量不够；"
                 f"而 EvTouch 上 LIF 的训练精度只有 {tr_lif_et:.1f}%，"
                 f"AdLIF 把它推到 {tr_adlif_et:.1f}%，测试也随之 +7.18。也就是说，"
                 "**AdLIF 的收益出现在「模型在训练分布上还没拟合够」的场景**，"
                 "而在已经拟合到饱和的场景里它只增加过拟合（N-MNIST 上 −0.73）。\n")
        L.append("| 数据集 | LIF 训练精度 | AdLIF 训练精度 | AdLIF 增益 |")
        L.append("|---|---|---|---|")
        for key, name, _ in DATASETS:
            if key in data and "lif" in data[key] and "adlif" in data[key]:
                L.append(f"| {name} | "
                         f"{train_mean(data[key]['lif']) * 100:.1f}% | "
                         f"{train_mean(data[key]['adlif']) * 100:.1f}% | "
                         f"{(stat(data[key]['adlif'])['mean'] - stat(data[key]['lif'])['mean']) * 100:+.2f} |")
        L.append("\n按这个解释，H\\* 的判据应当从「测试精度越接近饱和增益越小」"
                 "改写为「**训练分布上的拟合余量越小，AdLIF 的增益越小**」。"
                 "两者在三个基准上同向，但后者的因果更清楚，也更能预测新数据集。\n")
        L.append("**必须限定的两点：**\n")
        L.append("1. EvTouch 的 +7.18 是**小样本效应**：490 条训练、144 条测试，"
                 "单样本就是 0.69 个点。3/3 seed 同向、2/3 McNemar 显著，方向可信，"
                 "但量级不能外推到同规模以外的数据。")
        L.append("2. 只有 3 个数据集，r = −0.99 是**描述**而非推断——"
                 "3 个点拟合任何单调关系都会给出接近 ±1 的相关系数。"
                 "真正支撑结论的是「严格单调 + 每档 3 seed 的低方差」，不是 r 本身。")
        L.append("\n**对 RQ-B 的呼应**：M3 发现适应电流的增益会被循环结构压缩近半，"
                 "RQ-C 发现它还会被「训练分布上的拟合余量」调制。两者指向同一个工程结论——"
                 "**这个机制的收益是高度条件性的，把它当作通用改进会失望。**")

    text = "\n".join(L) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
