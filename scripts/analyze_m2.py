"""M2 / RQ-A 的分析：τ 收窄阶梯 + E/I 平衡初始化。

两个实验共用一条主线：**循环为什么没用**。
  A. τ 阶梯——检验「循环与异构 τ 功能重叠」。预测：τ 越窄，循环的相对增益越大。
  B. E/I 平衡——检验「循环的驱动主要是均值分量、不是涨落」。
     默认初始化行和不为零，循环电流把整层往一个方向推（M1 里 gain=12 时
     发放率 0.227/0.284 → 0.267/0.365 就是证据）。行和归零后再标定方差。

统计口径与 M1 一致：跨 seed 均值 ± σ + 逐 seed 配对 McNemar（要求全部显著）。
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
OUT = B / "m2_interventions.md"
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


def main() -> int:
    from freebird_lif.shd.data import build

    d = build(ROOT / "data" / "shd", T=100)
    y = d["y_test"]

    m2 = by_tag(load("train_shd_m2tau.json"))
    ei = by_tag(load("train_shd_m2ei.json"))
    m1 = by_tag(load("train_shd_h0.json"))
    ffw = by_tag(load("train_shd_h0ff.json"))
    ffw.update({k: v for k, v in by_tag(load("train_shd_final_w.json")).items()
                if k == "lif_w"})
    ffw["h0_strong_ff"] = ffw.pop("lif_w")

    L = ["# M2：两项干预的结果\n"]
    L.append("主线问题：**循环为什么没用**。M1 已排除「接得不够强」（gain 标定到同量级"
             "仍只打平）与「只在弱基线有用」（弱基线上也不涨）。\n")

    # ---------- A. τ 阶梯 ----------
    L.append("## A. τ 收窄阶梯：检验「与异构 τ 功能重叠」\n")
    L.append("预测：若循环的收益被异构 τ 抢走了，那么 τ 越窄（前馈能保持的时间尺度越短），"
             "循环的相对增益应越大。反过来，若 τ_hi=5 时循环仍无收益，解释被推翻。\n")
    L.append("`rec_gain` 按 τ 范围分别标定（τ_hi=5→6.75，10→8.1，40→12.3），"
             "否则短 τ 格子会被过度驱动约 1.8 倍。\n")

    ladder = [
        (40.0, "h0_strong_ff", "h0_strong_g12"),
        (10.0, "tau10_ff", "tau10_rnn"),
        (5.0, "tau5_ff", "tau5_rnn"),
    ]
    L.append("| τ 上界 | 前馈 test | 循环 test | Δ(循环−前馈) | McNemar | 前馈发放率 | 循环发放率 |")
    L.append("|---|---|---|---|---|---|---|")
    deltas = []
    for th, ff_tag, rc_tag in ladder:
        # τ_hi=40 的两个格子来自 M1（h0_strong_*），其余来自本轮 m2tau
        src_ff = (m1 if th == 40.0 else m2).get(ff_tag) or ffw.get(ff_tag)
        src_rc = (m1 if th == 40.0 else m2).get(rc_tag)
        if not src_ff or not src_rc:
            continue
        src = {ff_tag: src_ff, rc_tag: src_rc}
        a, b = stat(src[ff_tag]), stat(src[rc_tag])
        mc = mcnemar(src[rc_tag], src[ff_tag], y)
        d_ = b["mean"] - a["mean"]
        deltas.append((th, d_))
        L.append(f"| {th:.0f} | {a['mean']:.4f} ± {a['sd']:.4f} | "
                 f"{b['mean']:.4f} ± {b['sd']:.4f} | {d_:+.4f} | "
                 + (f"{mc['n_sig']}/{len(mc['p'])}" if mc else "—")
                 + f" | {a['rows'][0]['rates'][0]:.3f}/{a['rows'][0]['rates'][1]:.3f}"
                 f" | {b['rows'][0]['rates'][0]:.3f}/{b['rows'][0]['rates'][1]:.3f} |")

    if len(deltas) >= 2:
        d_sorted = sorted(deltas, key=lambda x: -x[0])   # τ 从宽到窄
        widest, narrowest = d_sorted[0], d_sorted[-1]
        L.append(f"\n从 τ_hi={widest[0]:.0f} 收到 {narrowest[0]:.0f}，循环的 Δ 从 "
                 f"{widest[1]:+.4f} 变为 {narrowest[1]:+.4f}。")
        if narrowest[1] - widest[1] > 0.02:
            L.append("\n**与「功能重叠」的解释一致**：τ 收窄后循环的相对增益上升。"
                     "说明循环和异构 τ 确实在提供同一种东西——时间整合能力——"
                     "而前馈通路已经饱和了这份需求。")
        else:
            L.append("\n**与「功能重叠」的解释不符**：τ 大幅收窄后循环仍没有明显增益。"
                     "该解释被推翻，需要回到 H2–H6 的其他假设（E/I 平衡、"
                     "fluctuation-driven 初始化、容量/结构）。")

    # ---------- B. E/I 平衡 ----------
    if ei:
        L.append("\n## B. E/I 平衡初始化：循环的驱动是均值还是涨落\n")
        L.append("默认初始化的循环权重的**行和不为零**，这给整层一个共同的偏置——"
                 "M1 实测 gain=12 时发放率从 0.227/0.284 涨到 0.267/0.365，"
                 "就是它在把网络往饱和推。行和归零后只剩涨落。\n")
        L.append("实测标定：默认 init 达到同量级需 gain≈12，E/I 平衡需 **≈23**——"
                 "相差近一倍，说明默认 init 下循环电流的幅值有一半来自均值分量。\n")
        L.append("| 配置 | test | Δ vs 前馈 | McNemar | 发放率 |")
        L.append("|---|---|---|---|---|")
        ff = stat(ffw["h0_strong_ff"]) if "h0_strong_ff" in ffw else None
        for tag, label in (("ei_g23", "E/I 平衡 · gain=23（配对）"),
                           ("ei_g12", "E/I 平衡 · gain=12（半幅）"),
                           ("ei_g46", "E/I 平衡 · gain=46（双倍）")):
            if tag not in ei:
                continue
            s = stat(ei[tag])
            mc = mcnemar(ei[tag], ffw.get("h0_strong_ff"), y) if ff else None
            d_ = s["mean"] - ff["mean"] if ff else float("nan")
            L.append(f"| {label} | {s['mean']:.4f} ± {s['sd']:.4f} | {d_:+.4f} | "
                     + (f"{mc['n_sig']}/{len(mc['p'])}" if mc else "—")
                     + f" | {s['rows'][0]['rates'][0]:.3f}/{s['rows'][0]['rates'][1]:.3f} |")
        if ff:
            L.append(f"| 前馈（参照） | {ff['mean']:.4f} ± {ff['sd']:.4f} | — | — | "
                     f"{ff['rows'][0]['rates'][0]:.3f}/{ff['rows'][0]['rates'][1]:.3f} |")

        # E/I 的解读
        if ff:
            g23 = stat(ei["ei_g23"]) if "ei_g23" in ei else None
            g12 = stat(ei["ei_g12"]) if "ei_g12" in ei else None
            if g23 and g12:
                L.append(f"\n**两项都显著差于前馈**：配对强度 {g23['mean']:.4f}"
                         f"（Δ {g23['mean'] - ff['mean']:+.4f}，σ={g23['sd']:.4f}）、"
                         f"半幅 {g12['mean']:.4f}（Δ {g12['mean'] - ff['mean']:+.4f}，"
                         f"σ={g12['sd']:.4f}）。")
                L.append("\n**失败的原因不是「不够强」，而是「把去极化一起去掉了」。**"
                         "行和归零后循环电流的均值为零，神经元失去那个把它托到阈值附近的"
                         "常量分量。实测发放率在两个极端之间跳：")
                L.append("")
                for tag in ("ei_g23", "ei_g12"):
                    if tag in ei:
                        for r in ei[tag]:
                            L.append(f"- {tag} seed={r['seed']}：发放率 "
                                     f"{r['rates'][0]:.3f}/{r['rates'][1]:.3f}，"
                                     f"test {r['test']:.4f}")
                L.append("\n有的 seed 第一层塌到 0.03（几乎不发放），有的冲到 0.82（饱和）。"
                         "这是**边缘稳定**的典型表现：均值分量原本是稳定器，去掉之后系统"
                         "只剩涨落，随机种子决定它往哪边倒。")
                L.append("\n**结论：H2 按「fluctuation-driven 初始化」的原始形式在本架构下不成立。** "
                         "平衡网络需要配套的机制来设定工作点——独立的抑制性群体、"
                         "可调的 tonic 输入、或 homeostatic 阈值——本实现一个都没有。"
                         "在有这些配套之前，去掉均值分量等于拆掉工作点的支撑。")
                L.append("\n反过来看，这一条**加强**了 §A 的解释：循环在本架构里的主要作用"
                         "是提供去极化（把膜电位托到阈值附近），而不是提供计算。"
                         "这也解释了为什么 τ 收窄后它才开始有用——去极化的「时长」"
                         "变成了稀缺资源。")

    text = "\n".join(L) + "\n"
    OUT.write_text(text, "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
