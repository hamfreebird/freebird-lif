"""温度校准驱动：在验证集上拟合温度，产出 ECE 报告与拒识曲线。

产物：
    models/calibration.json
    benchmarks/calibration.md

用法: python scripts/calibrate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from freebird_lif.calibrate import (
    ece,
    fit_temperature,
    nll,
    rejection_curve,
    softmax,
    suggest_tau,
)
from freebird_lif.data import load_raw, split_train_val, to_label, to_tensor
from freebird_lif.engine.export import load_bundle
from freebird_lif.engine.numba_engine import NumbaEngine

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "models" / "freebird-lif-T4.npz"
VAL_SIZE = 5000          # 与 train.py 的默认划分一致
VAL_SEED = 2024
TARGET_ACC = 0.995       # 拒识阈值的目标精度


def main() -> None:
    raw = load_raw()
    # 复现训练期的验证集划分：必须与训练集同源、与测试集隔离
    _, _, xv, yv = split_train_val(
        to_tensor(raw["x_train"]), to_label(raw["y_train"]), VAL_SIZE, seed=VAL_SEED
    )
    xv_np, yv_np = xv.numpy(), yv.numpy()

    w, cfg = load_bundle(BUNDLE)
    eng = NumbaEngine(w, cfg)
    logits = eng.forward(xv_np)
    print(f"校准集 {len(yv_np)} 条，未校准精度 {(logits.argmax(1) == yv_np).mean():.4f}", flush=True)

    t = fit_temperature(logits, yv_np)
    p_raw = softmax(logits)
    p_cal = softmax(logits / t)
    e_raw = ece(p_raw, yv_np)
    e_cal = ece(p_cal, yv_np)

    test_x = to_tensor(raw["x_test"]).numpy()
    test_y = raw["y_test"].astype(np.int64)
    test_logits = eng.forward(test_x)
    test_raw = ece(softmax(test_logits), test_y)
    test_cal = ece(softmax(test_logits / t), test_y)

    curve = rejection_curve(p_cal, yv_np)
    tau = suggest_tau(curve, TARGET_ACC)

    cal = {
        "temperature": t,
        "suggested_tau": tau,
        "target_accuracy": TARGET_ACC,
        "val": {"n": int(len(yv_np)), "ece_before": e_raw["ece"], "ece_after": e_cal["ece"],
                "nll_before": e_raw["nll"], "nll_after": e_cal["nll"],
                "acc": e_cal["accuracy"]},
        "test": {"n": int(len(test_y)), "ece_before": test_raw["ece"], "ece_after": test_cal["ece"],
                 "nll_before": test_raw["nll"], "nll_after": test_cal["nll"],
                 "acc": test_cal["accuracy"]},
        "val_split": {"val_size": VAL_SIZE, "seed": VAL_SEED},
    }
    (ROOT / "models" / "calibration.json").write_text(
        json.dumps(cal, indent=2, ensure_ascii=False), "utf-8")

    (ROOT / "benchmarks").mkdir(exist_ok=True)
    (ROOT / "benchmarks" / "calibration.md").write_text(render(cal, e_cal, curve), "utf-8")

    print(f"\n温度 T_cal = {t:.4f}")
    print(f"验证集 ECE {e_raw['ece']:.4f} → {e_cal['ece']:.4f}   NLL {e_raw['nll']:.4f} → {e_cal['nll']:.4f}")
    print(f"测试集 ECE {test_raw['ece']:.4f} → {test_cal['ece']:.4f}   NLL {test_raw['nll']:.4f} → {test_cal['nll']:.4f}")
    print(f"拒识 τ={tau}  目标精度 {TARGET_ACC:.1%}")
    print("\n" + (ROOT / "benchmarks" / "calibration.md").read_text("utf-8"))


def render(cal: dict, e_cal: dict, curve: list[dict]) -> str:
    L = [
        "# 置信度校准",
        "",
        f"校准集：训练集划出的 {cal['val']['n']} 条验证集（val_size={VAL_SIZE}, seed={VAL_SEED}），"
        "官方测试集完全隔离。",
        "",
        "## 温度缩放效果",
        "",
        "| 数据集 | ECE 前 | ECE 后 | NLL 前 | NLL 后 | 精度 |",
        "|---|---|---|---|---|---|",
        f"| 验证 | {cal['val']['ece_before']:.4f} | {cal['val']['ece_after']:.4f} | "
        f"{cal['val']['nll_before']:.4f} | {cal['val']['nll_after']:.4f} | {cal['val']['acc']:.4f} |",
        f"| 测试 | {cal['test']['ece_before']:.4f} | {cal['test']['ece_after']:.4f} | "
        f"{cal['test']['nll_before']:.4f} | {cal['test']['nll_after']:.4f} | {cal['test']['acc']:.4f} |",
        "",
        f"拟合温度 **T = {cal['temperature']:.4f}**"
        + (f"（{'高于' if cal['temperature'] > 1 else '低于'} 1，模型原本"
           f"{'过度自信' if cal['temperature'] > 1 else '信心不足'}）" if abs(cal['temperature'] - 1) > 0.02 else "（接近 1，模型已基本校准）"),
        "",
        "## 可靠性分箱（验证集，校准后）",
        "",
        "| 置信度区间 | 样本数 | 平均置信度 | 实测准确率 | 偏差 |",
        "|---|---|---|---|---|",
    ]
    for b in e_cal["bins"]:
        if b["n"] == 0:
            continue
        gap = b["conf"] - b["acc"]
        L.append(f"| {b['lo']:.2f}–{b['hi']:.2f} | {b['n']} | {b['conf']:.4f} | {b['acc']:.4f} | {gap:+.4f} |")
    L += [
        "",
        "## 拒识阈值曲线（验证集，校准后）",
        "",
        "| τ | 覆盖率 | 覆盖率内精度 |",
        "|---|---|---|",
    ]
    for p in curve[::4]:
        if p["accuracy"] is None:
            continue
        L.append(f"| {p['tau']:.3f} | {p['coverage']:.3f} | {p['accuracy']:.4f} |")
    L += [
        "",
        f"目标精度 **{cal['target_accuracy']:.1%}** 下建议 **τ = {cal['suggested_tau']}**：",
        "在满足该精度的前提下覆盖率最高。若上层更看重吞吐，可下调 τ。",
        "",
        "## 复现",
        "",
        "```bash",
        "python scripts/calibrate.py",
        "```",
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
