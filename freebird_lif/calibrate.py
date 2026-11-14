"""温度校准与置信度评估。

三项产出：
1. **温度 T_cal** —— 在验证集上最小化 NLL 拟合。用 log 空间网格搜索 + 局部细化，
   而非 LBFGS：NLL 对 T 近似单峰，网格搜索 200 点足以定位，且完全确定性可复现，
   不会因优化器超参不同给出不同答案。
2. **ECE** —— 期望校准误差，校准前后对比。
3. **拒识曲线** —— 置信度阈值 τ 与 (覆盖率, 覆盖率内精度) 的对应关系。

校准集必须与测试集隔离：使用训练脚本划分出的 5000 条验证集（seed=2024）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits / temperature)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lo: float = -2.0, hi: float = 2.0, n_grid: int = 241) -> float:
    """在 log10(T) ∈ [lo, hi] 上网格搜索最小 NLL，再在最优格点邻域细化两轮。"""
    best_t, best_v = 1.0, np.inf
    for t in np.logspace(lo, hi, n_grid):
        v = nll(logits, labels, t)
        if v < best_v:
            best_t, best_v = float(t), v
    # 两轮局部细化，步长逐次缩小 10 倍
    span = (hi - lo) / (n_grid - 1)
    for _ in range(2):
        span /= 10.0
        center = np.log10(best_t)
        for t in np.logspace(center - span, center + span, 21):
            v = nll(logits, labels, t)
            if v < best_v:
                best_t, best_v = float(t), v
    return best_t


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """期望校准误差：按最大置信度分箱，比较箱内平均置信度与实测准确率。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    total = len(labels)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if n == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        e += n / total * abs(c - a)
        bins.append({"lo": lo, "hi": hi, "n": n, "conf": c, "acc": a})
    return {"ece": float(e), "bins": bins, "mean_conf": float(conf.mean()),
            "accuracy": float(correct.mean()), "nll": float(-np.log(
                np.clip(probs[np.arange(total), labels], 1e-12, None)).mean())}


def rejection_curve(probs: np.ndarray, labels: np.ndarray, n_points: int = 41) -> list[dict]:
    """扫描拒识阈值 τ，返回覆盖率与覆盖率内精度。

    上层系统据此按需权衡：接受更多样本（覆盖率高）还是更高的准确率。
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    out = []
    for t in np.linspace(0.0, 0.999, n_points):
        m = conf >= t
        if m.sum() == 0:
            out.append({"tau": float(t), "coverage": 0.0, "accuracy": None, "n": 0})
            continue
        out.append({"tau": float(t), "coverage": float(m.mean()),
                    "accuracy": float(correct[m].mean()), "n": int(m.sum())})
    return out


def suggest_tau(curve: list[dict], target_accuracy: float) -> float | None:
    """在满足目标精度的前提下，取覆盖率最高的 τ。"""
    best = None
    for p in curve:
        if p["accuracy"] is not None and p["accuracy"] >= target_accuracy and p["coverage"] > 0:
            if best is None or p["coverage"] > best["coverage"]:
                best = p
    return None if best is None else best["tau"]
