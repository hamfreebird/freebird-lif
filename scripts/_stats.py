"""分析脚本共用的统计原语。

放在独立模块里而不是各脚本各写一份：McNemar 的精确二项检验和置信区间
是 M3/M4 全部结论的基础，两份实现漂移会让不同报告里的 p 值对不上。
"""

from __future__ import annotations

import math
import statistics

# t 分布双侧 95% 临界值。样本量小，查表比引 scipy 省一个依赖。
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def ci95(diffs: list[float]) -> tuple[float, float]:
    """配对差异的 95% 置信区间。n<2 时返回 (-inf, +inf)，表示无法估计。"""
    n = len(diffs)
    if n < 2:
        return (-math.inf, math.inf)
    m = statistics.fmean(diffs)
    half = T975.get(n - 1, 2.0) * statistics.stdev(diffs) / math.sqrt(n)
    return (m - half, m + half)


def exact_mcnemar(b: int, c: int) -> float:
    """McNemar 精确二项检验（双侧）。

    b = A 对 B 错的样本数，c = A 错 B 对的样本数。H0 下 b ~ Binom(b+c, 0.5)。
    样本量通常几十到几百，用精确分布而非卡方近似。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_pair(pred_a, pred_b, y) -> tuple[int, int, float]:
    """两次预测对同一批标签的 McNemar，返回 (b, c, p)。"""
    import numpy as np

    pa = np.asarray(pred_a, dtype=np.int64)
    pb = np.asarray(pred_b, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    ok_a, ok_b = pa == y, pb == y
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    return b, c, exact_mcnemar(b, c)
