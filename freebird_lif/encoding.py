"""输入编码：把 16×16 灰度图转成 LIF 层可消费的驱动信号。

三种方案（PLAN.md §5）：
    current    确定性电流注入 —— 主方案。恒定输入，可跨时间步缓存卷积结果
    poisson    泊松速率编码 —— 纯脉冲对照
    ttfs       首脉冲延迟编码 —— 每像素每样本恰发一个脉冲，最省
"""

from __future__ import annotations

import torch
from torch import Tensor


def current(x: Tensor, gain: float = 1.0) -> Tensor:
    """电流注入。x ∈ [0,1]，返回逐元素乘 gain 后的模拟电流，形状不变。

    因为信号与 t 无关，调用方只需计算一次 Conv1 并可复用全部 T 步。
    """
    return x * gain


def poisson(x: Tensor, T: int, gain: float = 1.0, generator: torch.Generator | None = None) -> Tensor:
    """速率编码。每步以概率 min(1, gain·x) 发放，输出 [B,T,...] 的 0/1 张量。"""
    p = (x * gain).clamp(0.0, 1.0).unsqueeze(0).expand((T,) + x.shape)
    u = torch.rand(p.shape, generator=generator, device=x.device, dtype=x.dtype)
    return (u < p).to(x.dtype)


def ttfs(x: Tensor, T: int, gain: float = 1.0) -> Tensor:
    """首脉冲延迟编码。像素越亮，发放越早；每像素在整段 T 内只发一次。

    延迟映射：t_first = round((1 - clip(gain·x,0,1)) · (T-1))
    因此纯黑像素在最后一步发放，纯白在第 0 步发放。
    """
    g = (x * gain).clamp(0.0, 1.0)
    delay = ((1.0 - g) * (T - 1)).round().to(torch.long)
    out = torch.zeros((T,) + x.shape, device=x.device, dtype=x.dtype)
    idx = delay.unsqueeze(0)
    out.scatter_(0, idx, 1.0)
    return out
