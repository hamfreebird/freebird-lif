"""Conv SNN 层栈、T 步前向、BN 折叠与 MACs 统计。

结构严格对应 PLAN.md §6：
    Conv1(1→12, 3×3, pad1) → BN → LIF → Pool → Conv2(12→24, 3×3, pad1)
    → BN → LIF → Pool → FC(384→96) → LIF → FC(96→10) → 泄漏积分读出

两处偏离教科书实现的地方，均已在 PLAN.md 变更记录登记：
1. 读出层 spiking=False —— 硬重置会把已累积的类别证据清零，必须用纯泄漏积分。
2. Conv1 结果跨时间步缓存 —— 电流注入模式下输入与 t 无关，缓存是数学等价的，
   且把 Conv1（占稠密 MACs 12%）的代价从 T 次降到 1 次。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .neuron import LIF


@dataclass
class SNNConfig:
    """完整超参，可直接序列化为 JSON 供推理引擎加载。"""

    T: int = 8
    tau: float = 2.0
    dt: float = 1.0
    thr: float = 1.0
    u_rest: float = 0.0
    alpha: float = 2.0
    surrogate: str = "fast_sigmoid"
    reset: str = "hard"
    method: str = "euler"
    pool_on: str = "spike"  # "spike" | "membrane"
    n_conv1: int = 12
    n_conv2: int = 24
    n_fc: int = 96
    n_class: int = 10
    input_gain: float = 1.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
    """把 BN 折进卷积权重：返回 (W', b')，使 conv'(x) ≡ bn(conv(x))。

    W' = γ/σ · W
    b' = γ/σ · (b - μ) + β
    推理引擎只认纯卷积 + 偏置，BN 在部署期应为零开销。
    卷积无偏置时按 b=0 处理。
    """
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std
    w = conv.weight * scale.reshape(-1, 1, 1, 1)
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    b = (bias - bn.running_mean) * scale + bn.bias
    return w, b


def fold_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> tuple[Tensor, Tensor]:
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std
    w = linear.weight * scale.reshape(-1, 1)
    b = (linear.bias - bn.running_mean) * scale + bn.bias
    return w, b


class ConvSNN(nn.Module):
    def __init__(self, cfg: SNNConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SNNConfig()
        c = self.cfg
        if c.pool_on not in ("spike", "membrane"):
            raise ValueError(f"pool_on 非法: {c.pool_on!r}，仅支持 'spike' 与 'membrane'")

        # 三个线性/卷积层均不带偏置：紧接其后的 BN 会减去通道均值，偏置在数学上被完全
        # 抵消（梯度恒为 0），保留只会制造死亡参数。
        self.conv1 = nn.Conv2d(1, c.n_conv1, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c.n_conv1)
        self.conv2 = nn.Conv2d(c.n_conv1, c.n_conv2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c.n_conv2)
        self.fc1 = nn.Linear(c.n_conv2 * 4 * 4, c.n_fc, bias=False)
        # FC 后必须有 BN：s2 是稀疏 0/1 脉冲，fc1 输出量级约 0.2，远低于 thr=1.0，
        # 缺 BN 时 LIF3 永远不会发放，整条深层通路在初始化即死亡。
        self.bn3 = nn.BatchNorm1d(c.n_fc)
        self.fc2 = nn.Linear(c.n_fc, c.n_class)

        lif_kw = dict(
            tau=c.tau, dt=c.dt, thr=c.thr, u_rest=c.u_rest,
            alpha=c.alpha, surrogate=c.surrogate, reset=c.reset, method=c.method,
        )
        self.lif1 = LIF(**lif_kw)
        self.lif2 = LIF(**lif_kw)
        self.lif3 = LIF(**lif_kw)
        self.lif4 = LIF(spiking=False, **lif_kw)  # 泄漏积分读出

    # ---- 前向 -------------------------------------------------------------

    def _conv_front(self, x: Tensor) -> Tensor:
        """Conv1 + BN（+ 可选膜电位池化）。恒定输入下只需算一次。"""
        h = self.bn1(self.conv1(x))
        if self.cfg.pool_on == "membrane":
            h = F.max_pool2d(h, 2)
        return h

    def _step(self, h1: Tensor, u1, u2, u3, u4, r1, r2, r3):
        s1, u1, r1 = self.lif1(h1, u1, r1)
        if self.cfg.pool_on == "spike":
            s1 = F.max_pool2d(s1, 2)

        h2 = self.bn2(self.conv2(s1))
        if self.cfg.pool_on == "membrane":
            h2 = F.max_pool2d(h2, 2)
        s2, u2, r2 = self.lif2(h2, u2, r2)
        if self.cfg.pool_on == "spike":
            s2 = F.max_pool2d(s2, 2)

        s3, u3, r3 = self.lif3(self.bn3(self.fc1(s2.flatten(1))), u3, r3)
        out, u4, _ = self.lif4(self.fc2(s3), u4, None)
        return out, u1, u2, u3, u4, r1, r2, r3

    def forward(self, x: Tensor, T: int | None = None, spikes: Tensor | None = None):
        """x: [B,1,16,16] 或 [B,256]；spikes: [T,B,1,16,16] 时走脉冲输入通路。

        返回 (logits [B,10], 逐时间步 logits 列表)。
        """
        if x.dim() == 2:
            x = x.reshape(-1, 1, 16, 16)
        T = T or self.cfg.T
        B = x.shape[0]
        dev, dt = x.device, x.dtype

        # 池化位置决定各级 LIF 的工作分辨率：
        #   spike   —— 池化在脉冲之后，LIF1 吃 16×16，LIF2 吃 8×8
        #   membrane—— 池化在 LIF 之前，LIF1 吃 8×8，LIF2 吃 4×4
        membrane = self.cfg.pool_on == "membrane"
        d1 = 8 if membrane else 16
        d2 = 4 if membrane else 8
        u1, r1 = self.lif1.init_state(B, self.cfg.n_conv1, d1, d1, device=dev, dtype=dt)
        u2, r2 = self.lif2.init_state(B, self.cfg.n_conv2, d2, d2, device=dev, dtype=dt)
        u3, r3 = self.lif3.init_state(B, self.cfg.n_fc, device=dev, dtype=dt)
        u4, _ = self.lif4.init_state(B, self.cfg.n_class, device=dev, dtype=dt)

        traces: list[Tensor] = []
        if spikes is None:
            # 电流注入：Conv1 与 t 无关，只算一次
            h1 = self._conv_front(x * self.cfg.input_gain)
            for _ in range(T):
                out, u1, u2, u3, u4, r1, r2, r3 = self._step(h1, u1, u2, u3, u4, r1, r2, r3)
                traces.append(out)
        else:
            if spikes.shape[0] != T:
                raise ValueError(f"spikes 时间维 {spikes.shape[0]} 与 T={T} 不符")
            for t in range(T):
                st = spikes[t]
                if st.dim() == 2:
                    st = st.reshape(-1, 1, 16, 16)
                h1 = self._conv_front(st)
                out, u1, u2, u3, u4, r1, r2, r3 = self._step(h1, u1, u2, u3, u4, r1, r2, r3)
                traces.append(out)

        return u4, traces

    # ---- 统计 -------------------------------------------------------------

    @torch.no_grad()
    def macs_report(self) -> dict[str, int]:
        """单时间步各层乘加次数。用于对齐 PLAN §6 并给基准提供分母。"""
        c = self.cfg
        m_conv1 = 16 * 16 * c.n_conv1 * (1 * 3 * 3)
        m_conv2 = 8 * 8 * c.n_conv2 * (c.n_conv1 * 3 * 3)
        m_fc1 = c.n_conv2 * 4 * 4 * c.n_fc
        m_fc2 = c.n_fc * c.n_class
        return {"conv1": m_conv1, "conv2": m_conv2, "fc1": m_fc1, "fc2": m_fc2}

    def amortized_macs(self, T: int | None = None) -> int:
        """T 步摊销后的总乘加。Conv1 只计一次（电流注入缓存）。"""
        T = T or self.cfg.T
        m = self.macs_report()
        return m["conv1"] + T * (m["conv2"] + m["fc1"] + m["fc2"])

    def param_count(self, include_bn: bool = True) -> int:
        n = 0
        for name, p in self.named_parameters():
            if not include_bn and (".bn" in name or name.startswith("bn")):
                continue
            n += p.numel()
        return n


def confidence(logits: Tensor, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
    """返回 (预测类别, 置信度)。temperature 为校准参数，训练期为 1.0。"""
    prob = torch.softmax(logits / temperature, dim=-1)
    conf, pred = prob.max(dim=-1)
    return pred, conf
