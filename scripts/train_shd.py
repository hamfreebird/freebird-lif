"""SHD 上的 LIF 基线训练与配方搜索（M2）。

M0 的冒烟结果是 val 72% / test 53.5%，而文献同量级前馈 LIF 在 SHD 上
能到 80% 以上。差距不是一个超参能补的，必须逐项定位。本脚本把可能
影响结果的维度全部做成开关，一次只动一个变量。

被暴露的维度及其先验判断：
    dropout      隐藏层脉冲的空间 dropout。SHD 训练集仅 7156 条而模型
                 有约 20 万参数，train loss 到 0.09 说明在背样本。
    in_dropout   输入通道 dropout。SHD 的 700 通道来自耳蜗滤波，通道响应
                 强依赖说话人声道；丢通道是逼模型不看"谁在说"的最直接手段。
    readout      读出方式。四种都在文献里出现过，无法先验判断孰优。
    norm         LayerNorm / 无。LN 在 M0 里是被 BN 崩塌逼出来的选择，
                 但它对每个时间步按隐维归一，会抹掉"这一帧来了多少脉冲"。
    thr          阈值。LN 后隐层激活 std 恒为 1，thr=0.5 对应约 31% 发放率。
    arch         层数与宽度。文献常用 3×128。

用法:
    python scripts/train_shd.py                          # 默认配方单跑
    python scripts/train_shd.py --dropout 0.5 --tag d05
    python scripts/train_shd.py --grid round1            # 跑预置网格
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freebird_lif.shd.data import ROOT, build  # noqa: E402
from freebird_lif.variants import build as build_neuron  # noqa: E402

OUT = ROOT / "benchmarks"
N_IN, N_CLS = 700, 20


def _init_ei_balanced(weight: torch.Tensor, ei_ratio: float = 0.8) -> None:
    """H2 的干预：E/I 平衡的循环权重初始化（原地）。

    当前默认初始化是 iid 均匀分布，**行和不为零**——它有一个均值分量，
    把整层神经元往同一个方向推。M1 实测 gain=12 时发放率从 0.227/0.284
    涨到 0.267/0.365，正是这个均值分量在把网络推向饱和。

    这里做两件事：
      1. 按 ei_ratio 指定符号：80% 兴奋（正）、20% 抑制（负）
      2. 把抑制列整体平移，使每行和为零——均值分量为 0，只剩涨落
    然后按行 std 归一到 1，使 `rec_gain` 仍然表示"方差倍数"，与 default
    初化的语义一致，两个 init 之间可以直接比较。

    已知近似：平移后少数抑制权重可能变正（符号约束被轻微破坏）。这是
    有意的取舍——严格保号的行和归零需要解一个受限最小二乘，而本项目的
    目的是检验"去掉均值分量是否让循环变有用"，不需要精确的 E/I 结构。
    """
    N = weight.shape[1]
    n_inh = int(round((1.0 - ei_ratio) * N))
    if n_inh <= 0:
        raise ValueError("ei_ratio 必须 < 1，否则没有抑制性连接")
    inh = torch.randperm(N, device=weight.device)[:n_inh]
    # 幅度 × 符号：兴奋列取正、抑制列取负。
    # 不能用「对称高斯 × ±1」——那样正负各占一半，"兴奋"列里也有一半是负权重，
    # 实测正比例 0.501，等于没设 E/I。
    w = torch.randn_like(weight).abs()
    w[:, inh] = -w[:, inh]

    # 行和归零：把**整行**的和平摊到抑制列上减掉。
    # 注意不是只减兴奋和——第一版就是这样写的，结果新行和等于原抑制和，
    # 均值分量根本没去掉（实测 |行和| 平均 3.97，等于没做）。
    w[:, inh] -= w.sum(dim=1, keepdim=True) / float(n_inh)

    # 归一到与 nn.Linear 默认初始化相同的尺度：U(-a,a) 的 std = a/sqrt(3)，
    # a = 1/sqrt(N)。若在这里归一到 std=1，rec_gain 在两种 init 下的含义会差
    # 约 20 倍（N=128 时），两个 init 就无法直接比较。
    target = 1.0 / (3.0 ** 0.5 * N ** 0.5)
    w = w / w.std(dim=1, keepdim=True).clamp_min(1e-6) * target
    weight.copy_(w)


@dataclass
class Cfg:
    """一次训练的全部自由度。除 `tag` 外每一项都会影响结果。"""

    tag: str = "base"
    T: int = 100
    frames: int = 0            # >0 时只用前 frames 帧（诊断时间利用度）
    hidden: int = 256
    n_layers: int = 2
    variant: str = "lif"
    thr: float = 0.5
    tau_ms: float = 2.0
    dt_ms: float = 1.0
    tau_mode: str = "fixed"    # fixed | het（层内对数铺开，等价于可学滤波器组）
    tau_lo: float = 1.0        # het 的最快 τ（帧）
    tau_hi: float = 20.0       # het 的最慢 τ（帧）
    tau_layer_gain: float = 2.0  # 每深一层的 τ 倍数：浅层快、深层慢
    learn_tau: bool = True
    recurrent: bool = False    # 隐层间加循环连接（W_rec·s[t-1]）
    rec_gain: float = 1.0      # 循环权重初始增益，按 1/sqrt(N) 缩放
    rec_scale: bool = False    # 用 1/(1-α) 逐神经元缩放循环电流（H1）
    rec_init: str = "default"  # default | ei_balanced（H2：E/I 平衡 + 行和归零）
    ei_ratio: float = 0.8      # ei_balanced 下兴奋性连接占比
    norm: str = "ln"           # ln | bn | none
    dropout: float = 0.0       # 隐藏层脉冲 dropout
    in_dropout: float = 0.0    # 输入脉冲 dropout
    in_scale: float = 1.0      # 输入电流增益
    readout: str = "mean_u"    # mean_u | last_u | max_u | sum_s
    alpha_out: float = 0.6     # 读出层泄漏系数；1.0 为无泄漏累加（会随 T 放大 logits）
    aug_chan: int = 0          # 频道平移上限（±通道），模拟声道长度差异（VTLP 的线性近似）
    aug_time: int = 0          # 时间平移上限（±帧）
    aug_drop: float = 0.0      # 输入脉冲随机丢弃比例
    rate_target: float = 0.0   # 发放率正则的目标（0 表示关闭）
    rate_lambda: float = 0.0   # 发放率正则权重
    save_weights: bool = False # 训练结束保存 state_dict 到 runs/
    # 阶段四：两隔室变体的内在周期谱（逐神经元异构）
    intrinsic_het: bool = False
    period_lo: float = 8.0     # 最短内在周期（帧）
    period_hi: float = 80.0    # 最长内在周期（帧）
    # 阶段四：独立慢群体（k 个非脉冲泄漏积分单元，读顶层、调制第 0 层）
    slow_k: int = 0            # 0 = 关闭
    slow_tau: float = 100.0    # 慢群体时间常数（帧）
    slow_gain: float = 1.0     # 反馈电流增益
    learn_a: bool = True       # AdLIF：膜电位驱动的适应电流是否可学
    learn_b: bool = True       # AdLIF：发放驱动的适应电流是否可学
    label_smooth: float = 0.0
    epochs: int = 40
    lr: float = 3e-3
    wd: float = 1e-4
    n_in: int = 700            # 输入通道数（SHD=700；EvTouch=39；N-MNIST=1156）
    n_classes: int = 20        # 类别数（SHD=20；EvTouch=36；N-MNIST=10）
    batch: int = 512
    val_mode: str = "random"   # random | speaker | all
    seed: int = 0


def augment(xb: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    """在设备上做整批增强，每种都用一次 gather/乘法完成。

    SHD 的关键难点是说话人泛化：官方测试集 81% 的样本来自训练时未见的
    两位说话人。文献里 Cramer 2020 用数据增强把 RSNN 从 71.4% 提到 83.2%，
    增幅远超 dropout 之类的正则化——所以增强才是这个数据集上的主正则手段。

    `aug_chan` 是声道长度扰动的线性近似：700 个耳蜗通道按频率排列，
    整体平移等价于改变共振峰位置，标签不变而说话人特征改变。
    """
    if cfg.aug_chan:
        b, T, C = xb.shape
        dev = xb.device
        c = torch.randint(-cfg.aug_chan, cfg.aug_chan + 1, (b,))
        idx = (torch.arange(C)[None, :] - c[:, None]) % C
        # 必须物化成连续张量：expand 出的 stride-0 视图会让 XPU 的 gather
        # 触发运行时断言并且整进程静默退出（无异常、无 traceback）。
        xb = xb.gather(2, idx[:, None, :].repeat(1, T, 1).to(dev))
    if cfg.aug_time:
        if xb.device.type == "xpu":
            # 时间平移用 gather 在 XPU 上会触发运行时断言并让整进程静默退出。
            # 小张量能过、训练 batch 必崩，是尺寸相关的驱动问题。
            # 频道平移（真正的主力增强）没有这个问题，故只禁掉这一项。
            raise RuntimeError("aug_time 在 XPU 上不可用，见源码注释")
        b, T, C = xb.shape
        dev = xb.device
        sh = torch.randint(-cfg.aug_time, cfg.aug_time + 1, (b,))
        ti = torch.arange(T)[None, :] - sh[:, None]
        keep = (ti >= 0).to(xb.dtype).to(dev)
        xb = xb.gather(1, ti.clamp_min(0)[:, :, None].repeat(1, 1, C).to(dev))
        xb = xb * keep[:, :, None]
    if cfg.aug_drop:
        mask = (torch.rand(xb.shape, device=xb.device) >= cfg.aug_drop).to(xb.dtype)
        xb = xb * mask / (1.0 - cfg.aug_drop)
    return xb


def _mk_norm(kind: str, n: int) -> nn.Module:
    if kind == "ln":
        return nn.LayerNorm(n)
    if kind == "bn":
        return nn.BatchNorm1d(n)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"未知 norm {kind!r}")


class SHDNet(nn.Module):
    """700 → hidden × n_layers → 20 的前馈 SNN。

    时间维是显式 Python 循环。这既是 SNN 的定义方式，也是 M0 里
    XPU 加速比随 batch 增长的原因——每步只有几个小算子，启动开销占主导。
    """

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        dims = [cfg.n_in] + [cfg.hidden] * cfg.n_layers
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(cfg.n_layers)])
        # norm=none 时偏置是必要的（没有 BN 来吸收它）
        self.norms = nn.ModuleList([_mk_norm(cfg.norm, cfg.hidden)
                                    for _ in range(cfg.n_layers)])
        kw = dict(thr=cfg.thr, dt_ms=cfg.dt_ms, learn_tau=cfg.learn_tau)
        if cfg.variant in ("adlif", "atlif"):
            kw.update(learn_a=cfg.learn_a, learn_b=cfg.learn_b)
        if cfg.variant in ("tclif", "atlif"):
            kw.update(intrinsic_het=cfg.intrinsic_het,
                      period_lo=cfg.period_lo, period_hi=cfg.period_hi)
        self.neurons = nn.ModuleList(
            [build_neuron(cfg.variant, cfg.hidden, tau_ms=cfg.tau_ms, **kw)
             for _ in range(cfg.n_layers)])
        if cfg.tau_mode == "het":
            # 同一层里 τ 从 tau_lo 到 tau_hi 对数铺开：快神经元抓瞬态、慢神经元
            # 做长积分，等价于一个可学的时间滤波器组。层间再乘 tau_layer_gain，
            # 使浅层处理快速声学特征、深层整合到音节尺度。
            base = torch.logspace(math.log10(cfg.tau_lo), math.log10(cfg.tau_hi),
                                  cfg.hidden)
            self.neurons = nn.ModuleList(
                [build_neuron(cfg.variant, cfg.hidden,
                              tau_ms=base * cfg.tau_layer_gain ** i, **kw)
                 for i in range(cfg.n_layers)])
        elif cfg.tau_mode != "fixed":
            raise ValueError(f"未知 tau_mode {cfg.tau_mode!r}")
        # 循环权重：不给偏置，初始化按 1/sqrt(N) 缩放使初始循环电流与
        # 前馈电流同量级（s 的密度约 0.1，W·s 的 std ≈ gain·sqrt(p)）
        self.recs = nn.ModuleList()
        if cfg.recurrent:
            for _ in range(cfg.n_layers):
                r = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
                with torch.no_grad():
                    if cfg.rec_init == "ei_balanced":
                        _init_ei_balanced(r.weight, cfg.ei_ratio)
                    elif cfg.rec_init != "default":
                        raise ValueError(f"未知 rec_init {cfg.rec_init!r}")
                    r.weight.mul_(cfg.rec_gain)
                self.recs.append(r)

        # H1 的 rec_scale 在 forward 里按当前 α 现算，不在这里缓存：
        #   1) 普通 list 不会被 .to(device) 搬移，会静默留在 CPU
        #   2) α 可学（learn_tau=True），快照会在训练中过期
        if cfg.recurrent and cfg.rec_scale:
            for ne in self.neurons:
                if ne.input_scale() is None:
                    raise ValueError(
                        f"变体 {cfg.variant} 的输入不是 (1-α)·x 形式，rec_scale 无定义；"
                        f"请关掉 rec_scale 或先单独标定该变体的循环增益")
        # 独立慢群体（阶段四选项 3）：k 个非脉冲泄漏积分单元，读顶层脉冲、
        # 以低秩投影调制第 0 层的输入电流。默认 slow_k=0 时完全不建模块，
        # 行为与之前逐位一致。
        #
        # 为什么用**非脉冲**单元：慢单元在 τ=100 帧下发放率极低（<1 Hz 等效），
        # 用脉冲会带来极大的方差而信息量很少。这里先测「慢群体 + 反馈」这个
        # 结构假设；把它换成脉冲群体是阶段五的正式任务（那时还要做层级）。
        self.slow_in = None
        self.slow_out = None
        self.slow_decay = 1.0
        if cfg.slow_k > 0:
            self.slow_in = nn.Linear(cfg.hidden, cfg.slow_k, bias=False)
            self.slow_out = nn.Linear(cfg.slow_k, cfg.hidden, bias=False)
            with torch.no_grad():
                self.slow_in.weight.normal_(0.0, 1.0 / math.sqrt(cfg.hidden))
                # 反馈电流初始要小：太大一上来就把第 0 层推向饱和
                self.slow_out.weight.normal_(0.0, 0.3 / math.sqrt(cfg.slow_k))
            self.slow_decay = math.exp(-1.0 / max(cfg.slow_tau, 1e-3))

        self.fc_out = nn.Linear(cfg.hidden, cfg.n_classes)
        # sum_s 读出需要一个会发放的输出层；其余读出用非脉冲积分器
        self.out_neuron = (build_neuron(cfg.variant, cfg.n_classes, **kw)
                           if cfg.readout == "sum_s" else None)

    # 扰动钩子：None 时完全不改变行为（训练路径不受影响）。
    # 支持单个 dict 或 dict 列表，每个事件形如
    #   {"t": int, "layer": int, "mask": Tensor[hidden], "amp": float}
    # 列表形式是阶段五 C2 协议需要的：在 t0 扰动高层、在 t0+Δ 给低层加探测脉冲。
    # 放在类属性而非 __init__ 里，是为了不污染 Cfg 字段的序列化。
    perturb: dict | None = None

    def _perturb_events(self) -> list:
        """把 `perturb` 归一化成事件列表（dict → 单元素列表）。"""
        p = self.perturb
        if p is None:
            return []
        return [p] if isinstance(p, dict) else list(p)

    def _slow_advance(self, slow, top, b, dev, dt):
        """慢群体前进一步。返回 (slow, 调制电流)。

        `top` 是**上一时刻**顶层脉冲；t=0 时为 None，不更新。
        两条前向路径（forward / spike_rates）都必须调这一个函数，
        否则发放率正则测到的动力学与真正训练的不是同一个。
        """
        if self.slow_in is None or top is None:
            return slow, None
        if slow is None:
            slow = torch.zeros(b, self.cfg.slow_k, device=dev, dtype=dt)
        slow = (self.slow_decay * slow
                + (1.0 - self.slow_decay) * torch.tanh(self.slow_in(top)))
        return slow, self.cfg.slow_gain * self.slow_out(slow)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        T = self.cfg.frames or x.shape[1]
        dev, dt = x.device, x.dtype
        states = [n.init_state(b, self.cfg.hidden, device=dev, dtype=dt)
                  for n in self.neurons]
        u_out = torch.zeros(b, self.cfg.n_classes, device=dev, dtype=dt)
        self.alpha_out = self.cfg.alpha_out
        self.rate_reg = None
        s_out = None
        out_st = (self.out_neuron.init_state(b, self.cfg.n_classes, device=dev, dtype=dt)
                  if self.out_neuron is not None else None)

        acc = torch.zeros(b, self.cfg.n_classes, device=dev, dtype=dt)
        peak = torch.full((b, self.cfg.n_classes), float("-inf"), device=dev, dtype=dt)

        h = x[:, :T] * self.cfg.in_scale
        if self.cfg.in_dropout > 0 and self.training:
            mask = (torch.rand_like(h) >= self.cfg.in_dropout).to(dt)
            # 反向补偿，保持每帧总电流的期望不变
            h = h * mask / (1.0 - self.cfg.in_dropout)

        prev = [None] * self.cfg.n_layers   # 上一步各层脉冲，供循环连接用
        slow = None                          # 慢群体状态 [B, slow_k]
        # H1：见构造函数注释。每层一个 [N] 向量，随 α 的当前值现算。
        rec_sc = ([ne.input_scale() for ne in self.neurons]
                  if (self.cfg.recurrent and self.cfg.rec_scale) else None)
        rate_sum = None                     # 可求导的发放率累计，供正则项使用
        rates = []
        for t in range(T):
            a = h[:, t]
            slow, slow_mod = self._slow_advance(slow, prev[-1], b, dev, dt)
            for i, (fc, nrm, ne) in enumerate(zip(self.fcs, self.norms, self.neurons)):
                cur = nrm(fc(a))
                if i == 0 and slow_mod is not None:
                    cur = cur + slow_mod
                if self.cfg.recurrent and prev[i] is not None:
                    rec = self.recs[i](prev[i])
                    cur = cur + (rec if rec_sc is None else rec * rec_sc[i])
                    # 缩放后再送进神经元，循环项在 u 方程里的系数即为 W_rec 本身

                if self.perturb is not None:
                    for ev in self._perturb_events():
                        if t == ev["t"] and i == ev["layer"]:
                            cur = cur + ev["mask"] * ev["amp"]
                s, states[i] = ne.step(cur, states[i])
                prev[i] = s
                if self.cfg.dropout > 0 and self.training:
                    s = F.dropout(s, self.cfg.dropout, training=True)
                a = s
                layer_mean = a.mean()
                rate_sum = layer_mean if rate_sum is None else rate_sum + layer_mean
            rates.append(a.detach())

            if self.out_neuron is not None:
                s, out_st = self.out_neuron.step(self.fc_out(a), out_st)
                s_out = s if s_out is None else s_out + s
            else:
                # 非脉冲读出用泄漏积分：u 有界，logits 量级不随 T 增长。
                # 无泄漏累加（alpha_out=1）会让 logits 按 O(T) 放大，T=100 时
                # 初始 CE 就到 19.6，训练直接发散——这条留给显式实验，不做默认。
                u_out = self.alpha_out * u_out + (1 - self.alpha_out) * self.fc_out(a)
                acc = acc + u_out
                peak = torch.maximum(peak, u_out)

        # rate_sum 是 L 个层的均值之和；除 (L*T) 得到"每层每步的平均发放率"
        self.rate_reg = (rate_sum / (self.cfg.n_layers * T)
                         if rate_sum is not None else None)
        if self.out_neuron is not None:
            return s_out
        if self.cfg.readout == "mean_u":
            return acc / T
        if self.cfg.readout == "last_u":
            return u_out
        if self.cfg.readout == "max_u":
            return peak
        raise ValueError(f"未知 readout {self.cfg.readout!r}")

    @torch.no_grad()
    def spike_rates(self, x: torch.Tensor) -> list[float]:
        """测量各隐层在真实输入下的平均发放率。train 模式下调用。"""
        was = self.training
        self.train()
        b = x.shape[0]
        T = self.cfg.frames or x.shape[1]
        dev, dt = x.device, x.dtype
        states = [n.init_state(b, self.cfg.hidden, device=dev, dtype=dt)
                  for n in self.neurons]
        tot = [0.0] * self.cfg.n_layers
        h = x[:, :T] * self.cfg.in_scale
        prev = [None] * self.cfg.n_layers
        slow = None
        rec_sc = ([ne.input_scale() for ne in self.neurons]
                  if (self.cfg.recurrent and self.cfg.rec_scale) else None)
        for t in range(T):
            a = h[:, t]
            slow, slow_mod = self._slow_advance(slow, prev[-1], b, dev, dt)
            for i, (fc, nrm, ne) in enumerate(zip(self.fcs, self.norms, self.neurons)):
                cur = nrm(fc(a))
                if i == 0 and slow_mod is not None:
                    cur = cur + slow_mod
                if self.cfg.recurrent and prev[i] is not None:
                    rec = self.recs[i](prev[i])
                    cur = cur + (rec if rec_sc is None else rec * rec_sc[i])
                    # 缩放后再送进神经元，循环项在 u 方程里的系数即为 W_rec 本身

                if self.perturb is not None:
                    for ev in self._perturb_events():
                        if t == ev["t"] and i == ev["layer"]:
                            cur = cur + ev["mask"] * ev["amp"]
                s, states[i] = ne.step(cur, states[i])
                prev[i] = s
                tot[i] += s.mean().item() / T
                a = s
        self.train(was)
        return tot


@torch.no_grad()
def predict(model: SHDNet, x, bs: int = 512) -> np.ndarray:
    """返回逐样本预测类别。用于 McNemar 检验——它比 3 seed 的配对 t 检验
    功力高得多，因为它逐样本比较两个模型在同一测试集上的对错。"""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i : i + bs]).float().to(dev)
        out.append(model(xb).argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(out)


@torch.no_grad()
def evaluate(model: SHDNet, x, y, bs: int = 512) -> float:
    model.eval()
    dev = next(model.parameters()).device
    c = 0
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i : i + bs]).float().to(dev)
        c += int((model(xb).argmax(1).cpu() == torch.from_numpy(y[i : i + bs])).sum())
    return c / len(x)


def make_splits(d: dict, val_mode: str) -> tuple:
    """返回 (训练集索引, 验证集索引)。

    val_mode="random"  从训练集随机切 1000 条（SHD 标准做法，但验证集与
                       训练集同说话人，无法反映测试集的说话人泛化难度）。
    val_mode="speaker" 留出 2 位说话人做验证。SHD 测试集 81% 的样本来自
                       训练时未见过的说话人，用同说话人验证集选模型等于
                       在优化错误的信号。
    """
    n = len(d["y_train"])
    if val_mode == "random":
        return None, None
    if val_mode == "all":
        # 用满全部训练数据。SHD 没有官方验证集，文献的通行做法就是在
        # 全量 8156 条上训练、报测试集。代价是无法早停，只能取末轮权重。
        return np.arange(n), None
    if val_mode != "speaker":
        raise ValueError(f"未知 val_mode {val_mode!r}")
    spk = d.get("speaker_train")
    if spk is None:
        raise RuntimeError("缺少 speaker_train，需重建缓存（build(..., with_speaker=True)）")
    # 训练集中样本最少的两类说话人做验证，尽量少损失训练数据
    uniq, counts = np.unique(spk, return_counts=True)
    order = uniq[np.argsort(counts)]
    hold = set(order[:2].tolist())
    mask = np.isin(spk, list(hold))
    return np.where(~mask)[0], np.where(mask)[0]


def run(cfg: Cfg, d: dict, dev, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    model = SHDNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    g = torch.Generator().manual_seed(cfg.seed)
    xn, yn = d["x_train"], d["y_train"]
    xv, yv = d["x_val"], d["y_val"]
    tr_idx, va_idx = make_splits(d, cfg.val_mode)
    if tr_idx is not None:
        xn, yn = xn[tr_idx], yn[tr_idx]
        xv, yv = ((d["x_train"][va_idx], d["y_train"][va_idx])
                  if va_idx is not None else (None, None))
    T = cfg.frames or d["meta"]["T"]
    t0 = time.time()
    hist = []
    diverged_at = None

    for ep in range(cfg.epochs):
        perm = torch.randperm(len(xn), generator=g)
        run_loss, seen, correct = 0.0, 0, 0
        for i in range(0, len(xn), cfg.batch):
            idx = perm[i : i + cfg.batch].numpy()
            xb = torch.from_numpy(xn[idx]).float().to(dev, non_blocking=True)
            yb = torch.from_numpy(yn[idx]).to(dev, non_blocking=True)
            if cfg.aug_chan or cfg.aug_time or cfg.aug_drop:
                xb = augment(xb, cfg)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=cfg.label_smooth)
            if not torch.isfinite(loss):
                # 静默跑完 40 轮 NaN 会浪费时间且污染结果。AdLIF 的 u→w→u 正反馈
                # 让它比 LIF 更靠近发散边界，这个检查是必要的。
                diverged_at = ep + 1
                print(f"    !! 第 {ep + 1} 轮出现非有限损失，提前终止", flush=True)
                break
            if cfg.rate_lambda and model.rate_reg is not None:
                # 家庭稳态式的发放率惩罚。循环网络失败的直接症状是发放率
                # 从 ~20% 爬到 40%+（正反馈失控），这一项直接针对该症状。
                loss = loss + cfg.rate_lambda * (model.rate_reg - cfg.rate_target) ** 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            correct += int((logits.argmax(1) == yb).sum())
            seen += len(idx)
        if diverged_at:
            break
        sched.step()
        if verbose and ((ep + 1) % 10 == 0 or ep == 0 or ep == cfg.epochs - 1):
            va = evaluate(model, xv, yv) if xv is not None else float("nan")
            hist.append({"epoch": ep + 1, "loss": run_loss / seen,
                         "train_acc": correct / seen, "val": va})
            print(f"    ep{ep + 1:3d}  loss={run_loss / seen:.4f}  "
                  f"train={correct / seen:.4f}  val={va:.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    xb = torch.from_numpy(np.ascontiguousarray(xn[:512])).float().to(dev)
    rates = model.spike_rates(xb)
    te_pred = predict(model, d["x_test"])
    if cfg.save_weights:
        # 只存 state_dict，不存优化器状态：前者约 1.7 MB，后者会再翻两倍。
        # 配置随权重一起存，避免"有权重但不知道是什么配方"。
        wdir = ROOT / "runs"
        wdir.mkdir(exist_ok=True)
        torch.save({"config": asdict(cfg), "state_dict": model.state_dict(),
                    "test_pred": te_pred}, wdir / f"{cfg.tag}_s{cfg.seed}.pt")
    r = {
        **asdict(cfg),
        "val": evaluate(model, xv, yv) if xv is not None else float("nan"),
        "test": evaluate(model, d["x_test"], d["y_test"]),
        "final_train_acc": correct / seen,
        "final_loss": run_loss / seen,
        "rates": rates,
        "epochs_run": cfg.epochs,
        "diverged_at": diverged_at,
        "seconds": time.time() - t0,
        "params": sum(p.numel() for p in model.parameters()),
        "float_state": model.neurons[0].float_state,
        "op_count": model.neurons[0].op_count(),
        "test_pred": te_pred.tolist(),
        "hist": hist,
    }
    return r


# 预置网格。每轮只动一个维度，便于归因。
GRIDS: dict[str, list[Cfg]] = {
    # 第一轮：先分清"拟合不足"与"泛化不足"，再看读出与归一化
    "r1": [
        Cfg(tag="base"),
        Cfg(tag="drop50", dropout=0.5),
        Cfg(tag="indrop20", in_dropout=0.2),
        Cfg(tag="drop30_indrop20", dropout=0.3, in_dropout=0.2),
        Cfg(tag="lastu", readout="last_u"),
        Cfg(tag="maxu", readout="max_u"),
        Cfg(tag="sums", readout="sum_s"),
        Cfg(tag="nonorm", norm="none", in_scale=5.0),
        Cfg(tag="arch3x128", n_layers=3, hidden=128),
    ],
    # 第三轮：循环连接。文献里 LIF 的 87.04% 属于循环结构而非前馈，
    # 纯前馈单隐层的公开基线只有 48.1%（Cramer 2020）。故先验证"前馈天花板"
    # 是否真实存在，再决定 RQ1 的主结构。
    "r3": [
        # 前馈参考（说话人留出验证）
        Cfg(tag="ff_2x256_nonorm", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0),
        # 循环：同一配方下只加 W_rec
        Cfg(tag="rnn_2x256_nonorm", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, recurrent=True),
        Cfg(tag="rnn_2x256_ln", epochs=60, val_mode="speaker", recurrent=True),
        # 文献结构：3 层 × 128
        Cfg(tag="rnn_3x128_nonorm", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, recurrent=True, n_layers=3, hidden=128),
        # 循环 + 输入通道 dropout（说话人不变性的直接手段）
        Cfg(tag="rnn_3x128_nonorm_ind20", epochs=60, val_mode="speaker",
            norm="none", in_scale=5.0, recurrent=True, n_layers=3, hidden=128,
            in_dropout=0.2),
    ],
    # 第四轮：时间常数与训练长度。r3 证明"加循环"不是免费的——
    # rec_gain=1.0 会把网络推进 40% 发放的饱和区，需要重新标定。
    # 本轮先验证两条有文献支持、且不依赖循环的路线：异构 τ、更长训练。
    "r4": [
        Cfg(tag="tauhet", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=20.0),
        Cfg(tag="tau10", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_ms=10.0),
        Cfg(tag="ep150", epochs=150, val_mode="speaker", norm="none",
            in_scale=5.0),
        Cfg(tag="bn_scaled", epochs=60, val_mode="speaker", norm="bn",
            in_scale=5.0),
        Cfg(tag="rec_gain03", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, recurrent=True, rec_gain=0.3),
    ],
    # 第五轮：围绕 tauhet 做正则化与 τ 范围。r4 里 train 90.2% / test 70.3%，
    # 说明表示已经够用，剩下的差距是泛化。同时验证 τ 范围是否还能放宽。
    "r5": [
        Cfg(tag="het_wd1e2", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", wd=1e-2),
        Cfg(tag="het_ind20", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", in_dropout=0.2),
        Cfg(tag="het_drop30", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", dropout=0.3),
        Cfg(tag="het_hi40", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0),
        Cfg(tag="het_3x128", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", n_layers=3, hidden=128),
    ],
    # 第六轮：数据增强。r5 说明 dropout / wd 都动不了这个差距，
    # 而 Cramer 2020 报告增强带来 +12 点（远超任何正则项）。
    # 本轮单独扫频道平移幅度，这是唯一直接针对说话人不变性的手段。
    "r6": [
        Cfg(tag="het_chan10", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=10),
        Cfg(tag="het_chan25", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25),
        Cfg(tag="het_chan50", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=50),
        Cfg(tag="het_chan25_time5", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25, aug_time=5),
        Cfg(tag="het_chan25_drop10", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25, aug_drop=0.1),
    ],
    # 第七轮：在 chan25 之上继续。r6 证明增强是唯一有效的泛化手段
    # （chan10 72.4% -> chan25 74.0%，train 90% -> 80%），chan50 开始回落。
    # 本轮验证：读出积分长度、输入尺度、容量、以及两种附加增强。
    "r7": [
        Cfg(tag="chan25_aout90", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25,
            alpha_out=0.9),
        Cfg(tag="chan25_scale3", epochs=60, val_mode="speaker", norm="none",
            in_scale=3.0, tau_mode="het", tau_hi=40.0, aug_chan=25),
        Cfg(tag="chan25_h384", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25, hidden=384),
        Cfg(tag="chan25_time5", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25, aug_time=5),
        Cfg(tag="chan25_drop10", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25, aug_drop=0.1),
    ],
    # 第八轮：容量与训练长度。r7 里 h384 到 75.8%，而 train 只有 84%——
    # 增强把训练精度压下来之后，模型重新变成"没吃饱"，加大容量/加长训练
    # 才有意义（这与 r4 里无增强时加长训练直接过拟合到 100% 不同）。
    "r8": [
        Cfg(tag="h384_ep120", epochs=120, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25, hidden=384),
        Cfg(tag="h512", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25, hidden=512),
        Cfg(tag="h384_chan35", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=35, hidden=384),
        Cfg(tag="h384_drop20", epochs=60, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25, hidden=384,
            dropout=0.2),
        Cfg(tag="h384_scale8", epochs=60, val_mode="speaker", norm="none",
            in_scale=8.0, tau_mode="het", tau_hi=40.0, aug_chan=25, hidden=384),
    ],
    # 第九轮：把 r8 的两条增益（120 epoch、chan35）组合，并探两端的边界。
    "r9": [
        Cfg(tag="h384_ep120_chan35", epochs=120, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=35, hidden=384),
        Cfg(tag="h384_ep200", epochs=200, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25, hidden=384),
        Cfg(tag="h512_ep120", epochs=120, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=25, hidden=512),
        Cfg(tag="h384_ep120_chan50", epochs=120, val_mode="speaker", norm="none",
            in_scale=5.0, tau_mode="het", tau_hi=40.0, aug_chan=50, hidden=384),
    ],
    # 第十轮：定稿。r9 的最优配方 + 用满全部 7156 条训练数据（val_mode=all）。
    # 用 3 个 seed 给出误差棒——n=2264 时 95% CI 半宽约 1.0–1.4 个百分点，
    # 单次结果无法支撑"比其他变体好多少"这类判断。
    "r10": [
        Cfg(tag="final", epochs=120, val_mode="all", norm="none", in_scale=5.0,
            tau_mode="het", tau_lo=1.0, tau_hi=40.0, aug_chan=35, hidden=384),
    ],
    # M3：RQ1 主实验。五个变体在 M2 定稿配方下逐一对齐。
    # 统一项：2×384、无归一化、in_scale=5、层内异构 τ（1→40 帧，层间 ×2）、
    #         频道平移 ±35、120 epoch、batch 512、同一 seed 集。
    # 唯一变量是神经元方程。用 val_mode=all（7156 条全用、固定 120 epoch、
    # 取末轮权重），避免"早停轮次不同"这个额外混淆。
    "m3": [
        Cfg(tag=v, variant=v, epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384)
        for v in ("lif", "alif", "adlif", "glif", "mclif")
    ],
    # M3b：AdLIF 适应电流的机制消融。w 由两项驱动，来源不同：
    #   a·u[t-1]  膜电位驱动 → u→w→u 二阶反馈，产生 class-2 兴奋性（共振）
    #   b·s[t-1]  发放驱动   → 经典脉冲频率适应
    # 分别冻结即可分开两者的贡献。"none" 两支都断，应还原为 LIF，
    # 用作内部对照（同时验证 learn_a/learn_b 开关本身没引入行为改变）。
    # "both" 与 M3 的 adlif 完全同配置，重跑用于确认重构无影响。
    "m3b": [
        # "both" 与 "none" 不重跑：M3 的 adlif（86.07 ± 0.68）就是 both，
        # LIF（83.04 ± 0.63）就是 none（a=b=0 时 w 恒为 0，单元测试已逐元素验证）。
        Cfg(tag="a_only", variant="adlif", epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, learn_b=False),
        Cfg(tag="b_only", variant="adlif", epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, learn_a=False),
    ],
    "m4b": [
        # 1) AdLIF 稳定性重测：此前有一次同配置训练在 ep60-70 发散成 NaN，
        #    但那次与另一个进程并发。这里单独重跑，判断是偶发还是系统性。
        Cfg(tag="adlif_recheck", variant="adlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384),
        # 2) recurrent 复核：M2 里循环失败的直接症状是发放率爬到 40%+。
        #    加发放率正则直接针对该症状。1 seed 探针，不作为结论。
        Cfg(tag="rnn_lif_reg", variant="lif", epochs=100, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, recurrent=True, rec_gain=0.5,
            rate_target=0.12, rate_lambda=2.0),
        Cfg(tag="rnn_adlif_reg", variant="adlif", epochs=100, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, recurrent=True, rec_gain=0.5,
            rate_target=0.12, rate_lambda=2.0),
    ],
    # 归档用：重跑两个定稿模型并存权重。训练是逐位可复现的（实测 AdLIF
    # 同 seed 重跑得 85.29%，与首次完全一致），所以这里的角色是"补出实体产物"，
    # 不是重新选型。期望值与 M3/r10 一模一样。
    "final_w": [
        Cfg(tag="lif_w", variant="lif", epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, save_weights=True),
        Cfg(tag="adlif_w", variant="adlif", epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, save_weights=True),
    ],
    "r2": [
        Cfg(tag="thr100", thr=1.0),
        Cfg(tag="thr025", thr=0.25),
        Cfg(tag="smooth", label_smooth=0.1),
        Cfg(tag="wd5e3", wd=5e-3),
    ],
    # M2 第一项：收窄 τ 范围，检验「循环与异构 τ 功能重叠」这一解释。
    #
    # 预测（若解释成立）：τ 越窄，前馈精度越低，而循环带来的增益越大。
    # 若 τ_hi=5 时循环仍然无收益，解释被推翻，转 H2–H6。
    #
    # **rec_gain 必须按 τ 范围分别标定**（scripts/diagnose_h1.py 的实测二分）：
    #   τ_hi=5 → 6.75   τ_hi=10 → 8.1   τ_hi=40 → 12.3
    # 直接用 12 会把短 τ 格子过度驱动 1.8 倍——配对增益不是常数，因为 τ 变了
    # 发放率就变，s 的密度和循环电流幅值都跟着变。
    #
    # τ_hi=40 的两个格子已由 M1 测过（h0_strong_ff 83.04 / h0_strong_g12 83.13），
    # 此处只补 τ_hi=5 与 10。
    "m2tau": [
        Cfg(tag="tau10_ff", epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=10.0,
            aug_chan=35, hidden=384),
        Cfg(tag="tau10_rnn", epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=10.0,
            aug_chan=35, hidden=384, recurrent=True, rec_gain=8.1,
            save_weights=True),
        Cfg(tag="tau5_ff", epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=5.0,
            aug_chan=35, hidden=384),
        Cfg(tag="tau5_rnn", epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=5.0,
            aug_chan=35, hidden=384, recurrent=True, rec_gain=6.75,
            save_weights=True),
    ],
    # M2 第二项：E/I 平衡的循环初始化（H2）。
    #
    # 默认初始化的循环权重的**行和不为零**——它给整层一个共同偏置。M1 实测
    # gain=12 时发放率从 0.227/0.284 涨到 0.267/0.365，就是它在推网络饱和。
    # 行和归零后只剩涨落，理论上可以在不饱和的前提下加大循环强度。
    #
    # 配对增益要重标：行和归零去掉了均值分量，达到同幅值需 gain≈23
    # （default 只需 12，见 scripts/diagnose_h1.py 的实测二分）。
    #
    # 两个格子：gain=23 是配对强度；gain=12 与 M1 的 default g12 同增益，
    # 构成"固定增益、只换初始化"的 A/B。
    "m2ei": [
        Cfg(tag="ei_g23", epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, recurrent=True, rec_init="ei_balanced",
            rec_gain=23.0, save_weights=True),
        Cfg(tag="ei_g12", epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, recurrent=True, rec_init="ei_balanced",
            rec_gain=12.0, save_weights=True),
    ],
    # 阶段四 M2：2×2 消融（适应电流 × 两隔室），固定 τ=40。
    #
    # 四个格子：
    #   单隔室 + 无适应 = v4_lif_t40（M1 已有，74.32）
    #   单隔室 + 适应   = v4_adlif_t40（本轮）
    #   两隔室 + 无适应 = v4_tclif（M1 已有，80.26）
    #   两隔室 + 适应   = v4_atlif（本轮）
    #
    # 为什么全部固定 τ=40：TC-LIF 忽略 tau_ms，异构 τ 的配方在两隔室上不适用。
    # 四个格子必须同配方，否则交互效应里混进配方差异。
    "m2v4": [
        Cfg(tag="v4_adlif_t40", variant="adlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", tau_ms=40.0,
            hidden=384, save_weights=True),
        Cfg(tag="v4_atlif", variant="atlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", hidden=384,
            save_weights=True),
    ],
    # 阶段四 M2c：两个候选机制的正面对比。
    #
    # 选项 1（intrinsic_het）：把两隔室的**内在周期**做成逐神经元谱（8→80 帧），
    #   给网络补上「快」的那一端。M1/M2 里所有神经元共用一个 44 帧周期，
    #   短于 ~6 帧的瞬态无从表示——这是 2.8 点精度缺口最可能的来源。
    # 选项 3（slow_k）：外挂一个 k=64 的**非脉冲慢群体**，读顶层脉冲、
    #   以低秩投影调制第 0 层。这是阶段五「层级因果结构」的最小版本：
    #   慢变量不是长在同一个神经元里，而是独立成一个群体。
    #
    # 两档都跑 TCLIF 与 ATLIF，另加 LIF τ=40 对照——若慢群体也能救 LIF，
    # 那它就不是两隔室专属的补丁。
    "m3v4": [
        Cfg(tag="v4_tclif_het", variant="tclif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", hidden=384,
            intrinsic_het=True, save_weights=True),
        Cfg(tag="v4_atlif_het", variant="atlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", hidden=384,
            intrinsic_het=True, save_weights=True),
        Cfg(tag="v4_tclif_slow", variant="tclif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", hidden=384,
            slow_k=64, slow_tau=100.0, save_weights=True),
        Cfg(tag="v4_atlif_slow", variant="atlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", hidden=384,
            slow_k=64, slow_tau=100.0, save_weights=True),
        Cfg(tag="v4_lif_slow", variant="lif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", tau_ms=40.0,
            hidden=384, slow_k=64, slow_tau=100.0, save_weights=True),
    ],
    # 阶段四 M2b（探针，1 seed，不作结论）：饱和是不是 GSA 塌缩的可修复原因？
    #
    # AT-LIF 撤输入后仍以 0.415 的发放率活动，胞体长期在阈值之上——
    # 这正是 β=0.22 的直接解释。发放率正则直接针对这个症状：
    # 它把平均发放率往 `rate_target` 拉，等价于一个 homeostatic 阈值。
    # 若加上正则后 β 回到 [0.5,1.5]，说明 GSA 的失败是工作点问题而非结构问题；
    # 若 β 不动，说明两隔室本身就把幅度信息转成了直流偏置。
    "m2v4b": [
        Cfg(tag="v4_atlif_reg", variant="atlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", hidden=384,
            rate_target=0.15, rate_lambda=2.0, save_weights=True),
        Cfg(tag="v4_tclif_reg", variant="tclif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", hidden=384,
            rate_target=0.15, rate_lambda=2.0, save_weights=True),
    ],
    # 阶段四 M1：TC-LIF 单点 + 配对基线。
    #
    # 为什么基线必须是**固定 τ**：TC-LIF 没有衰减因子，`tau_ms` 对它无效，
    # 阶段三的「层内异构 τ」配方在两隔室变体上不适用。若拿异构 τ 的 LIF（83.04）
    # 直接比，比的是两个不同配方，无法把差异归给神经元。
    #
    # τ=40 的选择：TC-LIF 的静息旋转周期由 β1β2 决定（初始 44.4 帧），
    # 单 τ 的 LIF 取 τ=40 是最接近的「长记忆」对照。另跑 τ=2 作为短记忆下界。
    "m1v4": [
        Cfg(tag="v4_tclif", variant="tclif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", hidden=384,
            save_weights=True),
        Cfg(tag="v4_lif_t40", variant="lif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", tau_ms=40.0,
            hidden=384, save_weights=True),
        Cfg(tag="v4_lif_t2", variant="lif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="fixed", tau_ms=2.0,
            hidden=384, save_weights=True),
    ],
    # RQ-B：AdLIF 适应电流的机制归因，在**循环结构**下重做。
    #
    # 阶段二只在前馈下做过（m3b：a_only 84.91 / b_only 86.93 / both 86.07 /
    # none 83.04），结论是「增益来自发放驱动 b，膜电位驱动 a 无可测贡献」。
    # 但文献的机制解释（a·u 的二阶反馈产生共振）是在**循环**结构下提出的——
    # a 的共振特性可能与循环动力学耦合后才显现。
    #
    # 「都断」格直接复用 M1 的 h0_strong_g12（LIF + 循环，83.13 ± 0.69）：
    # AdLIF 在 a=b=0 时与 LIF 逐位等价（本机逐时间步验证过），故无需重跑。
    #
    # rec_gain 用 12.0：τ=40 下的实测标定区间是 [12.06, 12.52]，
    # 12.0 只低 0.5%，且与上面那个控制格完全一致——一致性比追标定中点更重要。
    # AdLIF 在初始化时 a=b=0、与 LIF 同分布，故该增益对三种消融在初始时刻
    # 是同一个「配平」定义；训练后的实际耦合比另用检查点单独测（见分析脚本）。
    "rqb": [
        Cfg(tag="rqb_a_only", variant="adlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, recurrent=True, rec_gain=12.0,
            learn_b=False, save_weights=True),
        Cfg(tag="rqb_b_only", variant="adlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, recurrent=True, rec_gain=12.0,
            learn_a=False, save_weights=True),
        Cfg(tag="rqb_both", variant="adlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, recurrent=True, rec_gain=12.0,
            save_weights=True),
    ],
    # RQ-B 的第二支：在 τ_hi=5 下重做「仅 a」这一格。
    #
    # 为什么需要：τ=40 的循环是**弱版本**——M2 实测循环只贡献 +0.09，
    # 结构上标着 recurrent，功能上等于前馈。而 τ=5 时循环贡献 +2.65，
    # 是唯一「循环动力学确实在起作用」的配置。若 `a·u` 的共振真的需要
    # 循环耦合，效应应当在 τ=5 下最明显。
    #
    # 只跑 a_only 两格（前馈 / 循环）：这是交互效应的分子与分母所在。
    # b_only 与 both 只在交互为正时才补——先花 1.1 小时拿到判别信号，
    # 而不是一上来跑 18 次。
    #
    # rec_gain=6.75 取自 τ=5 的实测配平区间 [6.53, 6.99]。
    # 「都断」两格已有：前馈 = tau5_ff（82.32）、循环 = tau5_rnn（84.97）。
    "rqb_tau5": [
        Cfg(tag="rqb5_ff_a_only", variant="adlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=5.0,
            aug_chan=35, hidden=384, learn_b=False),
        Cfg(tag="rqb5_rnn_a_only", variant="adlif", epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=5.0,
            aug_chan=35, hidden=384, recurrent=True, rec_gain=6.75,
            learn_b=False, save_weights=True),
    ],
    # M1 的前馈对照：H0 的每一格都要配一个同基线的前馈，否则测到的是
    # "基线配方的差异"而不是"循环的贡献"。强基线的前馈已有（r10 final，
    # 3 seed 且带 test_pred），只需补弱基线的。
    "h0ff": [
        Cfg(tag="h0_weak_ff", save_weights=True),
    ],
    # 强基线的前馈：r10 有 3 seed 的预测但没有权重，诊断仪器需要权重，
    # 补一个 seed 即可（诊断只做机制对比，不参与精度统计）。
    "h0ffs": [
        Cfg(tag="h0_strong_ff", save_weights=True, epochs=120, val_mode="all",
            norm="none", in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384),
    ],
    # M1 / H0：循环的收益是否依赖基线强度（2×2）
    #
    # 弱基线的定义必须固定：Cfg() 默认 = 2×256 / LayerNorm / in_scale=1 /
    # 固定 τ=2 帧 / 无增强 / 40 epoch / val_mode=speaker，阶段二实测 test 54.15%。
    # 强基线 = 阶段二定稿配方，test 83.04%。
    #
    # 为什么要有 rec_gain 这一维：M0 实测 rec_gain=1 时循环电流只有前馈的 3.7%，
    # 而"同量级"需 ≈12（in_scale=5 下实测二分）。只用 gain=1 会重犯阶段二的错误——
    # 拿一个几乎没接上的循环去论证"循环无用"。
    "h0": [
        Cfg(tag="h0_weak_g1", save_weights=True, recurrent=True, rec_gain=1.0),
        Cfg(tag="h0_weak_g12", save_weights=True, recurrent=True, rec_gain=12.0),
        Cfg(tag="h0_strong_g1", save_weights=True, epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, recurrent=True, rec_gain=1.0),
        Cfg(tag="h0_strong_g12", save_weights=True, epochs=120, val_mode="all", norm="none",
            in_scale=5.0, tau_mode="het", tau_lo=1.0, tau_hi=40.0,
            aug_chan=35, hidden=384, recurrent=True, rec_gain=12.0),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=sorted(GRIDS), default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--frames", type=int, default=0)
    for f in ("hidden", "n_layers", "thr", "tau_ms", "dropout", "in_dropout",
              "in_scale", "lr", "wd", "batch", "label_smooth"):
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(getattr(Cfg, f)),
                        default=None, dest=f)
    for f in ("norm", "readout", "variant"):
        ap.add_argument(f"--{f.replace('_', '-')}", default=None, dest=f)
    args = ap.parse_args()

    dev = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    d = build(ROOT / "data" / "shd", T=args.T)
    print(f"device={dev}  T={args.T}", flush=True)

    if args.grid:
        cfgs = [replace(c, epochs=args.epochs or c.epochs) for c in GRIDS[args.grid]]
    else:
        c = Cfg()
        for f in ("hidden", "n_layers", "thr", "tau_ms", "dropout", "in_dropout",
                  "in_scale", "lr", "wd", "batch", "label_smooth", "norm",
                  "readout", "variant", "frames"):
            v = getattr(args, f)
            if v is not None:
                setattr(c, f, v)
        if args.epochs:
            c.epochs = args.epochs
        c.tag = args.tag or "run"
        cfgs = [c]

    OUT.mkdir(exist_ok=True)
    tag = args.grid or args.tag or "run"
    path = OUT / f"train_shd_{tag}.json"

    def dump(rows):
        """每个分支跑完就落盘。整轮跑完才写会在中途崩溃时丢掉全部结果——
        r6/r7 各因 XPU 的 gather 断言崩溃，两轮的 JSON 都没留下。"""
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), "utf-8")

    rows = []
    for cfg in cfgs:
        for s in args.seeds:
            cfg = replace(cfg, seed=s)
            print(f"\n=== {cfg.tag} seed={s}  {cfg.n_layers}×{cfg.hidden} "
                  f"drop={cfg.dropout} indrop={cfg.in_dropout} "
                  f"readout={cfg.readout} norm={cfg.norm} thr={cfg.thr} ===", flush=True)
            r = run(cfg, d, dev)
            rows.append(r)
            dump(rows)
            print(f"  → val={r['val']:.4f} test={r['test']:.4f} "
                  f"train={r['final_train_acc']:.4f} 发放率={[round(v, 3) for v in r['rates']]} "
                  f"({r['seconds']:.0f}s)", flush=True)

    dump(rows)
    print(f"\n{'配方':22s}{'val':>8s}{'test':>8s}{'train':>8s}  发放率")
    for r in rows:
        print(f"{r['tag']:22s}{r['val']:>8.4f}{r['test']:>8.4f}"
              f"{r['final_train_acc']:>8.4f}  {[round(v, 3) for v in r['rates']]}")
    print(f"\n→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
