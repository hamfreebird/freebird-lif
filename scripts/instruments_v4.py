"""阶段四的四个仪器：ITC / GSA / PCI / Φ_G。

每个仪器都先定义清楚再测（PLAN-v4 §4），并且都带 **metric-level 自检**——
用已知答案的合成输入验证实现，而不是直接拿网络结果当真理。

    ITC 内在时间连续性：撤去输入后是否仍有演化、相关长度多长、历史可解码多少
    GSA 梯度对输入幅值的敏感性：β = dlog‖∂y/∂x₀‖ / dlog α
    PCI 扰动复杂度指数：扰动后响应的归一化 LZ 复杂度
    Φ_G 几何整合信息：4 群组高斯 AR(1) 的最小二分 KL

用法：
    python scripts/instruments_v4.py sanity
    python scripts/instruments_v4.py all --ckpt runs/lif_s0.pt
    python scripts/instruments_v4.py gsa --variant lif --tau-hi 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_shd import Cfg, SHDNet  # noqa: E402


# ---------------------------------------------------------------------------
# 通用：模型与追踪
# ---------------------------------------------------------------------------

def make_model(args, cfg_over: dict | None = None) -> tuple[SHDNet, Cfg]:
    over = dict(variant=args.variant, hidden=args.hidden, n_layers=2,
                norm="none", in_scale=args.in_scale, tau_mode="het",
                tau_lo=1.0, tau_hi=args.tau_hi, thr=0.5,
                n_in=700, n_classes=20, frames=0)
    if cfg_over:
        over.update(cfg_over)
    if args.ckpt:
        # 从检查点自带的 config 重建：不能按命令行默认值构造，
        # 否则加载 AdLIF 权重时会缺 beta_w/a/b（实测报 Unexpected key）。
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = Cfg(**blob["config"])
        model = SHDNet(cfg)
        model.load_state_dict(blob["state_dict"])
    else:
        cfg = Cfg(**over)
        model = SHDNet(cfg)
    model.eval()
    return model, cfg


class Tap:
    """抓每层每步的胞体膜电位与脉冲。

    与阶段三的 SpikeTap 同一套路：包装实例的 `step`，不用 forward hook
    （神经元是方法调用，hook 不会触发）。退出时必须恢复。
    """

    def __init__(self, model: SHDNet):
        self.soma = [[] for _ in model.neurons]
        self.spk = [[] for _ in model.neurons]
        self._orig = []
        for i, ne in enumerate(model.neurons):
            self._orig.append(ne.step)
            ne.step = self._mk(i, ne, ne.step)

    def _mk(self, i, ne, orig):
        def step(x, state):
            s, st = orig(x, state)
            self.soma[i].append(st[ne.soma_index].detach())
            self.spk[i].append(s.detach())
            return s, st
        return step

    def close(self, model: SHDNet):
        for ne, orig in zip(model.neurons, self._orig):
            ne.step = orig


@torch.no_grad()
def trace(model: SHDNet, x: torch.Tensor) -> tuple[list, list]:
    """返回 (soma[L] = [B,T,N], spike[L] = [B,T,N])。"""
    tap = Tap(model)
    try:
        model(x)
    finally:
        tap.close(model)
    n = model.cfg.n_layers
    soma = [torch.stack(tap.soma[i], 1) for i in range(n)]
    spk = [torch.stack(tap.spk[i], 1) for i in range(n)]
    return soma, spk


_CACHE: dict = {}


def shd(T: int = 100, n: int = 256):
    if "shd" not in _CACHE:
        from freebird_lif.shd.data import build
        _CACHE["shd"] = build(ROOT / "data" / "shd", T=T)
    d = _CACHE["shd"]
    return d["x_test"][:n], d["y_test"][:n]


# ---------------------------------------------------------------------------
# ITC：内在时间连续性
# ---------------------------------------------------------------------------

def tau_int(rate: np.ndarray) -> float:
    """自相关积分尺度（帧）：1 + 2·Σρ(k)，截断到 ρ 首次为负。白噪声给 1。"""
    r = rate - rate.mean()
    if r.std() < 1e-12:
        return 0.0
    n = len(r)
    ac = np.correlate(r, r, mode="full")[n - 1:]
    ac = ac / ac[0]
    s = 0.0
    for k in range(1, n):
        if ac[k] <= 0:
            break
        s += ac[k]
    return float(1.0 + 2.0 * s)


def participation_ratio(x: np.ndarray) -> float:
    """(Σλ)² / Σλ²。x: [样本, 特征]。塌到 1 维 ≈1。"""
    x = x - x.mean(0, keepdims=True)
    # 用特征维的协方差；样本数远大于特征数时才稳定，故先转置判断
    if x.shape[0] < x.shape[1]:
        cov = x @ x.T / max(x.shape[0] - 1, 1)
    else:
        cov = x.T @ x / max(x.shape[0] - 1, 1)
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    if lam.sum() <= 0:
        return 0.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def _ridge_r2(Xtr, ytr, Xte, yte, lam_frac: float = 1e-2, n_pc: int = 32) -> float:
    # 先降维：特征 384 维而训练样本只有 ~1600 个时间步，直接回归会过拟合到 R²=-10。
    # PCA 只在训练集上拟合，避免泄漏。
    mu = Xtr.mean(0, keepdims=True)
    Xc = Xtr - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:min(n_pc, Vt.shape[0])].T
    Xtr = Xc @ V
    Xte = (Xte - mu) @ V
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    lam = lam_frac * np.trace(A.T @ A) / A.shape[1]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ w
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def decode_history(soma_l: np.ndarray, x_in: np.ndarray, k: int,
                   t_max: int | None = None) -> float:
    """从 t 时刻的状态解码 t-k 时刻的输入强度（活跃通道数）。

    **只在输入非零的窗口内解码**。第一版用了整段 T，而 t>t_in 之后输入被置零，
    测试半段的标签方差≈0，R² 分母趋零，一个随机网络会给出 −14 这种无意义的数。
    """
    B, T, N = soma_l.shape
    t_max = min(t_max or T, T)
    idx = np.arange(k, t_max)                     # 这些 t 的 t-k 落在输入区间内
    feats = soma_l[:, idx, :].transpose(1, 0, 2).reshape(len(idx) * B, N)
    tgt = x_in[:, idx - k, :].sum(-1).transpose(1, 0).reshape(-1)
    half = len(idx) // 2
    if half < 3:
        return float("nan")
    tr = np.arange(half * B)
    te = np.arange(half * B, len(idx) * B)
    return _ridge_r2(feats[tr], tgt[tr], feats[te], tgt[te])


def _dominant_freq(rate: np.ndarray) -> float:
    r = rate - rate.mean()
    if r.std() < 1e-12:
        return 0.0
    spec = np.abs(np.fft.rfft(r))
    spec[0] = 0
    k = int(np.argmax(spec))
    return float(k / len(r))


def _traj_ratio(states: np.ndarray) -> float:
    """首尾距离 / 最大成对距离。塌到不动点时 → 0。states: [T, N]。"""
    if len(states) < 3:
        return 0.0
    d = np.linalg.norm(states[-1] - states[0])
    dd = np.linalg.norm(states[:, None, :] - states[None, :, :], axis=-1)
    return float(d / max(dd.max(), 1e-12))


def itc(model: SHDNet, x: torch.Tensor, t_in: int = 50, spont: int = 250) -> dict:
    """撤去输入后的连续性：输入只给前 `t_in` 帧，之后**拉长**到 `spont` 帧全零。

    为什么必须拉长：阶段三的配方里 τ 最大到 40 帧、层间 ×2 → 80 帧。
    若只看 50 帧，LIF 靠膜电位惯性还能维持 0.15–0.33 的发放率，
    会「靠惯性通过」ITC-1。拉到 250 帧后，LIF 衰减到 exp(-250/80)≈4%，
    而真正的中性稳定振荡不会衰减——这才是两条曲线的分界。
    """
    B, _, C = x.shape
    T = t_in + spont
    xm = torch.zeros(B, T, C)
    xm[:, :t_in] = x[:, :t_in]
    soma, spk = trace(model, xm)
    x_np = xm.numpy()
    out = {}
    for i in range(len(soma)):
        r = spk[i].mean(dim=(0, 2)).numpy()            # [T]
        # 「之后」只看后半段：跳过刚撤输入时的惯性衰减
        after_slice = slice(t_in + spont // 2, T)
        during = float(r[:t_in].mean())
        after = float(r[after_slice].mean())
        # 稳态窗口：跳过刚撤输入时的惯性衰减。tau_int 若在含衰减段的整段上算，
        # 趋势会把自相关抬高（实测 LIF 从 12 变 43 帧），失去判别力。
        rate_sp = r[after_slice]
        s_l = soma[i].numpy()                          # [B,T,N]
        spont_states = s_l[:, t_in + spont // 2:, :].reshape(-1, s_l.shape[-1])
        out[f"layer{i}"] = {
            "rate_during": during,
            "rate_after": after,
            "sustain_ratio": float(after / max(during, 1e-9)),
            "tau_int": tau_int(rate_sp),
            # 负 R² 表示「不如直接用均值」，按 0 处理（无信息）。截断是保守的：
            # 它只会让基线显得更好，不会让新架构显得更好。
            "decode_r2_k30": max(0.0, decode_history(s_l, x_np, k=30, t_max=t_in)),
            "participation_ratio": participation_ratio(spont_states),
            "dominant_freq": _dominant_freq(rate_sp),
            "spont_frames": int(len(rate_sp)),
        }
    return out


# ---------------------------------------------------------------------------
# GSA
# ---------------------------------------------------------------------------

def gsa(model: SHDNet, x: torch.Tensor, y: torch.Tensor,
        alphas=(0.25, 0.5, 1.0, 2.0, 4.0)) -> dict:
    gs = []
    for a in alphas:
        x0 = x.clone().requires_grad_(True)
        logits = model(x0 * a)
        val = logits.gather(1, y[:, None]).mean()
        g = torch.autograd.grad(val, x0)[0][:, 0, :]
        gs.append(float(g.norm()))
    gs_np = np.asarray(gs)
    la = np.log(np.asarray(alphas))
    lg = np.log(np.maximum(gs_np, 1e-20))
    beta = float(np.polyfit(la, lg, 1)[0])
    return {"alphas": list(alphas),
            "grad_norm": [float(v) for v in gs_np],
            "beta": beta, "ratio_4_1": float(gs_np[-1] / max(gs_np[2], 1e-20))}


# ---------------------------------------------------------------------------
# PCI
# ---------------------------------------------------------------------------

def lz76(seq: str) -> int:
    """LZ76（Kaspar–Schuster）。全零串给 2，随机串约 n/log2(n)。"""
    n = len(seq)
    if n == 0:
        return 0
    u, v, w = 0, 1, 1
    v_max, c = 1, 1
    while True:
        if u + v - 1 >= n or w + v - 1 >= n:
            c += 1
            break
        if seq[u + v - 1] == seq[w + v - 1]:
            v += 1
            if w + v > n:
                c += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                c += 1
                w += v_max
                if w > n:
                    break
                u, v, v_max = 0, 1, 1
            else:
                v = 1
    return c


def _bits(mat: np.ndarray) -> str:
    return "".join("1" if v else "0" for v in np.asarray(mat).reshape(-1))


def pci_score(mat: np.ndarray) -> float:
    """归一化 LZ：LZ76 / (n/log2 n)。按时间优先展平（行=时间，列=神经元）。"""
    n = mat.size
    if n == 0:
        return 0.0
    return float(lz76(_bits(mat)) / (n / math.log2(max(n, 2))))


@torch.no_grad()
def pci(model: SHDNet, x: torch.Tensor, t0: int = 20, t_resp: int = 60,
        frac: float = 0.10, amp: float = 3.0, trials: int = 20,
        thresh: float = 3.0, abs_thresh: float | None = None,
        seed: int = 0) -> dict:
    """扰动第 0 层 `frac` 的神经元，测**扰动诱发**响应的归一化 LZ。

    关键设计：响应定义为 `|扰动轨迹 − 无扰动轨迹|`，而不是「相对扰动前均值的 z-score」。
    后者的致命问题在第一版里暴露：对持续振荡的网络，扰动前的 20 帧均值根本代表不了
    扰动时刻的相位，几乎所有神经元都被判为「响应」——TCLIF 的密度 0.669、
    PCI 0.536，其中 0.528 来自它自己的振荡，扰动只贡献 0.008。
    减去无扰动轨迹可以把持续活动完全抵消（模型在 eval 下是确定性的），
    剩下的才是扰动传播。

    归一化用**扰动前窗口的时间标准差**（逐神经元），使不同量级的膜电位可比。
    """
    g = torch.Generator().manual_seed(seed)
    N = model.cfg.hidden

    def _traj(perturb) -> torch.Tensor:
        model.perturb = perturb
        try:
            soma, _ = trace(model, x)
        finally:
            model.perturb = None
        return soma[0]                                   # [B,T,N]

    base = _traj(None)
    # 阈值口径：
    #   sigma 模式用扰动前窗口的时间标准差归一化。**对训练后的网络不适用**——
    #   那个标准差里包含网络自身的大幅波动（发放率 0.4 时膜电位起伏很大），
    #   3σ 等于把门槛抬到不可达（实测 amp=100 时密度仍只有 0.0026）。
    #   模型在 eval 下是确定性的，没有试验间噪声，所以用**绝对阈值**更合适：
    #   膜电位相对无扰动轨迹移动超过 `abs_thresh`（默认 10% 阈值）即算响应。
    denom = (torch.ones(1, 1, 1) if abs_thresh is not None
             else base[:, :t0].std(dim=1, keepdim=True).clamp_min(1e-6))
    cut = abs_thresh if abs_thresh is not None else thresh

    scores, dens = [], []
    for _ in range(trials):
        mask = torch.zeros(N)
        idx = torch.randperm(N, generator=g)[: max(1, int(N * frac))]
        mask[idx] = 1.0
        s0 = _traj({"t": t0, "layer": 0, "mask": mask, "amp": amp})
        diff = (s0[:, t0:t0 + t_resp] - base[:, t0:t0 + t_resp]).abs() / denom
        frac_resp = (diff > cut).float().mean(dim=0).numpy()       # [t_resp, N]
        binm = frac_resp > 0.2
        scores.append(pci_score(binm))
        dens.append(float(binm.mean()))
    return {"pci_mean": float(np.mean(scores)), "pci_sd": float(np.std(scores)),
            "pci_baseline": 0.0,          # 由构造：无扰动时 diff 恒为 0
            "pci_delta": float(np.mean(scores)),
            "density_mean": float(np.mean(dens)),
            "trials": trials,
            "threshold": cut if abs_thresh is not None else thresh,
            "thresh_mode": "abs" if abs_thresh is not None else "sigma",
            "n_neurons": N, "t_resp": t_resp, "amp": amp}


@torch.no_grad()
def pci_matched(model: SHDNet, x: torch.Tensor, t0: int = 20, t_resp: int = 60,
                frac: float = 0.10, amp: float = 10.0, trials: int = 20,
                target_density: float = 0.10, n_null: int = 5,
                n_time: int = 30, n_groups: int = 96,
                seed: int = 0) -> dict:
    """**密度匹配**的 PCI + 置换零模型。

    为什么要密度匹配：M2c 里慢群体的 PCI 0.157 完全由响应密度 0.72 撑起来，
    与「整合」无关。只要阈值固定，不同架构的响应密度天然不同（0.099 vs 0.732），
    PCI 就成了密度的单调函数。这里改为**按分位数取阈值**，强制每个配置的
    二值矩阵密度都等于 `target_density`，再比 LZ 复杂度。

    零模型：把二值矩阵原地打乱（保持 1 的个数不变），重算 LZ。
    若观测值落在零分布内（|z| < 2），说明「结构」与同密度的随机矩阵没有区别，
    这个 PCI 就没有解释力。

    计算取舍：LZ 在 23k 位上每次约 3 秒，20 次扰动 × 21 个置换要 20 分钟/模型。
    这里把响应矩阵**粗粒度**到 `n_time × n_groups`（默认 30×96，即时间 2 帧合并、
    每 4 个神经元合并）再算，代价约 0.3 秒/次。粗粒度对所有配置一致，
    不改变组间比较的方向；但它会抹掉细粒度结构，报告里必须写明。

    返回：PCI（密度匹配后）、z、以及未匹配的原始 PCI 供对照。
    """
    g = torch.Generator().manual_seed(seed)
    N = model.cfg.hidden

    def _traj(perturb):
        model.perturb = perturb
        try:
            soma, _ = trace(model, x)
        finally:
            model.perturb = None
        return soma[0]

    base = _traj(None)
    denom = base[:, :t0].std(dim=1, keepdim=True).clamp_min(1e-6)

    pcis, pcis_fixed, dens_fixed, zs = [], [], [], []
    for _ in range(trials):
        mask = torch.zeros(N)
        idx = torch.randperm(N, generator=g)[: max(1, int(N * frac))]
        mask[idx] = 1.0
        s0 = _traj({"t": t0, "layer": 0, "mask": mask, "amp": amp})
        # 先对 batch 取均值：否则 LZ 会在 32×60×384 = 73 万位上跑，慢 30 倍以上。
        # 这一步也把「跨样本一致性」纳入了响应强度：只有多数样本都响应的位置才算强。
        diff = ((s0[:, t0:t0 + t_resp] - base[:, t0:t0 + t_resp])
                .abs() / denom).mean(dim=0).numpy()          # [T_resp, N]
        pcis_fixed.append(pci_score(diff > 0.05))
        dens_fixed.append(float((diff > 0.05).mean()))
        # 粗粒度：时间与神经元都做块平均，保证尺寸可整除
        T_resp, N_neur = diff.shape
        bt, bn = T_resp // n_time, N_neur // n_groups
        coarse = diff[:n_time * bt, :n_groups * bn].reshape(
            n_time, bt, n_groups, bn).mean(axis=(1, 3))
        # 密度匹配：按分位数取阈值
        thr = float(np.quantile(coarse, 1.0 - target_density))
        binm = coarse > thr
        obs = pci_score(binm)
        pcis.append(obs)
        # 置换零模型
        flat = binm.reshape(-1).copy()
        null = []
        rng = np.random.default_rng(seed + _)
        for _j in range(n_null):
            rng.shuffle(flat)
            null.append(pci_score(flat.reshape(binm.shape)))
        null = np.asarray(null)
        zs.append(float((obs - null.mean()) / max(null.std(), 1e-9)))
    return {"pci_matched": float(np.mean(pcis)),
            "pci_matched_sd": float(np.std(pcis)),
            "target_density": target_density,
            "pci_fixed_thresh": float(np.mean(pcis_fixed)),
            "density_fixed_thresh": float(np.mean(dens_fixed)),
            "null_z": float(np.mean(zs)),
            "trials": trials, "n_null": n_null}


def pci_synthetic_battery(target_density: float = 0.10, seed: int = 0,
                          n_time: int = 30, n_groups: int = 96) -> dict:
    """metric-level 效度检验：同密度、同分辨率下，PCI 能否把「结构」与「随机」分开。

    分辨率必须与 pci_matched 一致（默认 30×96），否则两边的归一化不可比——
    第一版在 60×384 上算，模型那边在 30×96 上算，等于拿两把尺子。

    四种**按密度直接构造**的矩阵（2304→288 个 1）：
      sync   3 个整行全亮（全局同步、时间上完全刻板）
      blob   12×24 的连续块（局部、无传播）
      wave   宽度 10 的斜带（沿空间传播的时空结构）
      random 同密度均匀随机

    判据：random > wave > max(blob, sync)，且 random − max(blob, sync) > 0.10。
    **不要求**把 blob 与 sync 分开——两者都是「刻板响应」，
    指标只需要识别「有结构 vs 无结构」。
    """
    rng = np.random.default_rng(seed)
    T, N = n_time, n_groups
    n_on = int(round(T * N * target_density))      # 288

    sync = np.zeros((T, N), bool)
    sync[np.linspace(0, T - 1, n_on // N, dtype=int), :] = True

    blob = np.zeros((T, N), bool)
    blob[6:6 + 12, 30:54] = True                   # 12×24 = 288

    wave = np.zeros((T, N), bool)
    center = (np.arange(T) * 3 + 20) % N
    nn = np.arange(N)[None, :]
    wave = (np.abs(((nn - center[:, None] + N // 2) % N) - N // 2) < 5)

    flat = np.zeros(T * N, bool)
    flat[rng.choice(T * N, n_on, replace=False)] = True
    rand = flat.reshape(T, N)

    mats = {"sync": sync, "blob": blob, "wave": wave, "random": rand}
    out = {k: {"pci": pci_score(m), "density": float(m.mean())} for k, m in mats.items()}
    floor = max(out["blob"]["pci"], out["sync"]["pci"])
    ok = (out["random"]["pci"] > out["wave"]["pci"] > floor
          and out["random"]["pci"] - floor > 0.10)
    return {"battery": out, "pass": bool(ok),
            "target_density": target_density, "shape": [T, N]}


# ---------------------------------------------------------------------------
# Φ_G
# ---------------------------------------------------------------------------

def _stable(A: np.ndarray, rho_max: float = 0.99) -> float:
    """把谱半径压到 rho_max 以下，返回原谱半径。"""
    rho = float(np.max(np.abs(np.linalg.eigvals(A)))) if A.size else 0.0
    if np.isfinite(rho) and rho > rho_max:
        A *= rho_max / rho
    return rho


def _lyap(A: np.ndarray, S: np.ndarray, iters: int = 300) -> np.ndarray:
    """离散 Lyapunov 迭代，带收敛早停与有限性检查。"""
    X = np.eye(A.shape[0])
    for _ in range(iters):
        Xn = A @ X @ A.T + S
        if not np.all(np.isfinite(Xn)):
            return np.full_like(X, np.nan)
        if np.max(np.abs(Xn - X)) < 1e-10 * max(np.max(np.abs(Xn)), 1e-12):
            return Xn
        X = Xn
    return X


def _kl_gauss(S1: np.ndarray, S2: np.ndarray) -> float:
    k = S1.shape[0]
    S2i = np.linalg.inv(S2)
    ld1 = np.linalg.slogdet(S1)[1]
    ld2 = np.linalg.slogdet(S2)[1]
    return float(0.5 * (np.trace(S2i @ S1) - k + ld2 - ld1))


def phi_g(soma_layer: np.ndarray, k: int = 4, seed: int = 0) -> dict:
    """soma_layer: [B, T, N] → 组活动 [T, k] → 高斯 AR(1) → 最小二分 KL。

    k=4 是样本量妥协：4 群组有 7 个二分划分，AR(1) 参数估计稳定。
    k=8 的 127 个划分会让每个群组的残差协方差估计变噪。报告里必须写明。
    """
    from numpy.linalg import lstsq

    B, T, N = soma_layer.shape
    # 聚类对象是**神经元**，特征是它的时间剖面：prof[N, T*B]
    # 第一版写成了 [T*B, N]，结果把每个 (batch, t) 当成一个神经元，
    # lab 长度变成 3200，索引 N=384 的层状态时越界。
    prof = soma_layer.transpose(2, 0, 1).reshape(N, T * B)
    prof = prof - prof.mean(1, keepdims=True)
    rng = np.random.default_rng(seed)
    centers = prof[rng.choice(N, k, replace=False)]
    lab = np.zeros(N, dtype=int)
    for _ in range(30):
        d = ((prof[:, None, :] - centers[None, :, :]) ** 2).sum(-1)   # [N, k]
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                centers[j] = prof[lab == j].mean(0)
    # 组活动：对 batch 与成员取均值 → [T, k]
    X = np.stack([soma_layer[:, :, lab == j].mean(axis=(0, 2)) for j in range(k)], 1)
    # 必须 z-score：Φ_G 依赖变量尺度，不做归一化就变成在比"谁家膜电位大"。
    sd = X.std(0, keepdims=True)
    X = (X - X.mean(0, keepdims=True)) / np.clip(sd, 1e-6, None)
    Xc = X - X.mean(0, keepdims=True)
    A, *_ = lstsq(np.hstack([Xc[:-1], np.ones((T - 1, 1))]), Xc[1:], rcond=None)
    A = A[:k].T
    pred = np.hstack([Xc[:-1], np.ones((T - 1, 1))]) @ np.vstack([A.T, np.zeros((1, k))])
    E = Xc[1:] - pred
    S = np.cov(E, rowvar=False) + 1e-6 * np.eye(k)
    # 近单位根会让 Lyapunov 迭代发散，Φ 变成数值垃圾。把谱半径压到 0.99 以下，
    # 这是对「有耗散」的显式假设，而不是让它悄悄发散。
    rho = _stable(A)
    if rho > 0.99:
        E = Xc[1:] - np.hstack([Xc[:-1], np.ones((T - 1, 1))]) @ np.vstack(
            [A.T, np.zeros((1, k))])
        S = np.cov(E, rowvar=False) + 1e-6 * np.eye(k)
    SX = _lyap(A, S)
    best = np.inf
    for mask in range(1, 2 ** (k - 1)):
        left = [i for i in range(k) if (mask >> i) & 1]
        right = [i for i in range(k) if i not in left]
        if not left or not right:
            continue
        Ap = A.copy()
        Ap[np.ix_(left, right)] = 0
        Ap[np.ix_(right, left)] = 0
        Sp = S.copy()
        for i in left:
            for j in right:
                Sp[i, j] = Sp[j, i] = 0
        _stable(Ap)
        try:
            v = _kl_gauss(SX, _lyap(Ap, Sp))
            if np.isfinite(v):
                best = min(best, v)
        except np.linalg.LinAlgError:
            continue
    return {"phi_g": float(best), "k": k, "T": int(T), "n_neurons": int(N),
            "spectral_radius": rho,
            "group_sizes": [int((lab == j).sum()) for j in range(k)]}


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def sanity() -> dict:
    """metric-level 自检：已知答案的合成输入。"""
    rng = np.random.default_rng(0)
    mats = {
        "全零": np.zeros((32, 40), bool),
        "全一": np.ones((32, 40), bool),
        "随机 10%": rng.random((32, 40)) < 0.1,
        "随机 50%": rng.random((32, 40)) < 0.5,
        "分块（前 10 行全 1）": np.repeat(np.arange(32)[:, None] < 10, 40, 1),
    }
    pci = {k: pci_score(v) for k, v in mats.items()}
    lz = {k: lz76(_bits(v)) for k, v in mats.items()}
    white = rng.normal(size=4000)
    slow = np.cumsum(rng.normal(size=4000)) * 0.01 + rng.normal(scale=0.1, size=4000)
    ti = {"白噪声": tau_int(white), "随机游走": tau_int(slow)}
    ok = (pci["全零"] < 0.05 and pci["全一"] < 0.25
          and pci["随机 50%"] > pci["分块（前 10 行全 1）"]
          and ti["随机游走"] > 3 * ti["白噪声"])
    return {"pci": pci, "lz76": lz, "tau_int": ti, "pass": bool(ok)}


def main() -> int:
    ap = argparse.ArgumentParser()
    # 允许 `... sanity` 与 `--what sanity` 两种写法
    ap.add_argument("command", nargs="?", default=None,
                    choices=["sanity", "itc", "gsa", "pci", "pci2", "phi", "all"])
    ap.add_argument("--what", default="sanity",
                    choices=["sanity", "itc", "gsa", "pci", "pci2", "phi", "all"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--variant", default="lif")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--tau-hi", type=float, default=40.0)
    ap.add_argument("--in-scale", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--amp", type=float, default=10.0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--abs-thresh", type=float, default=0.05,
                    help="膜电位移动的绝对阈值（默认 0.05 = 10% 阈值）；设为负则用 sigma 模式")
    ap.add_argument("--target-density", type=float, default=0.10,
                    help="pci2 的密度匹配目标")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.command:
        args.what = args.command

    if args.what == "sanity":
        res = sanity()
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["pass"] else 1

    x_np, y_np = shd(n=args.batch)
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).long()
    model, _ = make_model(args)
    res = {"ckpt": args.ckpt, "variant": args.variant,
           "tau_hi": args.tau_hi, "in_scale": args.in_scale}
    if args.what in ("itc", "all"):
        res["itc"] = itc(model, x)
    if args.what in ("gsa", "all"):
        res["gsa"] = gsa(model, x, y)
    if args.what == "pci2":
        res["pci_matched"] = pci_matched(model, x, amp=args.amp,
                                         target_density=args.target_density)
    if args.what in ("pci", "all"):
        res["pci"] = pci(model, x, amp=args.amp, trials=args.trials,
                         abs_thresh=(None if args.abs_thresh < 0 else args.abs_thresh))
    if args.what in ("phi", "all"):
        soma, _ = trace(model, x)
        res["phi"] = phi_g(soma[-1].numpy(), k=4)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
