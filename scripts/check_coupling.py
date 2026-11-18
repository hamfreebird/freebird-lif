"""测量训练后模型的实际「循环 / 前馈」耦合比。

为什么需要这个：RQ-B 的三种消融（仅 a / 仅 b / 都开）在初始化时 a=b=0，
与 LIF 完全同分布，所以 `rec_gain=12.0` 是在**同一初始耦合**下配平的。
但训练之后 a/b 会让发放率分叉，`s` 的密度一变，实际耦合就不再等于初始值。
若三个格子的实际耦合差得多，"消融"就同时改变了「机制」和「耦合强度」两件事。

于是这里在**真实数据**上、对**训练后的权重**重测一次比值，作为操纵检查
（manipulation check）。用 forward hook 抓 `norms[i]` 与 `recs[i]` 的输出，
不复制模型的前向逻辑——复制必然漂移。

用法：
    python scripts/check_coupling.py runs/rqb_a_only_s0.pt [更多检查点...]
    # 输出 benchmarks/rqb_coupling.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_shd import Cfg, SHDNet  # noqa: E402

OUT = ROOT / "benchmarks" / "rqb_coupling.json"


@torch.no_grad()
def measure(ckpt: Path, batch: int = 128) -> dict:
    # torch 2.6 起 weights_only 默认为 True，而检查点里含 numpy 数组（test_pred），
    # 会抛 UnpicklingError。这些文件是本项目自己产出的，允许完整反序列化。
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = Cfg(**blob["config"])
    model = SHDNet(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    from freebird_lif.shd.data import build

    d = build(ROOT / "data" / "shd", T=100)
    # 不要在这里乘 in_scale：SHDNet.forward 内部已经做过 `h = x * in_scale`。
    # 外面再乘一次会把前馈电流放大 in_scale 倍，比值被系统性缩小。
    x = torch.from_numpy(d["x_test"][:batch]).float()

    base: list[list[float]] = [[] for _ in range(cfg.n_layers)]
    rec: list[list[float]] = [[] for _ in range(cfg.n_layers)]
    handles = []

    def mk(sink):
        def hook(_m, _inp, out):
            sink.append(float(out.abs().mean()))
        return hook

    for i in range(cfg.n_layers):
        handles.append(model.norms[i].register_forward_hook(mk(base[i])))
        handles.append(model.recs[i].register_forward_hook(mk(rec[i])))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()

    per_layer = []
    for i in range(cfg.n_layers):
        # rec 比 base 少一次（第 0 步没有 prev），对齐到同一批时间步
        n = len(rec[i])
        b = base[i][len(base[i]) - n:] if n else []
        per_layer.append({
            "layer": i,
            "base_mean": sum(b) / max(len(b), 1),
            "rec_mean": sum(rec[i]) / max(n, 1),
            "ratio": (sum(rec[i]) / n) / max(sum(b) / max(len(b), 1), 1e-9) if n else float("nan"),
            "n_steps": n,
        })

    return {"ckpt": ckpt.name, "tag": cfg.tag, "seed": cfg.seed,
            "variant": cfg.variant, "tau_hi": cfg.tau_hi,
            "rec_gain": cfg.rec_gain, "layers": per_layer,
            "ratio_mean": sum(p["ratio"] for p in per_layer) / len(per_layer),
            "ab": ab_stats(model, cfg)}


def ab_stats(model: SHDNet, cfg: Cfg) -> list[dict] | None:
    """读训练后的适应电流参数 a、b。

    这一步回答一个消融回答不了的问题：**`a` 到底有没有被训练用起来**。
    如果 a_only 组里 |a| 仍然接近 0，那「a 无可测贡献」就不只是消融结论，
    而是「这个参数在训练中没有获得梯度信号」——两者的含义完全不同：
    前者说明机制无用，后者说明优化没有找到它。
    """
    if cfg.variant != "adlif":
        return None
    out = []
    for i, ne in enumerate(model.neurons):
        a = ne.a.detach().float()
        b = ne.b.detach().float()
        out.append({
            "layer": i,
            "a_mean": float(a.mean()), "a_absmean": float(a.abs().mean()),
            "b_mean": float(b.mean()), "b_absmean": float(b.abs().mean()),
            "a_frac_gt_1e-3": float((a.abs() > 1e-3).float().mean()),
            "b_frac_gt_1e-3": float((b.abs() > 1e-3).float().mean()),
        })
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    out = []
    for a in argv:
        r = measure(Path(a))
        out.append(r)
        line = (f"{r['ckpt']:26s} {r['variant']:6s} ratio="
                + " / ".join(f"{p['ratio']:.3f}" for p in r["layers"])
                + f"  mean={r['ratio_mean']:.3f}")
        if r.get("ab"):
            line += ("  |a|=" + "/".join(f"{q['a_absmean']:.4f}" for q in r["ab"])
                     + "  |b|=" + "/".join(f"{q['b_absmean']:.4f}" for q in r["ab"]))
        print(line)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
