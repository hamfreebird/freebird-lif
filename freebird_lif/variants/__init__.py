"""LIF 变体库。

统一接口见 base.SpikingNeuron；六个变体中 POLARA 因原文不可获取而阻塞，
其余五个可用。
"""

from .base import SpikingNeuron, decay_from_tau
from .neurons import ALIF, GLIF, LIF, AdLIF, MCLIF
from .twocomp import ATLIF, TCLIF
from .polara import POLARA

REGISTRY = {
    "lif": LIF,
    "alif": ALIF,
    "adlif": AdLIF,
    "glif": GLIF,
    "mclif": MCLIF,
    # 阶段四：两隔室
    "tclif": TCLIF,
    "atlif": ATLIF,
}

__all__ = ["SpikingNeuron", "decay_from_tau", "LIF", "ALIF", "AdLIF",
           "GLIF", "MCLIF", "TCLIF", "ATLIF", "POLARA", "REGISTRY", "build"]


def build(name: str, n: int, **kw) -> SpikingNeuron:
    """按名字构造神经元。POLARA 会抛 NotImplementedError，这是预期行为。"""
    if name == "polara":
        raise NotImplementedError("POLARA 未实现，见 variants/polara.py")
    if name not in REGISTRY:
        raise KeyError(f"未知变体 {name!r}，可选 {sorted(REGISTRY)}")
    return REGISTRY[name](n, **kw)
