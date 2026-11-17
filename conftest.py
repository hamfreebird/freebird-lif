"""确保 tests/ 与 scripts/ 能直接 import freebird_lif（无需安装成包）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 同时加入 scripts/：测试要 import train_shd / analyze_* 这些脚本模块，
# 它们是脚本而非包，只能靠路径注入。
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
