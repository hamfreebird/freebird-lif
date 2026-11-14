"""确保 tests/ 与 scripts/ 能直接 import freebird_lif（无需安装成包）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
