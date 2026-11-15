"""下载并构建 SHD 缓存。

用法:
    python scripts/fetch_shd.py              # 默认 T=100，二分箱
    python scripts/fetch_shd.py --T 50
    python scripts/fetch_shd.py --force      # 忽略缓存重建
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freebird_lif.shd.data import ROOT, build, describe, download, file_sha256  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--counts", action="store_true", help="存计数而非二值")
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = ROOT / "data" / "shd"
    print("[1/2] 确保原始文件就位")
    paths = download(root, force=False)
    for s, p in paths.items():
        print(f"  {p.name}  {p.stat().st_size / 1024 / 1024:.0f} MB  "
              f"sha256={file_sha256(p)[:16]}…")

    print("[2/2] 分箱与切分")
    d = build(root, T=args.T, binary=not args.counts,
              val_size=args.val_size, force=args.force)
    print(describe(d))

    meta_path = root / "manifest.json"
    meta = {
        **d["meta"],
        "source": {s: {"name": p.name, "bytes": p.stat().st_size,
                       "sha256": file_sha256(p)} for s, p in paths.items()},
        "shapes": {k: list(d[k].shape) for k in
                   ("x_train", "x_val", "x_test", "y_train", "y_val", "y_test")},
        "note": "验证集从训练集切出，官方测试集全程隔离；与文献用测试集早停的做法不同",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    print(f"\n  manifest → {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
