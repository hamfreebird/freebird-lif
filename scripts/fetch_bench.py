"""多基准数据集下载（并行、可续传、幂等）。

为什么不用官方 zip：
  * N-MNIST 官方 `Train.zip` 是 1012 MB，本机到 Mendeley/S3 实测 0.7–15 KB/s，
    整包需数十小时。HuggingFace 镜像 `Blackrose900/N-MNIST` 是逐文件结构，
    可以按需取子集，实测 8 路并行约 33 KB/s。

镜像规模（M0 实测，递归列举得到）：Train 9968 / Test 10001 个 .bin。
官方是 60000 / 10000，**训练集只有官方的 1/6**。因此本阶段报告里的每个
N-MNIST 数字都必须注明子集规模——样本量是本阶段的自变量之一。

用法：
    python scripts/fetch_bench.py --per-class 300 --per-class-test 100
    python scripts/fetch_bench.py --check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "bench" / "nmnist"
REPO = "Blackrose900/N-MNIST"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true&limit=1000"


def list_all(retries: int = 5) -> list[str]:
    """一次列出仓库全部文件。

    必须用 `recursive=true` + 游标翻页：单目录接口会把结果截到 1000 条，
    而 Train/0 恰好有约 997 个，看起来"正常"实则已丢数据；非递归版本
    更会漏掉整个 Train 层级（第一版就撞上了）。
    """
    url, files = API, []
    for _page in range(40):
        last = None
        data = link = None
        for k in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "freebird-lif/0.3"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data, link = json.load(r), r.headers.get("Link", "")
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(2.0 * (k + 1))
        if data is None:
            raise RuntimeError(f"列举失败：{last}")
        files += [f["path"] for f in data if f.get("type") == "file"]
        if not link or 'rel="next"' not in link:
            break
        url = link.split("<")[1].split(">")[0]
    return files


def fetch_one(remote: str, local: Path, retries: int = 4) -> tuple[bool, int]:
    """下载单个文件。已存在且内容合法则跳过（幂等）。"""
    if local.exists() and local.stat().st_size > 100 and local.stat().st_size % 5 == 0:
        return True, 0
    local.parent.mkdir(parents=True, exist_ok=True)
    for k in range(retries):
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "120", "-o", str(local), f"{BASE}/{remote}"],
            capture_output=True,
        )
        if r.returncode == 0 and local.exists():
            n = local.stat().st_size
            # .bin 必须是 5 字节事件的整数倍。错误页/截断会被这条挡掉，
            # 否则它们会被当成"下载成功"，等到解析时才炸。
            if n > 100 and n % 5 == 0:
                return True, n
        time.sleep(1.5 * (k + 1))
    return False, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--per-class-test", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        fs = list(DEST.rglob("*.bin"))
        print(f"已有 {len(fs)} 个 .bin，{sum(f.stat().st_size for f in fs) / 1e6:.1f} MB")
        return 0

    print("列举仓库文件…", flush=True)
    allf = list_all()
    print(f"  共 {len(allf)} 个文件", flush=True)

    jobs: list[tuple[str, Path]] = []
    for split, n in (("Train", args.per_class), ("Test", args.per_class_test)):
        for digit in range(10):
            sub = sorted(p for p in allf
                         if p.startswith(f"{split}/{digit}/") and p.endswith(".bin"))
            if len(sub) < n:
                print(f"  警告：{split}/{digit} 只有 {len(sub)} 个，少于请求的 {n}")
            jobs += [(p, DEST / p) for p in sub[:n]]
    print(f"  计划下载 {len(jobs)} 个，目标 {DEST}", flush=True)

    t0, done, got = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_one, r, l) for r, l in jobs]
        for fu in futs:
            ok, n = fu.result()
            if ok:
                done += 1
                got += n
                if done % 200 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{len(jobs)}  {got / 1e6:.1f} MB  "
                          f"{got / max(el, 1) / 1024:.0f} KB/s", flush=True)

    bad = len(jobs) - done
    print(f"\n完成 {done}/{len(jobs)}，失败 {bad}，新增 {got / 1e6:.1f} MB，"
          f"用时 {(time.time() - t0) / 60:.1f} min")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
