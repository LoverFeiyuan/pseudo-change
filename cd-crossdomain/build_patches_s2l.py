"""build_patches_s2l.py —— 把 S2Looking 切成 patch（仅供误差分解使用）

S2Looking 原本走在线切块评估（eval_crossdomain.py），但 error_decomp.py 需要
patch 格式的磁盘文件。这里只切 test：1000 对 x 16 = 16000 块，约 9 GB。

用法：
  uv run build_patches_s2l.py --split test
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = Path(r"data/S2Looking\S2Looking")
OUT = Path(r"data/patches_s2l")
P = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    args = ap.parse_args()

    src = SRC / args.split
    dst = OUT / args.split
    for sub in ("A", "B", "L"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in (src / "Image1").glob("*.png"))
    if args.limit:
        names = names[: args.limit]

    print(f"{len(names)} 对图 -> {dst}")
    rows = []
    t0 = time.time()

    for i, n in enumerate(names):
        a = cv2.imread(str(src / "Image1" / n), cv2.IMREAD_COLOR)
        b = cv2.imread(str(src / "Image2" / n), cv2.IMREAD_COLOR)
        lab = cv2.imread(str(src / "label" / n), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or lab is None:
            continue
        if a.shape[0] % P or a.shape[1] % P:
            continue

        stem = Path(n).stem
        for r in range(a.shape[0] // P):
            for c in range(a.shape[1] // P):
                y0, x0 = r * P, c * P
                pa = a[y0:y0 + P, x0:x0 + P]
                pb = b[y0:y0 + P, x0:x0 + P]
                pl = (lab[y0:y0 + P, x0:x0 + P] > 127).astype(np.uint8)

                key = f"{stem}_{r}{c}"
                np.save(dst / "A" / f"{key}.npy", pa)
                np.save(dst / "B" / f"{key}.npy", pb)
                np.save(dst / "L" / f"{key}.npy", pl)

                n_comp = int(cv2.connectedComponents(pl, connectivity=8)[0] - 1)
                rows.append({
                    "split": args.split, "name": n, "stem": stem,
                    "row": r, "col": c, "key": key,
                    "change_pix": int(pl.sum()),
                    "change_ratio": round(float(pl.mean()), 6),
                    "n_comp": n_comp,
                })

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(names)}  {time.time() - t0:.0f}s")

    csv_path = OUT / "patch_stats_s2l.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n{len(rows)} patches -> {csv_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
