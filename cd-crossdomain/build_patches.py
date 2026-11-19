"""build_patches.py —— 预切块 + 每块统计

把 LEVIR-CD 的 1024² 双时相 PNG 切成 256² 小块存为 .npy，
同时算出每块的变化像素比例（分层采样要用）。

输出布局：
  <DATA>/patches/train/A/train_1_00.npy    (256,256,3) uint8  BGR
  <DATA>/patches/train/B/train_1_00.npy    (256,256,3) uint8  BGR
  <DATA>/patches/train/L/train_1_00.npy    (256,256)   uint8  0/1
  <DATA>/patches/val/...   <DATA>/patches/test/...
  <DATA>/patches/patch_stats.csv

约定：
  * 颜色通道一律 BGR（cv2 原生顺序），训练侧保持同一约定，别混用 RGB
  * label 存成 0/1 而不是 0/255，训练时直接当 target
  * 1024 / 256 = 4，不重叠切 4x4 = 16 块，无 padding

跑法：
  uv run build_patches.py                 # 全量
  uv run build_patches.py --split train   # 只处理 train
  uv run build_patches.py --split val --verify   # 顺便做重建校验
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:  # Windows 控制台默认 GBK，中文输出会乱码
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = Path(r"data")
OUT = DATA / "patches"
PATCH = 256
SPLITS = ["train", "val", "test"]


def build(split: str) -> list[dict]:
    src = DATA / split
    dst = OUT / split
    for sub in ("A", "B", "L"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in (src / "A").glob("*.png"))
    if not names:
        print(f"  !! no images under {src / 'A'}")
        return []

    rows: list[dict] = []
    t0 = time.time()

    for i, name in enumerate(names):
        a = cv2.imread(str(src / "A" / name), cv2.IMREAD_COLOR)
        b = cv2.imread(str(src / "B" / name), cv2.IMREAD_COLOR)
        lab = cv2.imread(str(src / "label" / name), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or lab is None:
            print(f"  !! unreadable, skip: {name}")
            continue
        if a.shape[0] % PATCH or a.shape[1] % PATCH:
            print(f"  !! shape {a.shape[:2]} not divisible by {PATCH}, skip: {name}")
            continue

        gh, gw = a.shape[0] // PATCH, a.shape[1] // PATCH
        stem = Path(name).stem

        for r in range(gh):
            for c in range(gw):
                y0, x0 = r * PATCH, c * PATCH
                pa = a[y0:y0 + PATCH, x0:x0 + PATCH]
                pb = b[y0:y0 + PATCH, x0:x0 + PATCH]
                pl = (lab[y0:y0 + PATCH, x0:x0 + PATCH] > 127).astype(np.uint8)

                key = f"{stem}_{r}{c}"
                np.save(dst / "A" / f"{key}.npy", pa)
                np.save(dst / "B" / f"{key}.npy", pb)
                np.save(dst / "L" / f"{key}.npy", pl)

                n_comp = int(cv2.connectedComponents(pl, connectivity=8)[0] - 1)
                rows.append({
                    "split": split,
                    "name": name,
                    "stem": stem,
                    "row": r,
                    "col": c,
                    "key": key,
                    "change_pix": int(pl.sum()),
                    "change_ratio": round(float(pl.mean()), 6),
                    "n_comp": n_comp,
                })

        if (i + 1) % 100 == 0 or i == len(names) - 1:
            print(f"  [{split}] {i + 1}/{len(names)}  {time.time() - t0:.0f}s")

    return rows


def verify(split: str, n: int = 3) -> None:
    """抽几块拼回原图，确认切块无损。"""
    names = sorted((DATA / split / "A").glob("*.png"))
    if not names:
        return
    pick = names[:: max(len(names) // n, 1)][:n]
    print("  verify (max abs pixel diff after rebuild):")
    for src in pick:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        rebuilt = np.zeros_like(img)
        for r in range(h // PATCH):
            for c in range(w // PATCH):
                p = np.load(OUT / split / "A" / f"{src.stem}_{r}{c}.npy")
                rebuilt[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = p
        diff = int(np.abs(rebuilt.astype(np.int16) - img.astype(np.int16)).max())
        status = "OK" if diff == 0 else "MISMATCH"
        print(f"    {src.name:20s} diff={diff:3d}  {status}")


def summarize(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n=== patch 概览 ===")
    bins = [
        ("全零(纯负样本)", lambda x: x == 0),
        ("(0, 0.1%]", lambda x: 0 < x <= 0.001),
        ("(0.1%, 1%]", lambda x: 0.001 < x <= 0.01),
        ("(1%, 5%]", lambda x: 0.01 < x <= 0.05),
        ("(5%, 20%]", lambda x: 0.05 < x <= 0.20),
        ("(20%, 100%]", lambda x: x > 0.20),
    ]
    for split in SPLITS:
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        total = len(sub)
        print(f"\n[{split}] {total} patches")
        for label, fn in bins:
            cnt = sum(1 for r in sub if fn(r["change_ratio"]))
            print(f"  {label:16s} {cnt:6d}  {100 * cnt / total:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS + ["all"], default="all")
    ap.add_argument("--verify", action="store_true", help="切完抽几块重建比对")
    args = ap.parse_args()

    splits = SPLITS if args.split == "all" else [args.split]
    all_rows: list[dict] = []

    for s in splits:
        print(f"== {s} ==")
        all_rows.extend(build(s))
        if args.verify:
            verify(s)

    if not all_rows:
        print("nothing produced")
        return

    csv_path = OUT / "patch_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nstats -> {csv_path}  ({len(all_rows)} patches)")

    summarize(all_rows)


if __name__ == "__main__":
    main()
