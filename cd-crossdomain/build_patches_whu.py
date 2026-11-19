"""build_patches_whu.py —— WHU-CD 切块

WHU-CD 是「两期巨图 + 一张同尺寸变化掩码」的形式，和 LEVIR 的目录结构完全不同：
  2012/whole_image/{split}/image/2012_{split}.tif   21243x15354 (train)
  2016/whole_image/{split}/image/2016_{split}.tif   同尺寸
  change_label/{split}/change_label.tif             mode=1 二值

不用它自带的 splited_images（512 块、且没有对应标签），直接按 256 网格从整图切，
网格干净、标签天然对齐。

输出（和 LEVIR 的 patches 同构，这样 evaluate.py 直接能用）：
  <OUT>/{split}/A/{split}_{r}_{c}.npy   (256,256,3) uint8 BGR
  <OUT>/{split}/B/...
  <OUT>/{split}/L/...                   (256,256)   uint8 0/1
  <OUT>/patch_stats_whu.csv

跑法：
  uv run build_patches_whu.py --split train
  uv run build_patches_whu.py --split all
"""
import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(r"data/Building change detection dataset_add")
BASE = ROOT / "Building change detection dataset_add" / "1. The two-period image data"
OUT = Path(r"data/patches_whu")
P = 256
SPLITS = ["train", "test"]


def load_rgb(p: Path) -> np.ndarray:
    """PIL 读成 RGB uint8，返回 (H,W,3)。"""
    im = Image.open(p)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.array(im)


def load_mask(p: Path) -> np.ndarray:
    """变化掩码：mode=1 时直接是 bool。"""
    im = Image.open(p)
    arr = np.array(im)
    if arr.dtype == bool:
        return arr.astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 127).astype(np.uint8)


def build(split: str) -> list[dict]:
    t0 = time.time()
    p12 = BASE / "2012" / "whole_image" / split / "image" / f"2012_{split}.tif"
    p16 = BASE / "2016" / "whole_image" / split / "image" / f"2016_{split}.tif"
    pl = BASE / "change_label" / split / "change_label.tif"
    for p in (p12, p16, pl):
        if not p.exists():
            print(f"  !! 缺失 {p}")
            return []

    print(f"  loading {p12.name} ...")
    a12 = load_rgb(p12)
    print(f"    {a12.shape} {a12.dtype}  ({time.time() - t0:.0f}s)")
    print(f"  loading {p16.name} ...")
    a16 = load_rgb(p16)
    print(f"    {a16.shape} {a16.dtype}  ({time.time() - t0:.0f}s)")
    lab = load_mask(pl)
    print(f"  label {lab.shape} {lab.dtype}  pos={lab.mean():.4f}  ({time.time() - t0:.0f}s)")

    if not (a12.shape[:2] == a16.shape[:2] == lab.shape):
        print(f"  !! 尺寸不一致 {a12.shape[:2]} / {a16.shape[:2]} / {lab.shape}")
        del a12, a16, lab
        gc.collect()
        return []

    H, W = a12.shape[:2]
    nr, nc = H // P, W // P
    print(f"  grid: {nr} rows x {nc} cols = {nr * nc} patches")

    dst = OUT / split
    for sub in ("A", "B", "L"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for r in range(nr):
        for c in range(nc):
            y0, x0 = r * P, c * P
            pa = a12[y0:y0 + P, x0:x0 + P]
            pb = a16[y0:y0 + P, x0:x0 + P]
            pl_ = lab[y0:y0 + P, x0:x0 + P]

            key = f"{split}_{r}_{c}"
            # PIL 给的是 RGB，这里转 BGR 保持和 LEVIR 的约定一致
            np.save(dst / "A" / f"{key}.npy", np.ascontiguousarray(pa[:, :, ::-1]))
            np.save(dst / "B" / f"{key}.npy", np.ascontiguousarray(pb[:, :, ::-1]))
            np.save(dst / "L" / f"{key}.npy", pl_)

            n_comp = int(cv2.connectedComponents(pl_, connectivity=8)[0] - 1)
            rows.append({
                "split": split, "name": p12.name, "stem": f"{split}",
                "row": r, "col": c, "key": key,
                "change_pix": int(pl_.sum()),
                "change_ratio": round(float(pl_.mean()), 6),
                "n_comp": n_comp,
            })
        if (r + 1) % 10 == 0:
            print(f"    rows {r + 1}/{nr}  ({time.time() - t0:.0f}s)")

    del a12, a16, lab
    gc.collect()
    print(f"  done {split}: {len(rows)} patches in {time.time() - t0:.0f}s")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS + ["all"], default="all")
    args = ap.parse_args()

    splits = SPLITS if args.split == "all" else [args.split]
    all_rows: list[dict] = []
    for s in splits:
        print(f"== {s} ==")
        all_rows.extend(build(s))

    if not all_rows:
        print("nothing produced")
        return

    csv_path = OUT / "patch_stats_whu.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nstats -> {csv_path}  ({len(all_rows)} patches)")

    for s in SPLITS:
        sub = [r for r in all_rows if r["split"] == s]
        if not sub:
            continue
        zero = sum(1 for r in sub if r["change_ratio"] == 0)
        print(f"[{s}] {len(sub)} patches, {zero} all-zero ({100 * zero / len(sub):.1f}%)")


if __name__ == "__main__":
    main()
