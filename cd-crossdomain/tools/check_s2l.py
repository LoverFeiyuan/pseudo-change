"""check_s2l.py —— 排除数据读取层面的问题

三个要确认的：
  1. PNG 位深（16-bit 被 cv2.imread 当 8-bit 读会整体变暗，特征响应全错）
  2. 通道顺序与色彩统计（和 LEVIR 差多少）
  3. 变化像素占比（S2Looking 本身比 LEVIR 稀疏多少）
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(r"data/S2Looking\S2Looking")
LEVIR = Path(r"data")


def probe_image(p: Path, tag: str) -> None:
    im = Image.open(p)
    arr = np.array(im)
    print(f"[{tag}] {p.name}")
    print(f"   PIL mode={im.mode} size={im.size} dtype={arr.dtype} shape={arr.shape}")
    if arr.ndim == 3:
        print(f"   per-channel mean {arr.reshape(-1, arr.shape[2]).mean(0).round(1)}")

    c = cv2.imread(str(p), cv2.IMREAD_COLOR)
    g = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    print(f"   cv2 COLOR     shape={c.shape} dtype={c.dtype} mean={c.mean():.2f}")
    print(f"   cv2 UNCHANGED shape={g.shape} dtype={g.dtype} mean={g.mean():.2f}")


def change_ratio(split: str, limit: int = 100) -> None:
    d = ROOT / split / "label"
    files = sorted(d.glob("*.png"))[:limit]
    ratios = []
    for f in files:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            ratios.append(float((m > 127).mean()))
    if ratios:
        print(f"[{split}] {len(ratios)} labels: change ratio "
              f"mean={np.mean(ratios):.4f} median={np.median(ratios):.4f} "
              f"max={np.max(ratios):.4f} zero={sum(1 for r in ratios if r == 0)}")


if __name__ == "__main__":
    s2 = ROOT / "test" / "Image1" / "1008.png"
    if s2.exists():
        probe_image(s2, "S2L Image1")

    lv = LEVIR / "train" / "A" / "train_1.png"
    if lv.exists():
        probe_image(lv, "LEVIR A")

    print()
    change_ratio("test", 200)
    change_ratio("train", 200)
