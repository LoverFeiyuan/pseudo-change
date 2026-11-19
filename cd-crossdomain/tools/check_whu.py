"""check_whu.py —— 摸清 WHU-CD 的几何与标签形式

要确认的：
  1. whole_image 的巨图尺寸 / 波段 / 位深
  2. change_label 的尺寸是否与巨图一致、取值集合、变化像素占比（那是"变化"，不是"建筑"）
  3. splited_images 的块尺寸和数量（可能已经切好了，省一道工序）
"""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# WHU 的整幅 tif 是 3.26 亿像素，超过 PIL 默认的 1.79 亿防爆阈值
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(r"data/Building change detection dataset_add")
BASE = ROOT / "Building change detection dataset_add" / "1. The two-period image data"


def probe(p: Path, label: str, sample: bool = False) -> None:
    if not p.exists():
        print(f"[{label}] 缺失: {p}")
        return
    im = Image.open(p)
    print(f"[{label}] {p.name}")
    print(f"    size={im.size}  mode={im.mode}  mb={p.stat().st_size / 1e6:.1f}")
    if sample:
        arr = np.array(im)
        u = np.unique(arr)
        print(f"    dtype={arr.dtype} shape={arr.shape} unique={u[:8].tolist()}"
              f"{' ...' if len(u) > 8 else ''} (共 {len(u)} 个值)")
        if len(u) <= 8:
            pos = (arr > 0).mean()
            print(f"    正像素占比 = {pos:.4f}")


def list_split(sub: str) -> None:
    d = BASE / "2012" / "splited_images" / sub / "image"
    if not d.exists():
        print(f"[splited/{sub}] 缺失")
        return
    files = sorted(d.glob("*.tif"))
    print(f"[splited/{sub}] {len(files)} 张")
    if files:
        im = Image.open(files[0])
        print(f"    示例 {files[0].name}: size={im.size} mode={im.mode}")
        ext = Counter(f.suffix for f in files)
        print(f"    扩展名分布 {dict(ext)}")


if __name__ == "__main__":
    print("=== whole image ===")
    probe(BASE / "2012" / "whole_image" / "train" / "image" / "2012_train.tif", "2012_train")
    probe(BASE / "2016" / "whole_image" / "train" / "image" / "2016_train.tif", "2016_train")
    probe(BASE / "2012" / "whole_image" / "test" / "image" / "2012_test.tif", "2012_test")
    probe(BASE / "2016" / "whole_image" / "test" / "image" / "2016_test.tif", "2016_test")

    print("\n=== change label ===")
    probe(BASE / "change_label" / "train" / "change_label.tif", "label_train", sample=True)
    probe(BASE / "change_label" / "test" / "change_label.tif", "label_test", sample=True)

    print("\n=== splited images ===")
    list_split("train")
    list_split("test")

    print("\n=== building label (单时相，作参考) ===")
    probe(BASE / "2012" / "whole_image" / "train" / "label" / "2012_train.tif", "bldg2012", sample=True)
