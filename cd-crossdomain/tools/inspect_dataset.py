"""数据集体检：尺寸、标签取值、变化像素占比、连通域数

这些数字后面每一个都要进论文：
  - 变化像素占比  -> 决定 loss 是否要加权(Dice/focal/pos_weight)
  - 每图连通域数  -> 决定"连通域面积过滤"的阈值 tau_area 取多少
  - 零变化图像数  -> 模块 B(零变化监督) 的真实版样本池

跑法： uv run inspect_dataset.py
输出： 控制台 + data/dataset_report.json
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"data")
SPLITS = ["train", "val", "test"]
UNIQ_SAMPLE = 20  # 只在前 20 张上统计标签取值，省时间


def main():
    report = {}
    for split in SPLITS:
        lab_dir = ROOT / split / "label"
        a_dir = ROOT / split / "A"
        b_dir = ROOT / split / "B"
        if not lab_dir.exists():
            continue

        labs = sorted(lab_dir.glob("*.png"))
        sizes: set = set()
        uniq: set = set()
        ratios: list[float] = []
        comps: list[int] = []

        for i, lp in enumerate(labs):
            m = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                print(f"  !! unreadable: {lp.name}")
                continue
            sizes.add(m.shape)
            if i < UNIQ_SAMPLE:
                uniq.update(np.unique(m).tolist())

            b = (m > 127).astype(np.uint8)
            ratios.append(float(b.mean()))
            n, _ = cv2.connectedComponents(b)
            comps.append(int(n - 1))

        report[split] = {
            "pairs": len(labs),
            "a_files": len(list(a_dir.glob("*"))),
            "b_files": len(list(b_dir.glob("*"))),
            "image_shapes": [list(s) for s in sorted(sizes)],
            "label_values_seen": sorted(uniq),
            "change_ratio": {
                "mean": round(float(np.mean(ratios)), 4),
                "median": round(float(np.median(ratios)), 4),
                "p10": round(float(np.percentile(ratios, 10)), 4),
                "p90": round(float(np.percentile(ratios, 90)), 4),
                "min": round(float(np.min(ratios)), 4),
                "max": round(float(np.max(ratios)), 4),
            },
            "components_per_image": {
                "mean": round(float(np.mean(comps)), 2),
                "median": round(float(np.median(comps)), 2),
                "max": int(np.max(comps)),
                "zero_change_images": int(sum(1 for c in comps if c == 0)),
                "le3_components_images": int(sum(1 for c in comps if 1 <= c <= 3)),
            },
        }
        print(f"[{split}] done: {len(labs)} pairs")

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print()
    print(text)
    out = ROOT / "dataset_report.json"
    out.write_text(text, encoding="utf-8")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
