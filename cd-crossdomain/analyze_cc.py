"""连通域面积分布 —— 为模块 C 的 tau_area 定阈值

回答三个问题：
  1. 真实变化的连通域到底有多大？（分位数 + 对数直方图）
  2. 不同 tau 下会丢掉多少真实变化？（阈值损失曲线）
  3. 伪变化（零变化图上的差分）和真实变化，在「面积」和「贴边性」上分得开吗？

跑法： uv run analyze_cc.py
输出： 控制台 + data/cc_report.json
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"data")
SPLITS = ["train", "val", "test"]
TAUS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
HIST_BINS = [1, 4, 10, 25, 50, 100, 250, 500, 1000, 2500]


def cc_areas(mask_bool: np.ndarray) -> np.ndarray:
    m = mask_bool.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(0, dtype=np.int64)
    return stats[1:, cv2.CC_STAT_AREA].astype(np.int64)


def summarize_areas(areas: np.ndarray) -> dict:
    if areas.size == 0:
        return {"n": 0}
    return {
        "n": int(areas.size),
        "mean": round(float(areas.mean()), 1),
        "p1": round(float(np.percentile(areas, 1)), 1),
        "p5": round(float(np.percentile(areas, 5)), 1),
        "p10": round(float(np.percentile(areas, 10)), 1),
        "p25": round(float(np.percentile(areas, 25)), 1),
        "p50": round(float(np.percentile(areas, 50)), 1),
        "p75": round(float(np.percentile(areas, 75)), 1),
        "p90": round(float(np.percentile(areas, 90)), 1),
        "p95": round(float(np.percentile(areas, 95)), 1),
        "p99": round(float(np.percentile(areas, 99)), 1),
        "max": int(areas.max()),
    }


def hist(areas: np.ndarray) -> dict:
    if areas.size == 0:
        return {}
    h, _ = np.histogram(areas, bins=HIST_BINS + [10**9])
    labels = [f"{HIST_BINS[i]}-{HIST_BINS[i + 1] - 1}" for i in range(len(HIST_BINS) - 1)]
    labels.append(f">={HIST_BINS[-1]}")
    total = int(areas.size)
    return {lab: {"count": int(c), "pct": round(100 * c / total, 2)} for lab, c in zip(labels, h)}


def tau_loss(areas: np.ndarray) -> dict:
    if areas.size == 0:
        return {}
    total_cc = int(areas.size)
    total_px = int(areas.sum())
    out = {}
    for t in TAUS:
        drop_cc = int((areas < t).sum())
        drop_px = int(areas[areas < t].sum())
        out[str(t)] = {
            "drop_cc_pct": round(100 * drop_cc / total_cc, 2),
            "drop_px_pct": round(100 * drop_px / max(total_px, 1), 2),
        }
    return out


def edge_affinity(img_path: Path, mask_bool: np.ndarray):
    g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    e = cv2.Canny(g, 50, 150) > 0
    e = cv2.dilate(e.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    sel = int(mask_bool.sum())
    if sel == 0:
        return None
    return float((mask_bool & e).sum() / sel)


def main():
    report = {"real_change": {}, "pseudo_change": {}, "edge_comparison": {}}

    for split in SPLITS:
        lab_dir = ROOT / split / "label"
        a_dir = ROOT / split / "A"
        all_areas = []
        edge_real = []
        zero_imgs = []

        for lp in sorted(lab_dir.glob("*.png")):
            m = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            mb = m > 127
            areas = cc_areas(mb)
            if areas.size == 0:
                zero_imgs.append(lp.name)
                continue
            all_areas.append(areas)
            ea = edge_affinity(a_dir / lp.name, mb)
            if ea is not None:
                edge_real.append(ea)

        areas = np.concatenate(all_areas) if all_areas else np.zeros(0, dtype=np.int64)
        report["real_change"][split] = {
            "pairs": len(list(lab_dir.glob("*.png"))),
            "zero_change_images": len(zero_imgs),
            "areas": summarize_areas(areas),
            "hist": hist(areas),
            "tau_loss": tau_loss(areas),
            "edge_affinity_mean": round(float(np.mean(edge_real)), 4) if edge_real else None,
        }
        print(f"[{split}] collected {areas.size} connected components")

    # 伪变化：train 里零变化图的 |grayA - grayB| + Otsu
    train_dir = ROOT / "train"
    zero_names = []
    for lp in sorted((train_dir / "label").glob("*.png")):
        m = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        if m is not None and int((m > 127).sum()) == 0:
            zero_names.append(lp.name)

    pseudo_areas, pseudo_edge, otsus = [], [], []
    for name in zero_names:
        ga = cv2.imread(str(train_dir / "A" / name), cv2.IMREAD_GRAYSCALE)
        gb = cv2.imread(str(train_dir / "B" / name), cv2.IMREAD_GRAYSCALE)
        if ga is None or gb is None:
            continue
        diff = cv2.absdiff(ga, gb)
        t, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mb = mask > 0
        otsus.append(float(t))
        pseudo_areas.append(cc_areas(mb))
        ea = edge_affinity(train_dir / "A" / name, mb)
        if ea is not None:
            pseudo_edge.append(ea)

    pa = np.concatenate(pseudo_areas) if pseudo_areas else np.zeros(0, dtype=np.int64)
    report["pseudo_change"] = {
        "source": "train split, zero-change pairs, |grayA-grayB| with Otsu",
        "n_images": len(zero_names),
        "otsu_mean": round(float(np.mean(otsus)), 2) if otsus else None,
        "areas": summarize_areas(pa),
        "hist": hist(pa),
    }
    report["edge_comparison"] = {
        "real_change_pixels_on_canny_pct": report["real_change"]["train"]["edge_affinity_mean"],
        "pseudo_change_pixels_on_canny_pct": round(float(np.mean(pseudo_edge)), 4) if pseudo_edge else None,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print()
    print(text)
    out = ROOT / "cc_report.json"
    out.write_text(text, encoding="utf-8")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
