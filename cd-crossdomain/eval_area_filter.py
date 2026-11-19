"""eval_area_filter.py —— 模块 C：连通域面积过滤

两件事一起做：
  1. τ 扫描：对模型输出的二值预测做连通域过滤，看 F1 怎么变
  2. 误检面积分布：在「真值全零」的 patch 上统计模型误检的连通域面积，
     替换掉之前用 Otsu 差分代理想的那套数字

为什么要在真值全零的 patch 上统计：那里模型的每一个正预测都是误检，
面积分布就是「伪变化」在模型眼里的真实形态。

注意：过滤是在 256×256 patch 内做的，跨 patch 边界的连通域会被截断，
对面积估计偏小。LEVIR 真实变化连通域中位数 862 像素，多数不至于跨界，
但边界效应需要在论文里提一句。

用法：
  uv run eval_area_filter.py --ckpt runs/modB_only_r10/best.pt --root data/patches --split test
  uv run eval_area_filter.py --ckpt runs/modB_only_r10/best.pt --root data/patches_whu --split test
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data import PATCH_ROOT, PatchDataset, build_loader
from train import CDNet, _metrics


def filter_by_area(mask: np.ndarray, tau: int) -> np.ndarray:
    """丢弃面积小于 tau 的连通域。tau<=1 时原样返回。"""
    if tau <= 1:
        return mask
    m = mask.astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_lut = np.concatenate([[False], areas >= tau])
    return keep_lut[lbl]


def cc_areas(mask: np.ndarray) -> list[int]:
    m = mask.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return []
    return stats[1:, cv2.CC_STAT_AREA].astype(int).tolist()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=str(PATCH_ROOT))
    ap.add_argument("--split", default="test")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--taus", default="0,32,64,128,256")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    taus = [int(t) for t in args.taus.split(",")]

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = CDNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = Path(args.root)
    keys = sorted(p.stem for p in (root / args.split / "L").glob("*.npy"))
    if not keys:
        raise SystemExit(f"没找到样本：{root / args.split / 'L'}")
    ds = PatchDataset(keys, split=args.split, root=root, sync_aug=False)
    dl = build_loader(ds, args.batch_size, False, args.workers)

    print(f"ckpt {args.ckpt}")
    print(f"root {root}  split {args.split}  n={len(ds)}  thr={args.thr}")
    print()

    # counters[tau] = [tp, fp, fn]
    counters = {t: [0, 0, 0] for t in taus}
    fp_areas: list[int] = []
    zero_patches = 0

    for a, b, lab, _ in dl:
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        probs = model(a, b).softmax(1)[:, 1].cpu().numpy()
        labs = lab.numpy()

        for i in range(probs.shape[0]):
            pred = probs[i] > args.thr
            gt = labs[i] > 0

            # 真值全零的 patch：模型的正预测全是误检，用来统计伪变化面积
            if not gt.any():
                zero_patches += 1
                fp_areas.extend(cc_areas(pred))

            for t in taus:
                p = filter_by_area(pred, t)
                c = counters[t]
                c[0] += int((p & gt).sum())
                c[1] += int((p & ~gt).sum())
                c[2] += int((~p & gt).sum())

    print(f"{'tau':>6} {'F1':>8} {'IoU':>8} {'P':>8} {'R':>8}")
    print("-" * 42)
    for t in taus:
        tp, fp, fn = counters[t]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        iou = tp / max(tp + fp + fn, 1)
        tag = "(不过滤)" if t <= 1 else ""
        print(f"{t:>6} {f1:>8.4f} {iou:>8.4f} {p:>8.4f} {r:>8.4f}  {tag}")

    print()
    print(f"=== 伪变化（模型在真值全零 patch 上的误检）===")
    print(f"零变化 patch: {zero_patches}，误检连通域: {len(fp_areas)}")
    if fp_areas:
        arr = np.array(fp_areas)
        print(f"  面积中位数 {np.median(arr):.0f}  均值 {arr.mean():.1f}  "
              f"p25 {np.percentile(arr, 25):.0f}  p75 {np.percentile(arr, 75):.0f}  "
              f"p95 {np.percentile(arr, 95):.0f}  max {arr.max()}")
        for t in (32, 64, 128, 256):
            drop = float((arr < t).mean())
            px = float(arr[arr < t].sum() / max(arr.sum(), 1))
            print(f"  tau={t:>4}: 过滤掉 {100 * drop:.1f}% 的连通域，"
                  f"占误检像素 {100 * px:.1f}%")


if __name__ == "__main__":
    main()
