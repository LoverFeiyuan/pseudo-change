"""error_decomp.py —— 跨域误差分解

把目标域上的误报（FP）按空间位置拆开，量化「伪变化」在总误差中的占比。

分类规则：
  zero  落在真值全零的 patch 内 —— 纯伪变化，没有任何真实变化可被误认
  near  落在有变化的 patch 内，但距真值边界 ≤ 3 像素 —— 定位/边界误差
  far   落在有变化的 patch 内，且距真值边界 > 3 像素 —— 远端误报

判定依据是「该像素距最近的真实变化像素有多远」，用距离变换得到。
near 与 far 的分界取 3 像素，对应建筑轮廓的典型配准误差量级。

关键输出：若某一类误报被完美消除，F1 能到多少。
这才是「伪变化是否为主要矛盾」的直接证据。

用法：
  uv run error_decomp.py --ckpt runs/base_seed0/best.pt --root data/patches_whu --split test
  uv run error_decomp.py --ckpt runs/base_seed0/best.pt --root data/patches --split test
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data import PATCH_ROOT, PatchDataset, build_loader
from train import CDNet


def f1_from_counts(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return f1, p, r


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=str(PATCH_ROOT))
    ap.add_argument("--split", default="test")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--near-tol", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    print(f"root {root}  split {args.split}  n={len(ds)}  thr={args.thr}  near_tol={args.near_tol}")
    print()

    tp = fn = 0
    fp_zero = fp_near = fp_far = 0
    zero_patches = 0
    total_patches = 0

    for a, b, lab, _ in dl:
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        probs = model(a, b).softmax(1)[:, 1].cpu().numpy()
        labs = lab.numpy()

        for i in range(probs.shape[0]):
            total_patches += 1
            pred = probs[i] > args.thr
            gt = labs[i] > 0

            tp += int((pred & gt).sum())
            fn += int((~pred & gt).sum())

            fp_mask = pred & ~gt
            n_fp = int(fp_mask.sum())
            if n_fp == 0:
                continue

            if not gt.any():
                zero_patches += 1
                fp_zero += n_fp
                continue

            # 每个非变化像素到最近真实变化像素的距离
            dist = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 5)
            near = fp_mask & (dist <= args.near_tol)
            fp_near += int(near.sum())
            fp_far += n_fp - int(near.sum())

    fp_total = fp_zero + fp_near + fp_far
    base_f1, base_p, base_r = f1_from_counts(tp, fp_total, fn)

    print(f"总 patch {total_patches}，其中真值全零 {zero_patches} "
          f"({100 * zero_patches / total_patches:.1f}%)")
    print(f"TP {tp:,}   FN {fn:,}   总 FP {fp_total:,}")
    print()
    print(f"基线：F1 {base_f1:.4f}  P {base_p:.4f}  R {base_r:.4f}")
    print()

    rows = [
        ("zero", fp_zero, "伪变化（零变化区误报）"),
        ("near", fp_near, f"边界误报（距真值 ≤{args.near_tol}px）"),
        ("far", fp_far, "远端误报"),
    ]

    print(f"{'类别':<28} {'FP 像素':>12} {'占 FP':>8} {'消除后 F1':>11} {'ΔF1':>9}")
    print("-" * 72)
    for _, cnt, label in rows:
        share = 100 * cnt / max(fp_total, 1)
        f1_after, _, _ = f1_from_counts(tp, fp_total - cnt, fn)
        print(f"{label:<28} {cnt:>12,} {share:>7.1f}% {f1_after:>11.4f} {f1_after - base_f1:>+9.4f}")

    # 伪变化与远端合在一起消除
    f1_both, _, _ = f1_from_counts(tp, fp_total - fp_zero - fp_far, fn)
    print("-" * 72)
    print(f"{'消除 zero + far':<28} {'':>12} {'':>8} {f1_both:>11.4f} {f1_both - base_f1:>+9.4f}")
    print()
    print(f"伪变化占全部误报的 {100 * fp_zero / max(fp_total, 1):.1f}%；"
          f"完美消除它只能把 F1 从 {base_f1:.4f} 推到 "
          f"{f1_from_counts(tp, fp_total - fp_zero, fn)[0]:.4f}")


if __name__ == "__main__":
    main()
