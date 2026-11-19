"""mine_hard_negatives.py —— 挑出模型误报最严重的零变化 patch

模块 B 失败的直接原因是「信号太散」：752 个零变化 patch 里，大多数模型本来就
判对了，真正在犯错的只是少数。把它们和正确的混在一起塞进 batch，梯度方向被稀释。

这里先用基线模型跑一遍，按误报像素数排序，挑出最严重的 top-N 供后续训练使用。

用法：
  uv run mine_hard_negatives.py --ckpt runs/base_seed0/best.pt --top 100
输出：
  runs/hard_negatives.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import PATCH_ROOT, PatchDataset, build_loader
from train import CDNet, load_zero_keys


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base_seed0/best.pt")
    ap.add_argument("--root", default=str(PATCH_ROOT))
    ap.add_argument("--split", default="train")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--out", default="runs/hard_negatives.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = CDNet(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    keys = load_zero_keys()
    ds = PatchDataset(keys, split=args.split, root=args.root, sync_aug=False)
    dl = build_loader(ds, 8, False, 4)

    print(f"ckpt {args.ckpt}")
    print(f"零变化 patch {len(keys)} 个，thr={args.thr}")
    print()

    scores: list[tuple[int, str]] = []
    for a, b, lab, kb in dl:
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        probs = model(a, b).softmax(1)[:, 1].cpu().numpy()
        for i in range(probs.shape[0]):
            fp = int((probs[i] > args.thr).sum())
            scores.append((fp, kb[i]))

    scores.sort(key=lambda x: (-x[0], x[1]))
    fp_all = np.array([s for s, _ in scores])

    n_zero_fp = int((fp_all == 0).sum())
    print(f"误报为 0 的 patch: {n_zero_fp} ({100 * n_zero_fp / len(fp_all):.1f}%)")
    print(f"误报像素总计: {fp_all.sum():,}")
    print(f"误报面积分布: 中位 {np.median(fp_all):.0f}  "
          f"p75 {np.percentile(fp_all, 75):.0f}  p95 {np.percentile(fp_all, 95):.0f}  "
          f"max {fp_all.max()}")
    print()

    for n in (50, 100, 200, 300):
        top = fp_all[:n]
        cover = 100 * top.sum() / max(fp_all.sum(), 1)
        print(f"  top-{n:<4}: 覆盖 {cover:5.1f}% 的误报像素，"
              f"平均每块 {top.mean():.0f} px")

    top_keys = [k for _, k in scores[:args.top]]
    out = {
        "source_ckpt": args.ckpt,
        "thr": args.thr,
        "n_total": len(keys),
        "n_with_fp": int((fp_all > 0).sum()),
        "top": top_keys,
        "scores": {k: int(s) for s, k in scores[: args.top * 3]},
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten -> {outp}  (top {len(top_keys)})")


if __name__ == "__main__":
    main()
