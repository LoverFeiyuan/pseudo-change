"""eval_crossdomain.py —— 零样本跨域评估（在线切块，不落盘）

为什么不在线切块而预切：S2Looking 有 5000 对 1024² 图，切块后是 8 万个 patch、
20 GB 以上、24 万个文件。零样本评估只跑一次，不值得为它铺一遍磁盘。

目录约定（S2Looking）：
  <root>/<split>/Image1/*.png   前时相
  <root>/<split>/Image2/*.png   后时相
  <root>/<split>/label/*.png    变化标注 0/255

用法：
  uv run eval_crossdomain.py --ckpt runs/base_r40/best.pt `
      --root data/S2Looking/S2Looking --split test
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data import IMAGENET_MEAN, IMAGENET_STD
from train import CDNet, _metrics


def to_tensor_bgr(img: np.ndarray) -> torch.Tensor:
    x = np.ascontiguousarray(img[:, :, ::-1]).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True, help="数据集根目录（含 train/val/test）")
    ap.add_argument("--split", default="test")
    ap.add_argument("--patch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    thr = float(ckpt.get("thr", 0.5))
    cargs = ckpt.get("args", {})

    model = CDNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = Path(args.root) / args.split
    names = sorted(p.name for p in (root / "Image1").glob("*.png"))
    if not names:
        raise SystemExit(f"没找到图像：{root / 'Image1'}")

    print(f"checkpoint: {args.ckpt}")
    print(f"  epoch {ckpt.get('epoch')} ratio {cargs.get('ratio')} "
          f"pos_weight {cargs.get('pos_weight')} 阈值 {thr:.2f}")
    print(f"target: {root}  {len(names)} pairs, patch={args.patch}")
    print()

    P = args.patch
    probs_l, labs_l = [], []
    skipped = 0

    with torch.no_grad():
        for i, n in enumerate(names):
            a = cv2.imread(str(root / "Image1" / n), cv2.IMREAD_COLOR)
            b = cv2.imread(str(root / "Image2" / n), cv2.IMREAD_COLOR)
            lab = cv2.imread(str(root / "label" / n), cv2.IMREAD_GRAYSCALE)
            if a is None or b is None or lab is None:
                skipped += 1
                continue
            if a.shape[0] % P or a.shape[1] % P:
                skipped += 1
                continue

            ta, tb, tl = [], [], []
            for r in range(a.shape[0] // P):
                for c in range(a.shape[1] // P):
                    y0, x0 = r * P, c * P
                    ta.append(to_tensor_bgr(a[y0:y0 + P, x0:x0 + P]))
                    tb.append(to_tensor_bgr(b[y0:y0 + P, x0:x0 + P]))
                    tl.append((lab[y0:y0 + P, x0:x0 + P] > 127).astype(np.uint8))

            ta = torch.stack(ta).to(device)
            tb = torch.stack(tb).to(device)
            prob = model(ta, tb).softmax(1)[:, 1].reshape(-1).cpu()
            probs_l.append(prob)
            labs_l.append(torch.from_numpy(np.stack(tl).reshape(-1).astype(np.int64)))

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(names)} images")

    if not probs_l:
        raise SystemExit("没有可用图像")

    probs = torch.cat(probs_l)
    labs = torch.cat(labs_l)
    print(f"\n{len(probs):,} pixels accumulated, skipped {skipped} pairs")

    m = _metrics(probs, labs, thr)
    m50 = _metrics(probs, labs, 0.5)
    print(f"  @thr {thr:.2f}   F1 {m['F1']:.4f}  IoU {m['IoU']:.4f}  "
          f"P {m['P']:.4f}  R {m['R']:.4f}")
    print(f"  @argmax    F1 {m50['F1']:.4f}  IoU {m50['IoU']:.4f}  "
          f"P {m50['P']:.4f}  R {m50['R']:.4f}")

    best = m50
    for t in np.arange(0.20, 0.91, 0.05):
        mm = _metrics(probs, labs, float(t))
        if mm["F1"] > best["F1"]:
            best = mm
    print(f"  @best thr  F1 {best['F1']:.4f} @ {best['thr']:.2f}  (上界参考，不能用于论文)")


if __name__ == "__main__":
    main()
