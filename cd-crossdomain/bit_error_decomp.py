"""bit_error_decomp.py —— 用 BIT 作为源模型重做误差分解

目的：验证第 5 节的误差结构（伪变化占 40%+）是"轻量孪生网络"的特性，
还是跨域变化的通性。若 BIT 也给出相近的分解，则论文的结论不再局限于单一架构。

BIT 官方预处理是 [-1,1] 的 RGB（不是 ImageNet），输出为 2 通道 logits。

用法：
  uv run bit_error_decomp.py --root data/patches_whu --split test
  uv run bit_error_decomp.py --root data/patches --split test      # 源域对照
"""
import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

# —— BIT 官方代码的兼容补丁（torchvision 旧 API）——
import torch.hub
import torchvision.models as _tvm

if not hasattr(_tvm, "utils"):
    _m = types.ModuleType("torchvision.models.utils")
    _m.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    sys.modules["torchvision.models.utils"] = _m

sys.path.insert(0, r"third_party/BIT_CD")
from models.networks import BASE_Transformer  # noqa: E402

CKPT = r"data/LEVIR-CD\BIT_LEVIR\best_ckpt.pt"


def build_bit(device):
    net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4,
                           resnet_stages_num=4, with_pos="learned",
                           enc_depth=1, dec_depth=8, decoder_dim_head=8)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict(ck["model_G_state_dict"], strict=False)
    assert not missing and not unexpected, f"权重不匹配: {len(missing)}/{len(unexpected)}"
    return net.to(device).eval(), ck.get("best_val_acc")


def to_bit_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    x = np.ascontiguousarray(img_bgr[:, :, ::-1]).astype(np.float32) / 255.0
    x = x * 2.0 - 1.0
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def f1_counts(tp, fp, fn):
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9), p, r


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"data/patches_whu")
    ap.add_argument("--split", default="test")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--near-tol", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, val_acc = build_bit(device)

    root = Path(args.root)
    keys = sorted(p.stem for p in (root / args.split / "L").glob("*.npy"))
    if not keys:
        raise SystemExit(f"没找到样本：{root / args.split / 'L'}")

    print(f"BIT on LEVIR (val acc {val_acc:.4f})")
    print(f"root {root}  split {args.split}  n={len(keys)}  thr={args.thr}")
    print()

    tp = fn = fp_zero = fp_near = fp_far = 0
    zero_patches = 0
    bs = args.batch_size

    for start in range(0, len(keys), bs):
        chunk = keys[start:start + bs]
        ta, tb, tl = [], [], []
        for k in chunk:
            a = np.load(root / args.split / "A" / f"{k}.npy")
            b = np.load(root / args.split / "B" / f"{k}.npy")
            lab = np.load(root / args.split / "L" / f"{k}.npy")
            ta.append(to_bit_tensor(a))
            tb.append(to_bit_tensor(b))
            tl.append(lab)

        ta = torch.stack(ta).to(device)
        tb = torch.stack(tb).to(device)
        probs = net(ta, tb).softmax(1)[:, 1].cpu().numpy()

        for i in range(len(chunk)):
            pred = probs[i] > args.thr
            gt = tl[i] > 0

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
            dist = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 5)
            near = fp_mask & (dist <= args.near_tol)
            fp_near += int(near.sum())
            fp_far += n_fp - int(near.sum())

        if (start // bs + 1) % 50 == 0:
            print(f"  {start + len(chunk)}/{len(keys)}")

    fp_total = fp_zero + fp_near + fp_far
    f1, p, r = f1_counts(tp, fp_total, fn)

    print()
    print(f"总 patch {len(keys)}，真值全零 {zero_patches}")
    print(f"TP {tp:,}  FN {fn:,}  总 FP {fp_total:,}")
    print(f"基线：F1 {f1:.4f}  P {p:.4f}  R {r:.4f}")
    print()
    print(f"{'类别':<28} {'FP 像素':>12} {'占 FP':>8} {'消除后 F1':>11} {'ΔF1':>9}")
    print("-" * 72)
    for label, cnt in (("伪变化（零变化区）", fp_zero),
                       (f"边界误报（≤{args.near_tol}px）", fp_near),
                       ("远端误报", fp_far)):
        share = 100 * cnt / max(fp_total, 1)
        after, _, _ = f1_counts(tp, fp_total - cnt, fn)
        print(f"{label:<28} {cnt:>12,} {share:>7.1f}% {after:>11.4f} {after - f1:>+9.4f}")


if __name__ == "__main__":
    main()
