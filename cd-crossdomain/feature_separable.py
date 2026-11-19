"""feature_separable.py —— 伪变化误报与真实检出的特征可分性

背景：四个抑制伪变化的方案全部失败。一个可能的原因是——伪变化区域与真实变化
区域在特征空间里高度纠缠，任何压制"差异"的机制都会同时压掉真实变化。

本脚本直接验证这一点：取模型融合层（fuse1 输出，64 通道）的特征，
分别在「伪变化误报像素」与「真实检出像素」上采样，测两类的可分性。

判据：最近质心分类器在两类上的准确率。
  50%  = 完全不可分（特征上没区别）
  100% = 完全可分

用法：
  uv run feature_separable.py --ckpt runs/base_seed0/best.pt --root data/patches_whu --split test
输出：
  vis/feature_sep.png  PCA 二维散点图
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import PATCH_ROOT, PatchDataset, build_loader
from train import CDNet

MAX_PER_PATCH = 20


def downsample_mask(mask: np.ndarray, k: int) -> np.ndarray:
    h, w = mask.shape
    h2, w2 = h // k, w // k
    return mask[: h2 * k, : w2 * k].reshape(h2, k, w2, k).mean(axis=(1, 3)) > 0.5


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base_seed0/best.pt")
    ap.add_argument("--root", default=str(PATCH_ROOT))
    ap.add_argument("--split", default="test")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=6000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = CDNet(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    feat_box: dict[str, torch.Tensor] = {}

    def hook(_m, _i, o):
        feat_box["f"] = o

    model.fuse1.register_forward_hook(hook)

    root = Path(args.root)
    keys = sorted(p.stem for p in (root / args.split / "L").glob("*.npy"))
    ds = PatchDataset(keys, split=args.split, root=root, sync_aug=False)
    dl = build_loader(ds, args.batch_size, False, 4)

    rng = np.random.default_rng(0)
    feats_pseudo: list[np.ndarray] = []
    feats_real: list[np.ndarray] = []

    for a, b, lab, _ in dl:
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        logits = model(a, b)
        f = feat_box["f"].cpu().numpy()          # (B,64,64,64) -> C,H,W
        probs = logits.softmax(1)[:, 1].cpu().numpy()
        labs = lab.numpy()

        for i in range(probs.shape[0]):
            pred = probs[i] > args.thr
            gt = labs[i] > 0
            if not pred.any():
                continue

            p = downsample_mask(pred, 4)         # (64,64)
            g = downsample_mask(gt, 4)
            feat = f[i].transpose(1, 2, 0)       # (64,64,64)

            # 伪变化误报：预测为变化、真值无变化，且整块真值为空
            if not gt.any():
                m = p & ~g
                idx = np.argwhere(m)
                if len(idx) > MAX_PER_PATCH:
                    idx = idx[rng.choice(len(idx), MAX_PER_PATCH, replace=False)]
                if len(idx):
                    feats_pseudo.append(feat[idx[:, 0], idx[:, 1]])

            # 真实检出：预测与真值都是变化
            m = p & g
            if m.any():
                idx = np.argwhere(m)
                if len(idx) > MAX_PER_PATCH:
                    idx = idx[rng.choice(len(idx), MAX_PER_PATCH, replace=False)]
                feats_real.append(feat[idx[:, 0], idx[:, 1]])

        if len(feats_pseudo) > args.max_samples and len(feats_real) > args.max_samples:
            break

    X0 = np.concatenate(feats_pseudo)[: args.max_samples]
    X1 = np.concatenate(feats_real)[: args.max_samples]
    n0, n1 = len(X0), len(X1)

    print(f"ckpt {args.ckpt}")
    print(f"伪变化像素样本 {n0}，真实检出像素样本 {n1}，特征维度 {X0.shape[1]}")
    print()

    mu0, mu1 = X0.mean(0), X1.mean(0)
    sd_pool = 0.5 * (X0.std(0) + X1.std(0)) + 1e-6
    # 类间距离 / 类内标准差：越小越纠缠
    effect = float(np.linalg.norm(mu0 - mu1) / np.linalg.norm(sd_pool))
    print(f"两类质心距离 / 类内标准差 = {effect:.3f}")

    # 最近质心分类准确率（对称）
    d00 = ((X0 - mu0) ** 2).sum(1)
    d01 = ((X0 - mu1) ** 2).sum(1)
    acc0 = float((d00 < d01).mean())
    d10 = ((X1 - mu0) ** 2).sum(1)
    d11 = ((X1 - mu1) ** 2).sum(1)
    acc1 = float((d11 < d10).mean())
    acc = 0.5 * (acc0 + acc1)
    print(f"最近质心分类准确率: 伪变化 {acc0:.3f}  真实 {acc1:.3f}  平均 {acc:.3f}")
    print("  （0.5 = 完全不可分，1.0 = 完全可分）")

    # PCA 二维投影
    X = np.concatenate([X0, X1])
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    evr = (S ** 2 / (S ** 2).sum())[:2]

    out = Path("vis")
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:n0, 0], Z[:n0, 1], s=4, alpha=0.4, label="pseudo-change FP")
    plt.scatter(Z[n0:, 0], Z[n0:, 1], s=4, alpha=0.4, label="true positive")
    plt.xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    plt.title(f"fuse1 features: centroid acc = {acc:.3f}")
    plt.legend(markerscale=3)
    plt.tight_layout()
    p = out / "feature_sep.png"
    plt.savefig(p, dpi=130)
    plt.close()
    print(f"\nPCA 图 -> {p}")


if __name__ == "__main__":
    main()
