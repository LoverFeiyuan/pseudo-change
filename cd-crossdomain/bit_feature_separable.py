"""bit_feature_separable.py —— 用 BIT 复现特征可分性分析

第 5 节的误差分解已做跨架构验证，但第 7 节的可分性只在自家模型上测过。
本脚本补上这一块：若 BIT 也呈现"伪变化易分、真实变化难分"的不对称，
则嵌入结构的结论同样不限于单一架构。

BIT 的 conv_pred 输出为 32 通道、全分辨率（256×256），可直接逐像素采样。

用法：
  uv run bit_feature_separable.py --root data/patches_whu --split test
"""
import argparse
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.hub
import torchvision.models as _tvm

if not hasattr(_tvm, "utils"):
    _m = types.ModuleType("torchvision.models.utils")
    _m.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    sys.modules["torchvision.models.utils"] = _m

sys.path.insert(0, r"third_party/BIT_CD")
from models.networks import BASE_Transformer  # noqa: E402

CKPT = r"data/LEVIR-CD\BIT_LEVIR\best_ckpt.pt"
MAX_PER_PATCH = 20


def build_bit(device):
    net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4,
                           resnet_stages_num=4, with_pos="learned",
                           enc_depth=1, dec_depth=8, decoder_dim_head=8)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict(ck["model_G_state_dict"], strict=False)
    assert not missing and not unexpected, "权重不匹配"
    return net.to(device).eval()


def to_bit_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    x = np.ascontiguousarray(img_bgr[:, :, ::-1]).astype(np.float32) / 255.0
    x = x * 2.0 - 1.0
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"data/patches_whu")
    ap.add_argument("--split", default="test")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=6000)
    ap.add_argument("--tag", default="bit_whu")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_bit(device)
    box = {}
    net.classifier[0].register_forward_hook(lambda _m, _i, o: box.update(f=o))

    root = Path(args.root)
    keys = sorted(p.stem for p in (root / args.split / "L").glob("*.npy"))
    print(f"BIT | root {root} | n={len(keys)} | thr={args.thr}")

    rng = np.random.default_rng(0)
    fp_list, tp_list = [], []
    bs = args.batch_size

    for start in range(0, len(keys), bs):
        chunk = keys[start:start + bs]
        ta, tb, tl = [], [], []
        for k in chunk:
            ta.append(to_bit_tensor(np.load(root / args.split / "A" / f"{k}.npy")))
            tb.append(to_bit_tensor(np.load(root / args.split / "B" / f"{k}.npy")))
            tl.append(np.load(root / args.split / "L" / f"{k}.npy"))

        ta = torch.stack(ta).to(device)
        tb = torch.stack(tb).to(device)
        out = net(ta, tb)
        probs = out.softmax(1)[:, 1].cpu().numpy()
        feats = box["f"].cpu().numpy()          # (B,32,256,256)

        for i in range(len(chunk)):
            pred = probs[i] > args.thr
            gt = tl[i] > 0
            if not pred.any():
                continue
            feat = feats[i].transpose(1, 2, 0)   # (256,256,32)

            if not gt.any():
                idx = np.argwhere(pred & ~gt)
                if len(idx) > MAX_PER_PATCH:
                    idx = idx[rng.choice(len(idx), MAX_PER_PATCH, replace=False)]
                if len(idx):
                    fp_list.append(feat[idx[:, 0], idx[:, 1]])

            idx = np.argwhere(pred & gt)
            if len(idx):
                if len(idx) > MAX_PER_PATCH:
                    idx = idx[rng.choice(len(idx), MAX_PER_PATCH, replace=False)]
                tp_list.append(feat[idx[:, 0], idx[:, 1]])

        if len(fp_list) > args.max_samples and len(tp_list) > args.max_samples:
            break

    X0 = np.concatenate(fp_list)[: args.max_samples]
    X1 = np.concatenate(tp_list)[: args.max_samples]
    n0, n1 = len(X0), len(X1)
    print(f"伪变化样本 {n0}，真实检出样本 {n1}，维度 {X0.shape[1]}")

    mu0, mu1 = X0.mean(0), X1.mean(0)
    sd_pool = 0.5 * (X0.std(0) + X1.std(0)) + 1e-6
    effect = float(np.linalg.norm(mu0 - mu1) / np.linalg.norm(sd_pool))

    d00 = ((X0 - mu0) ** 2).sum(1)
    d01 = ((X0 - mu1) ** 2).sum(1)
    acc0 = float((d00 < d01).mean())
    d10 = ((X1 - mu0) ** 2).sum(1)
    d11 = ((X1 - mu1) ** 2).sum(1)
    acc1 = float((d11 < d10).mean())
    acc = 0.5 * (acc0 + acc1)

    print(f"质心距离/类内标准差 = {effect:.3f}")
    print(f"最近质心准确率：伪变化 {acc0:.3f}  真实 {acc1:.3f}  平均 {acc:.3f}")
    print(f"  → 真实变化误杀下界 {100 * (1 - acc1):.1f}%")

    X = np.concatenate([X0, X1])
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    evr = (S ** 2 / (S ** 2).sum())[:2]

    out_dir = Path("vis")
    out_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:n0, 0], Z[:n0, 1], s=4, alpha=0.4, label="pseudo-change FP")
    plt.scatter(Z[n0:, 0], Z[n0:, 1], s=4, alpha=0.4, label="true positive")
    plt.xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    plt.title(f"BIT features: centroid acc = {acc:.3f}")
    plt.legend(markerscale=3)
    plt.tight_layout()
    p = out_dir / f"feature_sep_{args.tag}.png"
    plt.savefig(p, dpi=130)
    plt.close()
    print(f"PCA 图 -> {p}")


if __name__ == "__main__":
    main()
