"""probe_norm.py —— 跨域失败的归因实验

问题：LEVIR 模型在 S2Looking 上 F1 只有 0.048，到底是
  (a) 输入统计失配（亮度/色彩分布差了 1.5 个标准差，BN 统计全错）
  (b) 语义层面真的学不会

做法：模型不动，只换输入归一化，看 F1 能恢复到多少。
  (a) 成立 -> F1 会大幅跳起来，说明主要是统计问题
  仍然很低 -> 说明是语义问题，得靠训练解决

四种归一化：
  imagenet  当前训练时用的（基准）
  s2l_stats 用 S2Looking 自己的通道统计量替代 ImageNet 的
  instance  每张图各自减均值除标准差（最激进的域消除）
  raw       只做 [0,1] 缩放到 [-0.5,0.5]，不带任何统计假设

用法： uv run probe_norm.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data import IMAGENET_MEAN, IMAGENET_STD
from train import CDNet, _metrics

ROOT = Path(r"data/S2Looking\S2Looking")
P = 256


def to_rgb01(img_bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img_bgr[:, :, ::-1]).astype(np.float32) / 255.0


def norm_imagenet(img_bgr):
    return (to_rgb01(img_bgr) - IMAGENET_MEAN) / IMAGENET_STD


def make_norm_stats(stats):
    def fn(img_bgr):
        return (to_rgb01(img_bgr) - stats[0]) / stats[1]
    return fn


def norm_instance(img_bgr):
    x = to_rgb01(img_bgr)
    m = x.reshape(-1, 3).mean(0)
    s = x.reshape(-1, 3).std(0) + 1e-6
    return (x - m) / s


def norm_raw(img_bgr):
    return to_rgb01(img_bgr) - 0.5


def collect_stats(names, root):
    acc, acc2, n = np.zeros(3), np.zeros(3), 0
    for nm in names:
        img = cv2.imread(str(root / "Image1" / nm), cv2.IMREAD_COLOR)
        if img is None:
            continue
        x = to_rgb01(img).reshape(-1, 3)
        acc += x.sum(0)
        acc2 += (x ** 2).sum(0)
        n += x.shape[0]
    mean = acc / n
    var = acc2 / n - mean ** 2
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)


def run(model, names, root, norm_fn, device, thr):
    model.eval()
    probs_l, labs_l = [], []
    with torch.no_grad():
        for nm in names:
            a = cv2.imread(str(root / "Image1" / nm), cv2.IMREAD_COLOR)
            b = cv2.imread(str(root / "Image2" / nm), cv2.IMREAD_COLOR)
            lab = cv2.imread(str(root / "label" / nm), cv2.IMREAD_GRAYSCALE)
            if a is None or b is None or lab is None:
                continue

            ta, tb, tl = [], [], []
            for r in range(a.shape[0] // P):
                for c in range(a.shape[1] // P):
                    y0, x0 = r * P, c * P
                    ta.append(norm_fn(a[y0:y0 + P, x0:x0 + P]))
                    tb.append(norm_fn(b[y0:y0 + P, x0:x0 + P]))
                    tl.append((lab[y0:y0 + P, x0:x0 + P] > 127).astype(np.uint8))

            ta = torch.from_numpy(np.stack(ta).transpose(0, 3, 1, 2)).to(device)
            tb = torch.from_numpy(np.stack(tb).transpose(0, 3, 1, 2)).to(device)
            prob = model(ta, tb).softmax(1)[:, 1].reshape(-1).cpu()
            probs_l.append(prob)
            labs_l.append(torch.from_numpy(np.stack(tl).reshape(-1).astype(np.int64)))

    return torch.cat(probs_l), torch.cat(labs_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base_r40/best.pt")
    ap.add_argument("--n-images", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    thr = float(ckpt.get("thr", 0.5))
    model = CDNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    root = ROOT / "test"
    names = sorted(p.name for p in (root / "Image1").glob("*.png"))[:args.n_images]
    print(f"ckpt {args.ckpt} | {len(names)} images | thr {thr:.2f} | device {device}")

    stats = collect_stats(names, root)
    print(f"S2Looking test stats: mean {stats[0].round(3)} std {stats[1].round(3)}")
    print(f"ImageNet mean/std  : {IMAGENET_MEAN.round(3)} / {IMAGENET_STD.round(3)}")
    print()

    variants = [
        ("imagenet", norm_imagenet),
        ("s2l_stats", make_norm_stats(stats)),
        ("instance", norm_instance),
        ("raw", norm_raw),
    ]

    print(f"{'norm':<12} {'F1':>7} {'IoU':>7} {'P':>7} {'R':>7}  {'@bestthr':>9}")
    for name, fn in variants:
        probs, labs = run(model, names, root, fn, device, thr)
        m = _metrics(probs, labs, thr)
        best = m
        for t in np.arange(0.10, 0.91, 0.05):
            mm = _metrics(probs, labs, float(t))
            if mm["F1"] > best["F1"]:
                best = mm
        print(f"{name:<12} {m['F1']:>7.4f} {m['IoU']:>7.4f} {m['P']:>7.4f} {m['R']:>7.4f}  "
              f"{best['F1']:.4f}@{best['thr']:.2f}")


if __name__ == "__main__":
    main()
