"""make_figure_qualitative.py —— 定性结果对比图

三行（源域 / 温和目标域 / 极端目标域）× 五列（前时相、后时相、真值、本文预测、BIT 预测）。

选取有代表性的样本：源域取预测准确的，目标域取伪变化误报明显的。
两个模型的预处理不同（本文 ImageNet，BIT [-1,1]），分别处理。
"""
import sys
import types
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.hub
import torchvision.models as _tvm

from data import PATCH_ROOT
from train import CDNet

if not hasattr(_tvm, "utils"):
    _m = types.ModuleType("torchvision.models.utils")
    _m.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    sys.modules["torchvision.models.utils"] = _m
sys.path.insert(0, r"third_party/BIT_CD")
from models.networks import BASE_Transformer  # noqa: E402

DATA = Path(r"data")
FIG = Path("figures")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OURS_CKPT = "runs/base_seed0/best.pt"
BIT_CKPT = r"data/LEVIR-CD\BIT_LEVIR\best_ckpt.pt"


def load_ours(device):
    ck = torch.load(OURS_CKPT, map_location=device, weights_only=False)
    net = CDNet(pretrained=False).to(device)
    net.load_state_dict(ck["model"])
    return net.eval()


def load_bit(device):
    net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4,
                           resnet_stages_num=4, with_pos="learned",
                           enc_depth=1, dec_depth=8, decoder_dim_head=8)
    ck = torch.load(BIT_CKPT, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model_G_state_dict"])
    return net.to(device).eval()


def norm_ours(img_bgr):
    x = np.ascontiguousarray(img_bgr[:, :, ::-1]).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


def norm_bit(img_bgr):
    x = np.ascontiguousarray(img_bgr[:, :, ::-1]).astype(np.float32) / 255.0
    x = x * 2.0 - 1.0
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


def pick_patch(root: Path, split: str, lo: float, hi: float, idx: int = 0):
    import csv
    csv_path = root / ("patch_stats_" + root.name.replace("patches_", "") + ".csv")
    if not csv_path.exists():
        csv_path = root / "patch_stats.csv"
    if not csv_path.exists():
        cand = sorted(p.stem for p in (root / split / "L").glob("*.npy"))
        return cand[idx]
    with open(csv_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    cand = [r for r in rows if lo <= float(r["change_ratio"]) <= hi]
    if not cand:
        cand = rows
    cand.sort(key=lambda r: -float(r["change_ratio"]))
    return cand[min(idx, len(cand) - 1)]["key"]


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ours = load_ours(device)
    bit = load_bit(device)

    cases = [
        ("LEVIR-CD\n(source)", DATA / "patches", "test", 0.05, 0.12, 0),
        ("WHU-CD\n(target, +30%)", DATA / "patches_whu", "test", 0.02, 0.06, 0),
        ("S2Looking\n(target, −63%)", DATA / "patches_s2l", "test", 0.02, 0.06, 0),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(11.5, 7.6))
    titles = ["Pre-event", "Post-event", "Ground truth", "Ours (10%)", "BIT"]

    for r, (name, root, split, lo, hi, idx) in enumerate(cases):
        key = pick_patch(root, split, lo, hi, idx)
        a = np.load(root / split / "A" / f"{key}.npy")
        b = np.load(root / split / "B" / f"{key}.npy")
        lab = np.load(root / split / "L" / f"{key}.npy")

        p_ours = ours(norm_ours(a).to(device), norm_ours(b).to(device)).softmax(1)[0, 1].cpu().numpy()
        p_bit = bit(norm_bit(a).to(device), norm_bit(b).to(device)).softmax(1)[0, 1].cpu().numpy()

        panels = [
            a[:, :, ::-1],
            b[:, :, ::-1],
            lab,
            (p_ours > 0.5).astype(np.uint8),
            (p_bit > 0.5).astype(np.uint8),
        ]

        for c, img in enumerate(panels):
            ax = axes[r, c]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=9.5)
        print(f"{name.splitlines()[0]:<24} {key}")

    fig.tight_layout()
    out = FIG / "fig6_qualitative.png"
    fig.savefig(out, bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
