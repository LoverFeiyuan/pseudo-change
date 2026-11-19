"""bench_protocol.py —— 统一评测协议

之前每个数字用的子集和阈值都不一样，结论会随协议翻来覆去。这里定死：

  * 数据：每个域的完整 test 集，不截断
  * 阈值：主表统一用 0.5（跨域时"在源域 val 上选的阈值"其实也不可靠，
    统一 0.5 是最干净的比法）；另外报 best-thr 上界仅作参考，不进主表
  * 模型：同一套 checkpoint，同一套预处理

用法：
  uv run bench_protocol.py --domains LEVIR-test,WHU-test
  uv run bench_protocol.py --domains S2L-test
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data import IMAGENET_MEAN, IMAGENET_STD, PatchDataset, build_loader
from train import CDNet, _metrics

DATA = Path(r"data")

DOMAINS = {
    "LEVIR-test": ("patch", DATA / "patches", "test"),
    "WHU-test": ("patch", DATA / "patches_whu", "test"),
    "S2L-test": ("s2l", DATA / "S2Looking" / "S2Looking", "test"),
}

MODELS = [
    ("base_pw3", "runs/base_pw3/best.pt", "10% · 基线"),
    ("base_r40", "runs/base_r40/best.pt", "40% · 基线"),
    ("pa_r10", "runs/pa_r10/best.pt", "10% · 数据集增广"),
    ("cons_r10", "runs/cons_r10/best.pt", "10% · 一致性"),
    ("cons_r10b", "runs/cons_r10b/best.pt", "10% · 一致性(对齐)"),
    ("cons_r10c", "runs/cons_r10c/best.pt", "10% · 一致性(宽通道)"),
    ("modB_only_r10", "runs/modB_only_r10/best.pt", "10% · 模块B(零变化)"),
    ("modAB_r10", "runs/modAB_r10/best.pt", "10% · A+B 组合"),
    ("modB_b6_r10", "runs/modB_b6_r10/best.pt", "10% · 模块B(batch6)"),
    ("modB_z1_r10", "runs/modB_z1_r10/best.pt", "10% · 模块B(zero=1)"),
    ("modB_z4_r10", "runs/modB_z4_r10/best.pt", "10% · 模块B(zero=4)"),
    ("modB_r40", "runs/modB_r40/best.pt", "40% · 模块B(zero=2)"),
    ("modB_z1_r40", "runs/modB_z1_r40/best.pt", "40% · 模块B(zero=1)"),
    ("seed0", "runs/seed0/best.pt", "10% · 模块B seed0"),
    ("seed1", "runs/seed1/best.pt", "10% · 模块B seed1"),
    ("seed2", "runs/seed2/best.pt", "10% · 模块B seed2"),
    ("base_seed0", "runs/base_seed0/best.pt", "10% · 基线 seed0"),
    ("base_seed1", "runs/base_seed1/best.pt", "10% · 基线 seed1"),
    ("base_seed2", "runs/base_seed2/best.pt", "10% · 基线 seed2"),
]

P = 256


def to_tensor_bgr(img: np.ndarray) -> torch.Tensor:
    x = np.ascontiguousarray(img[:, :, ::-1]).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


@torch.no_grad()
def probs_patch(model, root: Path, split: str, device: str):
    keys = sorted(p.stem for p in (Path(root) / split / "L").glob("*.npy"))
    if not keys:
        return None, None
    ds = PatchDataset(keys, split=split, root=root, sync_aug=False)
    dl = build_loader(ds, 8, False, 4)
    ps, ls = [], []
    for a, b, lab, _ in dl:
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        ps.append(model(a, b).softmax(1)[:, 1].reshape(-1).cpu())
        ls.append(lab.reshape(-1))
    return torch.cat(ps), torch.cat(ls)


@torch.no_grad()
def probs_s2l(model, root: Path, split: str, device: str, limit: int = 0):
    base = Path(root) / split
    names = sorted(p.name for p in (base / "Image1").glob("*.png"))
    if limit:
        names = names[:limit]
    ps, ls = [], []
    for n in names:
        a = cv2.imread(str(base / "Image1" / n), cv2.IMREAD_COLOR)
        b = cv2.imread(str(base / "Image2" / n), cv2.IMREAD_COLOR)
        lab = cv2.imread(str(base / "label" / n), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or lab is None:
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
        ps.append(model(ta, tb).softmax(1)[:, 1].reshape(-1).cpu())
        ls.append(torch.from_numpy(np.stack(tl).reshape(-1).astype(np.int64)))
    if not ps:
        return None, None
    return torch.cat(ps), torch.cat(ls)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="LEVIR-test,WHU-test")
    ap.add_argument("--limit", type=int, default=0, help="S2L 图数上限，0=全量")
    ap.add_argument("--models", default="", help="逗号分隔，只跑这些模型")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wanted = [d.strip() for d in args.domains.split(",") if d.strip()]

    header = f"{'model':<12} {'setting':<18} {'domain':<12} {'n':>7} " \
             f"{'F1@0.5':>8} {'P@0.5':>8} {'R@0.5':>8} {'bestF1':>8} {'bestThr':>8}"
    print(header)
    print("-" * len(header))

    only = {m.strip() for m in args.models.split(",") if m.strip()}
    for name, ckpt_path, setting in MODELS:
        if only and name not in only:
            continue
        p = Path(ckpt_path)
        if not p.exists():
            print(f"{name:<12} {setting:<18} (missing checkpoint)")
            continue
        ck = torch.load(p, map_location=device, weights_only=False)
        model = CDNet(pretrained=False).to(device)
        model.load_state_dict(ck["model"])
        model.eval()

        for dom in wanted:
            kind, root, split = DOMAINS[dom]
            if kind == "patch":
                probs, labs = probs_patch(model, root, split, device)
            else:
                probs, labs = probs_s2l(model, root, split, device, args.limit)
            if probs is None:
                print(f"{name:<12} {setting:<18} {dom:<12} (empty)")
                continue

            m05 = _metrics(probs, labs, 0.5)
            best = m05
            for t in np.arange(0.05, 0.96, 0.05):
                mm = _metrics(probs, labs, float(t))
                if mm["F1"] > best["F1"]:
                    best = mm
            print(f"{name:<12} {setting:<18} {dom:<12} {len(probs):>7} "
                  f"{m05['F1']:>8.4f} {m05['P']:>8.4f} {m05['R']:>8.4f} "
                  f"{best['F1']:>8.4f} {best['thr']:>8.2f}")
            del probs, labs

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
