"""train.py —— 基线训练

模型：孪生 ResNet-18（共享权重）+ 四级 skip 融合解码
损失：加权 CE + Dice（Dice 直接优化重叠度，对 5% 正像素这种极端不平衡更友好）
评估：在 val 上搜索最优决策阈值，而不是固定 argmax(0.5)

为什么要搜阈值：模型输出的概率分布整体偏高，固定 0.5 会让 Precision 一直上不去。
搜阈值是零成本的，而且论文里报「阈值在 val 上选定、apply 到 test」是标准流程。

用法：
  uv run train.py --epochs 30 --pos-weight 3 --lr 1e-4
  uv run train.py --ratio 0.05 --epochs 30 --batch-size 4
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data import PATCH_ROOT, PatchDataset, build_loader


class ConvBN(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class CDNet(nn.Module):
    """Siamese ResNet-18（权重共享）+ 四级 skip 融合解码，输出全分辨率 logits。"""

    def __init__(self, arch: str = "resnet18", pretrained: bool = True):
        super().__init__()
        try:
            r = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        except Exception as e:
            print(f"! 预训练权重不可用（{type(e).__name__}），改用随机初始化")
            r = getattr(torchvision.models, arch)(weights=None)

        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4

        self.reduce4 = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.fuse3 = ConvBN(256 + 128, 128)
        self.fuse2 = ConvBN(128 + 128, 64)
        self.fuse1 = ConvBN(64 + 64, 64)
        self.head = nn.Sequential(ConvBN(64, 32, 1), nn.Conv2d(32, 2, 1))

    def enc(self, x):
        x = self.stem(x)
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        return f1, f2, f3, f4

    def forward(self, a, b):
        a1, a2, a3, a4 = self.enc(a)
        b1, b2, b3, b4 = self.enc(b)

        x = self.reduce4(torch.abs(a4 - b4))
        x = F.interpolate(x, size=a3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse3(torch.cat([x, torch.abs(a3 - b3)], 1))
        x = F.interpolate(x, size=a2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([x, torch.abs(a2 - b2)], 1))
        x = F.interpolate(x, size=a1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse1(torch.cat([x, torch.abs(a1 - b1)], 1))
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.head(x)


class CombinedLoss(nn.Module):
    """加权 CE + Dice。Dice 对类别极端不平衡比纯 CE 稳得多。"""

    def __init__(self, ce_weight: torch.Tensor, dice_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, target):
        loss = self.ce(logits, target)
        if self.dice_weight <= 0:
            return loss
        prob = logits.softmax(1)[:, 1]
        t = target.float()
        inter = (prob * t).sum()
        union = prob.sum() + t.sum()
        dice = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        return loss + self.dice_weight * dice


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_norm(x_uint8: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) BGR uint8 -> ImageNet 归一化的 RGB float。"""
    global _MEAN, _STD
    x = x_uint8.flip(1).float() / 255.0
    if _MEAN.device != x.device:
        _MEAN = _MEAN.to(x.device)
        _STD = _STD.to(x.device)
    return (x - _MEAN) / _STD


def photo_perturb(x_uint8: torch.Tensor) -> torch.Tensor:
    """按样本独立的成像统计扰动，返回 uint8。

    两层增益，和 data.PatchDataset._photo 保持完全一致：
      g1 ∈ [0.75, 1.30] 每通道
      g2 ∈ [0.45, 1.70] 全局
    合起来 0.34–2.21，下限 0.34 是为了覆盖 S2Looking 那种 −63% 的亮度衰减
    （源域亮度 102.7，S2L 是 38.5，比值 0.375）。
    """
    x = x_uint8.float()
    bsz = x.size(0)
    dev = x.device

    g1 = torch.rand(bsz, 3, 1, 1, device=dev) * 0.55 + 0.75
    g2 = torch.rand(bsz, 1, 1, 1, device=dev) * 1.05 + 0.55
    x = x * g1 * g2 + (torch.rand(bsz, 1, 1, 1, device=dev) * 60.0 - 30.0)

    if float(torch.rand(())) < 0.5:
        gamma = torch.rand(bsz, 1, 1, 1, device=dev) * 0.6 + 0.75
        x = 255.0 * (x.clamp(0.0, 255.0) / 255.0) ** gamma
    return x.clamp(0.0, 255.0).to(torch.uint8)


def _metrics(probs: torch.Tensor, labs: torch.Tensor, thr: float) -> dict:
    pred = probs > thr
    pos = labs == 1
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    return {"F1": f1, "IoU": iou, "P": p, "R": r, "thr": thr}


@torch.no_grad()
def evaluate(model, loader, device, search_threshold: bool = True):
    """返回 (最优阈值下的指标, 该阈值, 固定 0.5 时的指标)"""
    model.eval()
    probs_l, labs_l = [], []
    for a, b, lab, _ in loader:
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        prob = model(a, b).softmax(1)[:, 1]
        probs_l.append(prob.reshape(-1).cpu())
        labs_l.append(lab.reshape(-1).cpu())

    probs = torch.cat(probs_l)
    labs = torch.cat(labs_l)

    m50 = _metrics(probs, labs, 0.5)
    if not search_threshold:
        return m50, 0.5, m50

    best = m50
    for t in np.arange(0.20, 0.91, 0.05):
        m = _metrics(probs, labs, float(t))
        if m["F1"] > best["F1"]:
            best = m
    del probs, labs
    return best, best["thr"], m50


def load_keys(split: str) -> list[str]:
    keys = []
    with open(PATCH_ROOT / "patch_stats.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                keys.append(r["key"])
    return keys


def load_zero_keys() -> list[str]:
    """「整图零变化」的 patch —— 模块 B 的负样本来源。

    判据：同一张 1024² 图切出来的 16 个 patch **全部** change_ratio == 0。
    这类图双时相之间有真实的季节/光照差异（否则两期影像会完全一致），
    但没有任何语义变化。它们就是「伪变化」的天然样本，
    比随便找一片空地当负样本有价值得多——空地本身没有差异可学。
    """
    rows: list[tuple[str, str, float]] = []
    with open(PATCH_ROOT / "patch_stats.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                rows.append((r["name"], r["key"], float(r["change_ratio"])))

    by_name: dict[str, list[float]] = {}
    for name, _, ratio in rows:
        by_name.setdefault(name, []).append(ratio)

    zero_names = {n for n, rs in by_name.items() if max(rs) == 0.0}
    keys = [k for n, k, _ in rows if n in zero_names]
    print(f"zero-change patches: {len(keys)} from {len(zero_names)} images")
    return keys


def infinite_loader(loader):
    """无限循环取 batch，不在内存里缓存整个 epoch。"""
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", default="0.10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--cons-weight", type=float, default=0.0,
                    help="成像统计一致性约束的权重，>0 时启用双前向")
    ap.add_argument("--cons-thr", type=float, default=0.2,
                    help="局部化阈值：只约束两路概率差超过该值的像素")
    ap.add_argument("--cons-scope", choices=["global", "local"], default="global",
                    help="global=全图约束（模块A原版）；local=只在预测不一致处约束")
    ap.add_argument("--zero-per-batch", type=int, default=0,
                    help="每个 batch 混入几个「整图零变化」patch（模块 B）")
    ap.add_argument("--train-seed", type=int, default=0,
                    help="训练随机种子（权重初始化、数据顺序）；与 --seed 的标注集采样无关")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="runs/base")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)

    splits = json.loads((PATCH_ROOT / "splits.json").read_text(encoding="utf-8"))
    tag = f"{float(args.ratio):.2f}"
    if tag not in splits["splits"]:
        raise SystemExit(f"ratio {tag} 不在 splits.json 里，可选：{list(splits['splits'])}")
    train_keys = splits["splits"][tag][f"seed{args.seed}"]["labeled"]
    val_keys = load_keys("val")

    use_cons = args.cons_weight > 0
    train_ds = PatchDataset(train_keys, split="train", sync_aug=True, raw=use_cons)

    zero_iter = None
    if args.zero_per_batch > 0:
        zero_keys = load_zero_keys()
        if zero_keys:
            zds = PatchDataset(zero_keys, split="train", sync_aug=True, raw=use_cons)
            zdl = build_loader(zds, args.zero_per_batch, True, args.workers)
            zero_iter = infinite_loader(zdl)
        else:
            print("! 没找到零变化 patch，模块 B 跳过")
    val_ds = PatchDataset(val_keys, split="val", sync_aug=False)
    train_loader = build_loader(train_ds, args.batch_size, True, args.workers)
    val_loader = build_loader(val_ds, args.batch_size, False, args.workers)

    model = CDNet(pretrained=not args.no_pretrained).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"device {device} | params {n_param / 1e6:.2f}M | train {len(train_ds)} | "
          f"val {len(val_ds)} | ratio {tag} seed{args.seed}")
    print(f"lr {args.lr} | pos_weight {args.pos_weight} | dice_weight {args.dice_weight} | "
          f"cons_weight {args.cons_weight} | cons_scope {args.cons_scope} | "
          f"train_seed {args.train_seed} | epochs {args.epochs}")

    ce_w = torch.tensor([1.0, args.pos_weight], device=device)
    criterion = CombinedLoss(ce_w, args.dice_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        tot, n = 0.0, 0

        for a, b, lab, _ in train_loader:
            if zero_iter is not None:
                za, zb, zlab, _ = next(zero_iter)
                a = torch.cat([a, za], 0)
                b = torch.cat([b, zb], 0)
                lab = torch.cat([lab, zlab], 0)

            a = a.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)

            if use_cons:
                logits = model(to_norm(a), to_norm(b))
                loss = criterion(logits, lab)

                logits_p = model(to_norm(photo_perturb(a)), to_norm(photo_perturb(b)))
                # stop-gradient：以原图预测为目标，避免两路一起塌缩
                target = logits.softmax(1).detach()
                p_pert = logits_p.softmax(1)

                if args.cons_scope == "local":
                    # 只在两路打分不一致的像素上施加约束。
                    # 模型高置信的区域（含大部分 TP）diff 接近 0，掩码为假，梯度不流向那里，
                    # 因此不会被整体压扁——这正是模块 A 全局版掉 TP 的原因。
                    diff = (target - p_pert).abs().sum(1)
                    mask = diff > args.cons_thr
                    if bool(mask.any()):
                        loss_cons = ((target - p_pert) ** 2).sum(1)[mask].mean()
                    else:
                        loss_cons = logits.sum() * 0.0
                else:
                    loss_cons = F.mse_loss(p_pert, target)

                loss = loss + args.cons_weight * loss_cons
            else:
                logits = model(a, b)
                loss = criterion(logits, lab)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot += loss.item() * a.size(0)
            n += a.size(0)

        sched.step()
        dt = time.time() - t0

        m, thr, m50 = evaluate(model, val_loader, device)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

        print(f"ep {ep:3d}/{args.epochs}  loss {tot / max(n, 1):.4f}  "
              f"F1 {m['F1']:.4f} IoU {m['IoU']:.4f} P {m['P']:.4f} R {m['R']:.4f} @thr {thr:.2f}"
              f"  (argmax F1 {m50['F1']:.4f})  {dt:4.0f}s  peak {peak:.2f}GB")

        if m["F1"] > best:
            best = m["F1"]
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "val": m, "thr": thr,
                 "args": vars(args)},
                out_dir / "best.pt",
            )

    print(f"\nbest val F1 {best:.4f}  ->  {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
