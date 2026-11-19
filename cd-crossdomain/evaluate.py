"""evaluate.py —— 加载 checkpoint，在 val / test 上评估

和训练时不同：这里用 checkpoint 里存下来的阈值（在 val 上选定的），
直接 apply 到 test —— 这才是论文里能报的流程。
顺便也报 argmax(0.5) 和 test 上重新搜阈值的上界，方便看清差距。

keys 来源有两种：
  * 有 patch_stats.csv 就从里面读（LEVIR）
  * 没有就直接扫 <root>/<split>/L/*.npy 的文件名（WHU、S2Looking 适配后都走这条）

用法：
  uv run evaluate.py --ckpt runs/base_pw3/best.pt
  uv run evaluate.py --ckpt runs/base_r40/best.pt --root data/patches_whu --splits test
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data import PATCH_ROOT, PatchDataset, build_loader
from train import CDNet, _metrics


def load_keys(split: str, root: Path) -> list[str]:
    stats = Path(root) / "patch_stats.csv"
    if stats.exists():
        keys = []
        with open(stats, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] == split:
                    keys.append(r["key"])
        if keys:
            return keys

    # 退回到扫目录
    d = Path(root) / split / "L"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.npy"))


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    probs_l, labs_l = [], []
    for a, b, lab, _ in loader:
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        prob = model(a, b).softmax(1)[:, 1]
        probs_l.append(prob.reshape(-1).cpu())
        labs_l.append(lab.reshape(-1).cpu())
    return torch.cat(probs_l), torch.cat(labs_l)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--root", default=str(PATCH_ROOT), help="patch 根目录")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.root)

    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cargs = ckpt.get("args", {})
        thr = float(ckpt.get("thr", 0.5))

        model = CDNet(pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])

        print(f"\n== {ckpt_path} ==")
        print(f"   epoch {ckpt.get('epoch')} | ratio {cargs.get('ratio')} "
              f"seed{cargs.get('seed')} | pos_weight {cargs.get('pos_weight')} | "
              f"阈值（val 上选定）{thr:.2f}")
        print(f"   root {root}")

        for split in args.splits:
            keys = load_keys(split, root)
            if not keys:
                print(f"   [{split}] 没找到样本，跳过")
                continue
            ds = PatchDataset(keys, split=split, root=root, sync_aug=False)
            dl = build_loader(ds, args.batch_size, False, args.workers)
            probs, labs = collect(model, dl, device)

            m_thr = _metrics(probs, labs, thr)
            m_arg = _metrics(probs, labs, 0.5)

            best = m_arg
            for t in np.arange(0.10, 0.91, 0.05):
                m = _metrics(probs, labs, float(t))
                if m["F1"] > best["F1"]:
                    best = m

            print(f"   [{split:4s}] n={len(ds):5d}  "
                  f"@{thr:.2f}  F1 {m_thr['F1']:.4f} IoU {m_thr['IoU']:.4f} "
                  f"P {m_thr['P']:.4f} R {m_thr['R']:.4f}")
            print(f"            argmax0.5 F1 {m_arg['F1']:.4f}   "
                  f"best-thr 上界 F1 {best['F1']:.4f} @ {best['thr']:.2f}")

            del probs, labs


if __name__ == "__main__":
    main()
