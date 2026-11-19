"""aggregate.py —— 多 seed 汇总（mean ± std）

单次实验的方差可以轻松吃掉一个模块的全部收益，所以从今往后所有对比都必须
走这个脚本：先按 3 个 seed 训练，再用它汇总。

用法：
  # 组名=glob，可给多组
  uv run aggregate.py --group "基线=runs/base_seed*" --group "模块B=runs/seed*" --domains WHU-test
  uv run aggregate.py --group "模块A=runs/cons_seed*" --domains LEVIR-test,WHU-test

说明：
  * 指标固定阈值 0.5，模型各自取训练时保存的 best.pt
  * 每组内按 glob 匹配到的 ckpt 数量决定 n
  * 输出同时给逐 seed 明细，便于看方差
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from bench_protocol import DOMAINS, probs_patch, probs_s2l
from train import CDNet, _metrics


def eval_ckpt(ckpt: str, domain: str, device: str, limit: int) -> dict:
    kind, root, split = DOMAINS[domain]
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = CDNet(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    if kind == "patch":
        probs, labs = probs_patch(model, root, split, device)
    else:
        probs, labs = probs_s2l(model, root, split, device, limit)

    if probs is None:
        raise RuntimeError(f"no samples for {domain}")

    m = _metrics(probs, labs, 0.5)
    del model, probs, labs
    torch.cuda.empty_cache()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True,
                    help='形如 "组名=runs/xxx_seed*"，可重复')
    ap.add_argument("--domains", default="LEVIR-test,WHU-test")
    ap.add_argument("--limit", type=int, default=500, help="S2L 图数上限")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    groups = []
    for spec in args.group:
        name, pattern = spec.split("=", 1)
        ckpts = sorted(glob.glob(pattern))
        # 允许 pattern 匹配到目录（自动补 best.pt）
        ckpts = [c if c.endswith(".pt") else str(Path(c) / "best.pt") for c in ckpts]
        ckpts = [c for c in ckpts if Path(c).exists()]
        if not ckpts:
            print(f"! 组 {name} 没有匹配到 ckpt: {pattern}")
            continue
        groups.append((name, ckpts))

    if not groups:
        raise SystemExit("没有可评估的组")

    # 逐组逐域评估，结果存起来
    results: dict[str, dict[str, list[float]]] = {}
    for name, ckpts in groups:
        print(f"== {name} ({len(ckpts)} seeds) ==")
        per_domain: dict[str, list[float]] = {d: [] for d in domains}
        for ck in ckpts:
            row = []
            for dom in domains:
                m = eval_ckpt(ck, dom, device, args.limit)
                per_domain[dom].append(m["F1"])
                row.append(f"{dom}={m['F1']:.4f}")
            print(f"  {Path(ck).parent.name:<24} " + "  ".join(row))
        results[name] = per_domain

    print()
    header = f"{'group':<20}" + "".join(f"{d:>22}" for d in domains)
    print(header)
    print("-" * len(header))
    for name, per_domain in results.items():
        cells = []
        for dom in domains:
            arr = np.array(per_domain[dom])
            cells.append(f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}")
        print(f"{name:<20}" + "".join(f"{c:>22}" for c in cells))

    # 组间差值（第一组为参照）
    if len(results) >= 2:
        base_name = groups[0][0]
        print()
        print(f"相对 {base_name} 的差值：")
        for name, per_domain in results.items():
            if name == base_name:
                continue
            cells = []
            for dom in domains:
                a = np.array(results[base_name][dom])
                b = np.array(per_domain[dom])
                cells.append(f"{dom} {b.mean() - a.mean():+.4f}")
            print(f"  {name:<20} " + "  ".join(cells))


if __name__ == "__main__":
    main()
