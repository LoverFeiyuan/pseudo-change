"""smoke_test.py —— 数据管线冒烟测试

每次改 Dataset / 采样器之后跑一遍，确认四件事：
  1. 不同 seed 抽出来的标注集确实不同（不是同一批 patch）
  2. 标注集的正负比例符合预期
  3. 多 worker（Windows spawn）能正常出 batch
  4. 单 epoch 吞吐的大致量级

跑法： uv run smoke_test.py
"""
import json
import time

import torch

from data import PATCH_ROOT, PatchDataset, build_loader, bucket_of


def check_seed_independence(splits: dict) -> None:
    print("== seed independence (5%) ==")
    cfg = splits["splits"]["0.05"]
    names = list(cfg)
    sets = {n: set(cfg[n]["labeled"]) for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sets[names[i]], sets[names[j]]
            inter = len(a & b)
            print(f"  {names[i]} vs {names[j]}: overlap {inter}/{len(a)} ({100 * inter / len(a):.0f}%)")


def check_composition(splits: dict, ratio: str = "0.10") -> None:
    print(f"\n== composition ({ratio}) ==")
    keys = splits["splits"][ratio]["seed0"]["labeled"]
    ratio_of = {r["key"]: r["ratio"] for r in _load_stats()}
    buckets: dict[str, int] = {}
    pos = neg = 0
    for k in keys:
        r = ratio_of[k]
        if r > 0:
            pos += 1
            b = bucket_of(r)
            buckets[b] = buckets.get(b, 0) + 1
        else:
            neg += 1
    print(f"  labeled={len(keys)}  pos={pos}  neg={neg}")
    print("  pos by bucket:", {k: v for k, v in sorted(buckets.items())})


def check_loader(splits: dict, ratio: str = "0.10", batch_size: int = 8,
                 num_workers: int = 4, iters: int = 20, warmup: int = 5) -> None:
    print(f"\n== loader (batch={batch_size}, workers={num_workers}) ==")
    keys = splits["splits"][ratio]["seed0"]["labeled"]
    ds = PatchDataset(keys, split="train", sync_aug=True)
    dl = build_loader(ds, batch_size=batch_size, num_workers=num_workers)

    it = iter(dl)
    for _ in range(warmup):  # Windows spawn: 每个 worker 启动都要 import torch
        next(it)

    t0 = time.time()
    for i in range(iters):
        a, b, lab, k = next(it)
        if i == 0:
            print(f"  first batch: a{tuple(a.shape)} b{tuple(b.shape)} lab{tuple(lab.shape)} "
                  f"{a.dtype}/{lab.dtype}  workers={dl.num_workers}")
            print(f"  a range [{a.min():.2f}, {a.max():.2f}]  "
                  f"pos ratio in label {lab.float().mean():.4f}")
    dt = time.time() - t0

    per_batch = dt / iters
    n_batches = len(ds) / batch_size
    print(f"  warm {iters} batches in {dt:.2f}s -> {per_batch * 1000:.0f} ms/batch")
    print(f"  extrapolated epoch ({len(ds)} samples): {per_batch * n_batches / 60:.1f} min "
          f"(data only, no GPU)")


def _load_stats() -> list[dict]:
    import csv
    rows = []
    with open(PATCH_ROOT / "patch_stats.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "train":
                continue
            rows.append({"key": r["key"], "ratio": float(r["change_ratio"])})
    return rows


def main() -> None:
    splits = json.loads((PATCH_ROOT / "splits.json").read_text(encoding="utf-8"))
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    check_seed_independence(splits)
    check_composition(splits, "0.10")
    check_loader(splits, "0.10", batch_size=8, num_workers=4)


if __name__ == "__main__":
    main()
