"""make_splits.py —— 分层采样，产出 splits.json

要点：
  * 标注集按「正负各半」抽，正样本内部再按变化比例分桶分层，避免全被大变化占满
  * 固定 seed 并落盘，保证每次实验用的是同一批 patch（不然实验之间没法比）
  * unlabeled 是「剩下的全部」，半监督训练时整份参与一致性正则

跑法：
  uv run make_splits.py            # 生成 splits.json
  uv run make_splits.py --check    # 生成后顺手验一下 Dataset
"""
import argparse
import csv
import json
import math

import numpy as np

from data import POS_BUCKETS, PATCH_ROOT, bucket_of

RATIOS = [0.05, 0.10, 0.20, 0.40]
SEEDS = [0, 1, 2]
OUT = PATCH_ROOT / "splits.json"


def load_train_stats() -> list[dict]:
    rows = []
    with open(PATCH_ROOT / "patch_stats.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "train":
                continue
            rows.append({"key": r["key"], "ratio": float(r["change_ratio"])})
    rows.sort(key=lambda x: x["key"])
    return rows


def allocate(cap: dict[str, int], total: int) -> dict[str, int]:
    """按各桶容量的比例把 total 个名额分下去，且不超过各自容量。"""
    total = min(int(total), sum(cap.values()))
    if total <= 0:
        return {k: 0 for k in cap}

    grand = sum(cap.values())
    raw = {k: cap[k] * total / grand for k in cap}
    out = {k: min(int(math.floor(raw[k])), cap[k]) for k in cap}

    left = total - sum(out.values())
    guard = 0
    while left > 0 and guard < 10000:
        guard += 1
        cand = [k for k in cap if out[k] < cap[k]]
        if not cand:
            break
        cand.sort(key=lambda k: raw[k] - out[k], reverse=True)
        for k in cand:
            if left <= 0:
                break
            out[k] += 1
            left -= 1
    return out


def make_one(records: list[dict], ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    n_lab = int(round(len(records) * ratio))
    n_pos_target = n_lab // 2

    pos = [r for r in records if r["ratio"] > 0]
    neg = [r for r in records if r["ratio"] == 0]

    pools: dict[str, list[str]] = {b: [] for b in POS_BUCKETS}
    for r in pos:
        pools[bucket_of(r["ratio"])].append(r["key"])

    quotas = allocate({b: len(pools[b]) for b in POS_BUCKETS}, n_pos_target)

    labeled: list[str] = []
    for b in POS_BUCKETS:
        q = quotas[b]
        if q <= 0:
            continue
        idx = rng.choice(len(pools[b]), size=q, replace=False)
        labeled += [pools[b][i] for i in idx]

    n_neg = n_lab - len(labeled)
    if n_neg > 0:
        take = min(n_neg, len(neg))
        idx = rng.choice(len(neg), size=take, replace=False)
        labeled += [neg[i]["key"] for i in idx]

    lab_set = set(labeled)
    unlabeled = [r["key"] for r in records if r["key"] not in lab_set]
    return sorted(labeled), sorted(unlabeled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="生成后构造一个 Dataset 样本验证")
    args = ap.parse_args()

    records = load_train_stats()
    ratio_of = {r["key"]: r["ratio"] for r in records}
    n_pos = sum(1 for r in records if r["ratio"] > 0)

    print(f"train patches: {len(records)}")
    print(f"  with change : {n_pos}")
    print(f"  all-zero    : {len(records) - n_pos}")
    print()

    out = {
        "meta": {
            "n_train": len(records),
            "n_pos": n_pos,
            "ratios": RATIOS,
            "seeds": SEEDS,
            "sampling": "half positive (bucket-stratified) + half negative",
        },
        "splits": {},
    }

    for ratio in RATIOS:
        tag = f"{ratio:.2f}"
        out["splits"][tag] = {}
        for seed in SEEDS:
            lab, unl = make_one(records, ratio, seed)
            out["splits"][tag][f"seed{seed}"] = {"labeled": lab, "unlabeled": unl}

            lab_pos = sum(1 for k in lab if ratio_of[k] > 0)
            by_bucket: dict[str, int] = {}
            for k in lab:
                if ratio_of[k] > 0:
                    b = bucket_of(ratio_of[k])
                    by_bucket[b] = by_bucket.get(b, 0) + 1
            detail = " ".join(f"{b}:{by_bucket.get(b, 0)}" for b in POS_BUCKETS)
            print(f"[{tag}] seed{seed}  labeled={len(lab):5d} (pos {lab_pos:4d} / neg {len(lab) - lab_pos:4d})"
                  f"  unlabeled={len(unl):5d}   pos-buckets[{detail}]")

    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten -> {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")

    if args.check:
        from data import PatchDataset

        keys = out["splits"]["0.05"]["seed0"]["labeled"]
        ds = PatchDataset(keys, split="train", sync_aug=True, normalize=True)
        a, b, lab, k = ds[0]
        print("\ncheck (5% / seed0, first sample):")
        print(f"  key={k}")
        print(f"  a {tuple(a.shape)} {a.dtype}  range [{a.min():.2f}, {a.max():.2f}]")
        print(f"  b {tuple(b.shape)} {b.dtype}")
        print(f"  label {tuple(lab.shape)} {lab.dtype}  pos_pix={int(lab.sum())}")


if __name__ == "__main__":
    main()
