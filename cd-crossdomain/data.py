"""data.py —— PatchDataset 与分桶工具

约定（和 build_patches*.py 对齐，别在别处改）：
  * 磁盘上的 patch 是 BGR uint8
  * Dataset 输出 RGB float32，按 ImageNet 均值方差归一化
  * label 是 (H,W) int64，直接喂 CrossEntropyLoss；要用 BCE 自己 unsqueeze(1)

两种增广，语义完全不同，别混：
  * sync_aug  几何（翻转/旋转 90°）—— A / B / label 必须同步，否则像素对应关系被破坏
  * photo_aug 成像统计（增益/偏置/gamma/噪声）—— A / B **必须独立**采样

photo_aug 为什么必须独立：如果两张图用同一个增益，模型可以靠「亮度一致」去
判断「没变化」，这个捷径在跨域时会立刻崩掉（S2Looking 的亮度比 LEVIR 低 63%）。
独立采样等于逼模型只看结构差异，不看成像差异。
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA = Path(r"data")
PATCH_ROOT = DATA / "patches"
PATCH = 256

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# (名字, 下界开区间, 上界闭区间)
BUCKETS = [
    ("neg", 0.0, 0.0),
    ("tiny", 0.0, 0.001),
    ("small", 0.001, 0.01),
    ("mid", 0.01, 0.05),
    ("large", 0.05, 0.20),
    ("huge", 0.20, 1.01),
]
BUCKET_NAMES = [b[0] for b in BUCKETS]
POS_BUCKETS = [b for b in BUCKET_NAMES if b != "neg"]


def bucket_of(ratio: float) -> str:
    """变化像素比例 -> 桶名。边界一律取上界闭、下界开。"""
    if ratio <= 0.0:
        return "neg"
    if ratio <= 0.001:
        return "tiny"
    if ratio <= 0.01:
        return "small"
    if ratio <= 0.05:
        return "mid"
    if ratio <= 0.20:
        return "large"
    return "huge"


class PatchDataset(Dataset):
    """一个样本 = (A, B, label)。

    keys 是 patch 的 key（如 "train_1_23"），由 make_splits.py 或目录扫描得到。
    """

    def __init__(self, keys, split="train", root=PATCH_ROOT,
                 normalize=True, sync_aug=False, photo_aug=False, raw=False):
        self.root = Path(root) / split
        self.keys = list(keys)
        self.normalize = normalize
        self.sync_aug = sync_aug
        self.photo_aug = photo_aug
        self.raw = raw

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        key = self.keys[i]
        a = np.load(self.root / "A" / f"{key}.npy")
        b = np.load(self.root / "B" / f"{key}.npy")
        lab = np.load(self.root / "L" / f"{key}.npy")

        if self.sync_aug:
            a, b, lab = self._geo(a, b, lab)
        if self.photo_aug:
            a = self._photo(a)
            b = self._photo(b)

        if self.raw:
            # 训练侧要做「原图 / 扰动图」双前向的一致性约束，
            # 所以这里交还未归一化的 BGR uint8，归一化搬到训练循环里
            return (
                torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(lab).astype(np.int64)),
                key,
            )

        return (
            self._to_tensor(a),
            self._to_tensor(b),
            torch.from_numpy(np.ascontiguousarray(lab).astype(np.int64)),
            key,
        )

    def _to_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        x = np.ascontiguousarray(img_bgr[:, :, ::-1]).astype(np.float32) / 255.0
        if self.normalize:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    def _geo(self, a, b, lab):
        """同步几何增广。用 torch 的随机数，DataLoader 会给每个 worker 不同的种子。"""
        if bool(torch.rand(()) < 0.5):
            a, b, lab = a[:, ::-1], b[:, ::-1], lab[:, ::-1]
        if bool(torch.rand(()) < 0.5):
            a, b, lab = a[::-1], b[::-1], lab[::-1]
        k = int(torch.randint(0, 4, ()).item())
        if k:
            a, b, lab = np.rot90(a, k), np.rot90(b, k), np.rot90(lab, k)
        return a, b, lab

    def _photo(self, img_bgr: np.ndarray) -> np.ndarray:
        """成像统计扰动，模拟不同传感器 / 光照 / 大气条件。

        增益范围取 0.55–1.6，覆盖到 S2Looking 那种 −63% 的亮度差；
        每通道独立再乘一次，模拟传感器响应曲线的差异。
        """
        x = img_bgr.astype(np.float32)

        # per-channel 0.75–1.30 是扫出来的最优区间：再宽就伤判别力
        # （cons_r10c 试过 0.55–1.85，WHU 从 0.73 掉到 0.55）
        gain = torch.empty(3).uniform_(0.75, 1.30).numpy()
        gain = gain * float(torch.empty(1).uniform_(0.55, 1.60))
        bias = float(torch.empty(1).uniform_(-30.0, 30.0))

        x = x * gain.astype(np.float32) + bias

        if bool(torch.rand(()) < 0.5):
            gamma = float(torch.empty(1).uniform_(0.75, 1.35))
            x = 255.0 * np.power(np.clip(x, 0.0, 255.0) / 255.0, gamma)

        if bool(torch.rand(()) < 0.30):
            sigma = float(torch.empty(1).uniform_(2.0, 8.0))
            x = x + np.random.normal(0.0, sigma, x.shape).astype(np.float32)

        return np.clip(x, 0.0, 255.0).astype(np.uint8)


def build_loader(ds: Dataset, batch_size: int = 8, shuffle: bool = True,
                 num_workers: int = 4) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
