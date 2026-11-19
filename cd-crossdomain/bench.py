"""GTX 1650 4GB 显存/速度基准 —— 孪生 ResNet-18 + 轻量解码器

用法：
  uv run bench.py               # 全部配置（进程内连续，显存读数会被污染，仅看速度）
  uv run bench.py 256 4         # 单配置，独立进程，显存读数可信

显存读数必须在独立进程里取：CUDA context 和 cuDNN workspace 一旦建立，
后续配置的 peak_memory 读数会失真。
"""
import gc
import sys
import time

import torch
import torch.nn as nn
import torchvision

torch.backends.cudnn.benchmark = True
DEV = "cuda"

# LEVIR-CD 官方划分：train 445 对，1024->256 切 4x4 = 16 块 => 7120 patch/epoch
LEVIR_PATCHES = 7120


class CDNet(nn.Module):
    """Siamese ResNet-18（权重共享）+ 五级上采样解码，输出全分辨率变化图。"""

    def __init__(self, arch="resnet18"):
        super().__init__()
        r = getattr(torchvision.models, arch)(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.head = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 2, 1),
        )

    def enc(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.l4(x)

    def forward(self, a, b):
        return self.head(torch.abs(self.enc(a) - self.enc(b)))


def bench(bs, res, iters=10):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    net = CDNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    a = torch.randn(bs, 3, res, res, device=DEV)
    b = torch.randn(bs, 3, res, res, device=DEV)
    y = torch.randint(0, 2, (bs, res, res), device=DEV)

    for _ in range(3):
        out = net(a, b)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()

    t = time.time()
    for _ in range(iters):
        out = net(a, b)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9

    del net, opt, a, b, y, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    return dt, peak_alloc, peak_reserved


if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)

    if len(sys.argv) == 3:
        res, bs = int(sys.argv[1]), int(sys.argv[2])
        dt, pa, pr = bench(bs, res)
        it_per_epoch = LEVIR_PATCHES / bs
        print(
            f"RESULT res={res} bs={bs} s_iter={dt:.4f} alloc={pa:.2f} resv={pr:.2f} "
            f"iter_epoch={it_per_epoch:.0f} min_epoch={dt * it_per_epoch / 60:.1f}"
        )
        sys.exit(0)

    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | avail {torch.cuda.is_available()}")
    print(f"dev {p.name} | total VRAM {p.total_memory / 1e9:.2f} GB | cc {p.major}.{p.minor}")
    print(f"assume {LEVIR_PATCHES} patch/epoch (LEVIR-CD 1024->256)")
    print()
    print(f"{'res':>5} {'bs':>3} {'s/iter':>8} {'allocGB':>8} {'resvGB':>7} {'it/epoch':>9} {'min/epoch':>10}")

    for res, bs in [(256, 1), (256, 2), (256, 4), (192, 4), (128, 8)]:
        try:
            dt, pa, pr = bench(bs, res)
            it_per_epoch = LEVIR_PATCHES / bs
            print(f"{res:>5} {bs:>3} {dt:>8.3f} {pa:>8.2f} {pr:>7.2f} {it_per_epoch:>9.0f} {dt * it_per_epoch / 60:>10.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"{res:>5} {bs:>3}   FAILED: {type(e).__name__}: {str(e)[:70]}")
            gc.collect()
            torch.cuda.empty_cache()
