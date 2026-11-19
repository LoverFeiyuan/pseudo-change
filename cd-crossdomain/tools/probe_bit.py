"""probe_bit.py — 验证 BIT 权重可加载，并确认输出格式

BIT 官方代码用 [-1,1] 归一化（不是 ImageNet），输入 RGB。
输出通道为 2，需确认是 logits 还是已过 softmax/sigmoid。
"""
import sys
import types

import torch

# BIT 官方代码写于 torchvision 0.10 之前，用了已移除的 torchvision.models.utils
import torch.hub
import torchvision.models as _tvm

if not hasattr(_tvm, "utils"):
    _m = types.ModuleType("torchvision.models.utils")
    _m.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    sys.modules["torchvision.models.utils"] = _m

sys.path.insert(0, r"third_party/BIT_CD")
from models.networks import BASE_Transformer  # noqa: E402

CKPT = r"data/LEVIR-CD\BIT_LEVIR\best_ckpt.pt"


def main() -> None:
    net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4,
                           resnet_stages_num=4, with_pos="learned",
                           enc_depth=1, dec_depth=8, decoder_dim_head=8)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_G_state_dict"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"缺失 {len(missing)} 项，多余 {len(unexpected)} 项")
    if missing[:5]:
        print("  missing 前几项:", missing[:5])
    if unexpected[:5]:
        print("  unexpected 前几项:", unexpected[:5])
    print(f"best_val_acc {ck.get('best_val_acc'):.4f}  epoch {ck.get('best_epoch_id')}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev).eval()

    a = torch.randn(2, 3, 256, 256, device=dev)
    b = torch.randn(2, 3, 256, 256, device=dev)
    with torch.no_grad():
        out = net(a, b)
    print(f"输出 shape {tuple(out.shape)}  dtype {out.dtype}")
    print(f"值域 [{out.min().item():.4f}, {out.max().item():.4f}]")
    s = out.sum(1)
    print(f"通道和 [{s.min().item():.4f}, {s.max().item():.4f}]  （≈1 表示已 softmax）")


if __name__ == "__main__":
    main()
