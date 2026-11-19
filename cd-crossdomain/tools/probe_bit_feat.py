"""probe_bit_feat.py — 确认 BIT 中间特征的形状与位置

特征分析需要一张 2D 特征图，按像素采样。BIT 的 classifier 结构是
conv_pred(256->32) -> BN -> ReLU -> conv(32->2)，取 conv_pred 输出即可。
"""
import sys
import types

import torch
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
    net.load_state_dict(ck["model_G_state_dict"])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev).eval()

    print("classifier 结构：", net.classifier)

    box = {}

    def hook(_m, _i, o):
        box["f"] = o

    net.classifier[0].register_forward_hook(hook)

    a = torch.randn(2, 3, 256, 256, device=dev)
    b = torch.randn(2, 3, 256, 256, device=dev)
    with torch.no_grad():
        out = net(a, b)

    f = box["f"]
    print(f"conv_pred 输出 {tuple(f.shape)}")
    print(f"网络输出 {tuple(out.shape)}")


if __name__ == "__main__":
    main()
