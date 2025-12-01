from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

try:
    import timm
except Exception:
    timm = None


# ------------------------------
# Swin + FPN（timm features_only）
# ------------------------------
class TimmBackboneWithFPN(nn.Module):
    def __init__(
        self,
        model_name: str,
        out_channels: int = 256,
        pretrained: bool = True,
        trainable_layers: int = 4,
        norm_layer: Optional[nn.Module] = nn.BatchNorm2d,
        exportable: bool = False,
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        assert timm is not None, "timm is required. Please install timm."

        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            exportable=exportable,
            img_size=(img_size, img_size) if img_size else None,
        )
        feature_info = self.body.feature_info
        in_channels_list = [f["num_chs"] for f in feature_info]
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels, norm_layer=norm_layer)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        xs: List[torch.Tensor] = self.body(x)  # list of feature maps

        # 一些 timm 版本返回 NHWC，需要转 NCHW
        fixed = []
        fi = getattr(self.body, "feature_info", None)
        for i, t in enumerate(xs):
            if t.dim() == 4:
                expected_c = None
                if fi is not None:
                    try:
                        expected_c = fi[i]["num_chs"]
                    except Exception:
                        expected_c = None
                if (expected_c is not None and t.shape[-1] == expected_c and t.shape[1] != expected_c) or (
                    expected_c is None and t.shape[-1] >= t.shape[1] and t.shape[-1] >= t.shape[2]
                ):
                    t = t.permute(0, 3, 1, 2).contiguous()
            fixed.append(t)

        xdict = OrderedDict({str(i): t for i, t in enumerate(fixed)})  # '0','1','2','3'
        xdict = self.fpn(xdict)
        return xdict


# --------------------------------------------
# ViTDet-style：ViT 主干 + Simple Feature Pyramid
# --------------------------------------------
class ViTBackboneWithSFP(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224", out_channels: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        assert timm is not None, "timm is required. Please install timm."
        self.vit = timm.create_model(model_name, pretrained=pretrained)

        # patch size
        self.patch_size = getattr(self.vit.patch_embed, "patch_size", 16)
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]

        self.embed_dim = self.vit.num_features
        self.out_channels = out_channels

        # C5: 把 ViT 的通道映射到 256
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)

        # 简单金字塔（自顶向下）
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.GELU()

    def _tokens_to_map(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, 1+N, C) 或 (B, N, C) -> (B, C, H, W)
        if x.dim() == 3:
            if x.shape[1] == (H * W + 1):  # 去 cls
                x = x[:, 1:, :]
            x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], H, W)
        return x

    def _interpolate_pos_embed(self, pos_embed, Hp, Wp):
        # 把 patch 的位置编码插值到 (Hp, Wp)
        num_extra_tokens = 1  # cls token
        cls_pos = pos_embed[:, :num_extra_tokens, :]
        pos_tokens = pos_embed[:, num_extra_tokens:, :]
        orig_N = pos_tokens.shape[1]
        orig_size = int(orig_N ** 0.5)
        pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(Hp, Wp), mode="bicubic", align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, Hp * Wp, -1)
        return torch.cat((cls_pos, pos_tokens), dim=1)

    def forward(self, x: torch.Tensor):
        # x: (B, 3, H, W)
        B, _, H, W = x.shape

        # 允许任意分辨率
        pe = self.vit.patch_embed
        if hasattr(pe, "img_size"):
            pe.img_size = (H, W)
        if hasattr(pe, "strict_img_size"):
            pe.strict_img_size = False

        # 1) 打补丁嵌入
        x = self.vit.patch_embed(x)  # (B, N, C) 或 (B, C, Hp, Wp)

        # 2) tokens & 网格
        if x.dim() == 4:
            Hp, Wp = x.shape[-2:]
            x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        else:
            N = x.shape[1]
            Hp = max(1, H // self.patch_size)
            Wp = max(1, N // Hp) if N % Hp == 0 else int(N ** 0.5)
            if Hp * Wp != N:
                Hp = int(N ** 0.5)
                Wp = N // Hp

        # 3) 拼接 cls token
        if getattr(self.vit, "cls_token", None) is not None:
            cls_tokens = self.vit.cls_token.expand(B, -1, -1)  # (B,1,C)
            x = torch.cat((cls_tokens, x), dim=1)

        # 4) 位置编码
        pos = self._interpolate_pos_embed(self.vit.pos_embed, Hp, Wp)
        x = x + pos[:, : x.shape[1], :]
        x = self.vit.pos_drop(x)

        # 5) Transformer blocks
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)

        # 6) tokens -> (B, C, Hp, Wp)
        x = self._tokens_to_map(x, Hp, Wp)

        # 7) 简单金字塔 P3~P6（与 ROIAlign 的 featmap_names 对齐）
        c5 = self.proj(x)                                             # stride 16
        p5 = self.smooth5(self.act(c5))
        p4 = self.smooth4(self.act(F.interpolate(p5, scale_factor=2.0, mode="bilinear", align_corners=False)))  # s8
        p3 = self.smooth3(self.act(F.interpolate(p4, scale_factor=2.0, mode="bilinear", align_corners=False)))  # s4
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2)  # s32

        return {"0": p3, "1": p4, "2": p5, "3": p6}  # keys 必须是 '0'..'3'
