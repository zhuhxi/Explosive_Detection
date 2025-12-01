import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN, AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from .backbones import ViTBackboneWithSFP


def build_vitdet_fasterrcnn(
    num_classes: int = 1,
    vit_name: str = "vit_base_patch16_224",
    pretrained_backbone: bool = True,
    fixed_size: int | None = 896,
):
    backbone = ViTBackboneWithSFP(vit_name, out_channels=256, pretrained=pretrained_backbone)

    # 先用标准 32/64/128/256（更稳）；如确认多为小目标，再改成 16/32/64/128
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes + 1,  # +1 背景
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,      # 多尺度 ROI 对齐
        box_detections_per_img=100,
        min_size=(fixed_size if fixed_size is not None else 800),
        max_size=(fixed_size if fixed_size is not None else 1333),
    )

    # 提高 RPN 提案数，早期更易有框
    model.rpn.pre_nms_top_n_train  = 2000
    model.rpn.pre_nms_top_n_test   = 1000
    model.rpn.post_nms_top_n_train = 1000
    model.rpn.post_nms_top_n_test  = 500

    # 评估时阈值会在 main 中临时设置；这里不收紧
    if hasattr(model.roi_heads, "score_thresh"):
        model.roi_heads.score_thresh = 0.0

    return model
