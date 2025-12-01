
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN, AnchorGenerator
from .backbones import TimmBackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN, AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

def build_swin_fasterrcnn(
    num_classes: int = 1,
    backbone_name: str = "swin_tiny_patch4_window7_224",
    pretrained_backbone: bool = True,
    fixed_size: int | None = 896,
):
    # Swin + FPN，输出4层：{'0','1','2','3'}
    backbone = TimmBackboneWithFPN(
        backbone_name, out_channels=256, pretrained=pretrained_backbone, img_size=fixed_size
    )

    # 更利于小目标的 anchors；4层金字塔 => 4组 sizes
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    # 关键：显式指定 ROIAlign 使用四个尺度
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],  # 对应 backbone.forward 返回的字典键
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes + 1,     # +1 背景
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,         # <<< 关键：多尺度 ROI 对齐
        box_detections_per_img=100,
        min_size=(fixed_size if fixed_size is not None else 800),
        max_size=(fixed_size if fixed_size is not None else 1333),
    )

    # 让 RPN 起量更快（可选）
    model.rpn.pre_nms_top_n_train  = 2000
    model.rpn.pre_nms_top_n_test   = 1000
    model.rpn.post_nms_top_n_train = 1000
    model.rpn.post_nms_top_n_test  = 500

    return model
