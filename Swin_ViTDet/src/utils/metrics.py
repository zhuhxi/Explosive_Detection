from __future__ import annotations
from typing import Dict, Iterable, List, Optional

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluator:
    def __init__(self, coco_gt: COCO, iou_types: Iterable[str] = ("bbox",), label_to_coco: Optional[Dict[int, int]] = None):
        self.coco_gt = coco_gt
        self.iou_types = tuple(iou_types)
        assert "bbox" in self.iou_types, "This evaluator only supports 'bbox'."
        self.label_to_coco = label_to_coco  # {label(1..K): coco_category_id}
        self.img_ids: List[int] = []
        self.results: List[dict] = []

    def update(self, predictions: Dict[int, dict]):
        for img_id, output in predictions.items():
            boxes = output.get("boxes", torch.empty(0, 4))
            scores = output.get("scores", torch.empty(0))
            labels = output.get("labels", torch.empty(0, dtype=torch.int64))
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu()

            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b.tolist()
                w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                # 将模型的 label（1..K）映射回 COCO category_id
                if self.label_to_coco is not None and int(l) in self.label_to_coco:
                    cat_id = int(self.label_to_coco[int(l)])
                else:
                    cat_id = int(l)  # 兜底：若单类且 id=1/0 与 COCO 一致时也能工作
                self.results.append(
                    {
                        "image_id": int(img_id),
                        "category_id": cat_id,
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(s),
                    }
                )
            self.img_ids.append(int(img_id))

    def synchronize_between_processes(self):  # 单机单卡：空实现
        return

    def accumulate(self):
        # 单类/小数据时可能预测为空；给出 0 向量兜底而不是报错
        if len(self.results) == 0:
            self._stats = [0.0] * 12
            return
        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        if len(self.img_ids) > 0:
            coco_eval.params.imgIds = sorted(list(set(self.img_ids)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()  # 仍然会打印官方表格
        self._stats = list(coco_eval.stats)

    def summarize(self) -> dict:
        # 返回常用指标字典
        if not hasattr(self, "_stats"):
            self.accumulate()
        s = self._stats
        # s[0]=mAP@[.5:.95], s[1]=AP50, s[2]=AP75, s[6]=AR1, s[7]=AR10, s[8]=AR100
        return {
            "mAP": float(s[0]),
            "AP50": float(s[1]),
            "AP75": float(s[2]),
            "AR1": float(s[6]),
            "AR10": float(s[7]),
            "AR100": float(s[8]),
        }
