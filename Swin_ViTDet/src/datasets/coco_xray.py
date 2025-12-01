import os
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoXrayDataset(Dataset):
    def __init__(self, root: str, ann_file: str, image_dir: str, transforms=None):
        """
        root: 数据集根目录
        ann_file: COCO json 的绝对路径
        image_dir: 图像目录的绝对路径
        """
        self.root = root
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transforms = transforms

        assert os.path.exists(self.ann_file), f"Annotation file not found: {self.ann_file}"
        assert os.path.exists(self.image_dir), f"Image dir not found: {self.image_dir}"

        self.coco = COCO(self.ann_file)
        self.ids = sorted(self.coco.getImgIds())

        # --- 构建稳定的类别映射 ---
        cats = self.coco.loadCats(self.coco.getCatIds())
        # 按 COCO 原始 id 排序，避免不同运行时顺序抖动
        self.coco_cat_ids = sorted(c["id"] for c in cats)
        # 训练用 label: 1..num_classes
        self.coco_to_label = {coco_id: i + 1 for i, coco_id in enumerate(self.coco_cat_ids)}
        self.label_to_coco = {v: k for k, v in self.coco_to_label.items()}
        self.num_classes = len(self.coco_cat_ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.image_dir, info["file_name"])
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            if "bbox" not in ann:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            # 将 COCO category_id 映射为训练 label
            labels.append(self.coco_to_label.get(int(ann["category_id"]), 1))
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.as_tensor([img_id], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
            # 原图尺寸（H, W）——供评估时反 letterbox
            "orig_size": torch.as_tensor([img.height, img.width], dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, dict]]):
        return tuple(zip(*batch))
