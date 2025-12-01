#!/usr/bin/env python3
"""
Convert a YOLO-format dataset (Ultralytics style) to COCO format.

Expected YOLO layout under --root:
  data.yaml  # contains names mapping
  images/
    train/ *.jpg|*.png
    val/   *.jpg|*.png
    test/  *.jpg|*.png   (optional)
  labels/
    train/ *.txt
    val/   *.txt
    test/  *.txt         (optional)

Outputs:
  annotations/instances_{split}.json under --root

Usage:
  python tools/yolo_to_coco.py \
    --root data/explosive_dataset_coco_1 \
    --splits train val test

Notes:
  - Class ids in YOLO .txt start from 0; in COCO we map to category_id starting from 1.
  - Images without labels are included with empty annotations.
  - Bounding boxes are converted from normalized (xc,yc,w,h) to COCO (x,y,w,h) in pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    from PIL import Image
except Exception as e:
    print("[ERROR] Pillow is required: pip install pillow", file=sys.stderr)
    raise


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_class_names(data_yaml: Path) -> List[str]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    if yaml is None:
        # simple fallback parser for minimal case
        names: List[str] = []
        text = data_yaml.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("names"):
                # next lines may contain mapping
                pass
        # Fallback is too brittle; require PyYAML
        raise RuntimeError("PyYAML not available. Please `pip install pyyaml`. "
                           "Or edit script to hardcode class names.")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    # names can be list or dict like {0: 'class0', 1: 'class1'}
    if isinstance(names, dict):
        # sort by numeric key
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if not isinstance(names, list):
        raise ValueError("Invalid names in data.yaml; expected list or mapping")
    return names


def yolo_box_to_coco(
    xywhn: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    xc, yc, w, h = xywhn
    x = (xc - w / 2.0) * img_w
    y = (yc - h / 2.0) * img_h
    bw = w * img_w
    bh = h * img_h
    # clamp
    x = max(0.0, min(x, img_w - 1.0))
    y = max(0.0, min(y, img_h - 1.0))
    bw = max(0.0, min(bw, img_w - x))
    bh = max(0.0, min(bh, img_h - y))
    return x, y, bw, bh


def gather_images(split_img_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    if not split_img_dir.exists():
        return imgs
    for p in sorted(split_img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs


def convert_split(root: Path, split: str, names: List[str]) -> Dict:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images = gather_images(images_dir)

    categories = [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in enumerate(names)
    ]

    coco: Dict = {
        "info": {
            "description": f"YOLO->COCO converted dataset ({split})",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    ann_id = 1
    for img_id, img_path in enumerate(images, start=1):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            print(f"[WARN] Failed to open image: {img_path}")
            continue

        file_name = img_path.name  # used with data_prefix img='images/{split}/'
        coco["images"].append(
            {
                "id": img_id,
                "file_name": file_name,
                "width": w,
                "height": h,
            }
        )

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            # include image with no annotations
            continue

        try:
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            print(f"[WARN] Failed to read label: {label_path}")
            lines = []

        for line in lines:
            if not line:
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                # Sometimes format includes confidences etc.; ignore malformed
                try:
                    cls = int(float(parts[0]))
                    nums = list(map(float, parts[1:5]))
                except Exception:
                    continue
            else:
                cls = int(float(parts[0]))
                nums = list(map(float, parts[1:5]))

            if cls < 0 or cls >= len(names):
                continue

            x, y, bw, bh = yolo_box_to_coco((nums[0], nums[1], nums[2], nums[3]), w, h)
            if bw <= 0 or bh <= 0:
                continue

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,  # COCO ids start at 1
                    "bbox": [x, y, bw, bh],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    return coco


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to COCO format")
    parser.add_argument("--root", type=str, required=True, help="Path to YOLO dataset root")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["train", "val", "test"],
        help="Dataset splits to convert",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output annotations dir (default: <root>/annotations)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    data_yaml = root / "data.yaml"
    names = load_class_names(data_yaml)

    out_dir = Path(args.out) if args.out else (root / "annotations")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        images_dir = root / "images" / split
        if not images_dir.exists():
            print(f"[INFO] Skip split '{split}': {images_dir} not found")
            continue
        coco = convert_split(root, split, names)
        out_file = out_dir / f"instances_{split}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        print(f"[OK] Wrote {out_file}  images={n_img}  annotations={n_ann}")


if __name__ == "__main__":
    main()
