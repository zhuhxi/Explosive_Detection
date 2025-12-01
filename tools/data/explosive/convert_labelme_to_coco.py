#!/usr/bin/env python3
"""Convert LabelMe rectangle annotations into COCO json."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class Shape:
    label: str
    points: Sequence[Sequence[float]]


@dataclass
class ImageInfo:
    id: int
    file_name: str
    width: int
    height: int


@dataclass
class AnnotationInfo:
    id: int
    image_id: int
    category_id: int
    bbox: List[float]
    area: float
    segmentation: List[List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("data/fewshot_dataset/Explosive/Annotations"),
        help="Directory containing LabelMe json files.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/fewshot_dataset/Explosive/Images"),
        help="Directory with the corresponding image files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/coco/annotations/explosive_all.json"),
        help="Output COCO json path.",
    )
    parser.add_argument(
        "--category-file",
        type=Path,
        default=None,
        help="Optional text file containing one category label per line to fix ordering.",
    )
    return parser.parse_args()


def load_labelme(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def gather_categories(annotation_paths: Iterable[Path], category_file: Path | None) -> List[str]:
    if category_file:
        labels = [line.strip() for line in category_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not labels:
            raise ValueError(f"Category file {category_file} is empty.")
        return labels

    label_set = set()
    for path in annotation_paths:
        data = load_labelme(path)
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label:
                label_set.add(label)
    return sorted(label_set)


def rectangle_to_bbox(points: Sequence[Sequence[float]]) -> List[float]:
    if len(points) < 2:
        raise ValueError("Rectangle requires at least two points")
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = max(xmax - xmin, 0.0)
    height = max(ymax - ymin, 0.0)
    if width == 0.0 or height == 0.0:
        return []
    return [xmin, ymin, width, height]


def bbox_to_segmentation(bbox: Sequence[float]) -> List[List[float]]:
    x, y, w, h = bbox
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def build_coco_structure(
    annotation_paths: Iterable[Path],
    categories: Dict[str, int],
    image_dir: Path,
) -> Dict:
    images: List[ImageInfo] = []
    annotations: List[AnnotationInfo] = []
    ann_id = 1

    for img_id, ann_path in enumerate(sorted(annotation_paths), start=1):
        data = load_labelme(ann_path)
        file_name = data.get("imagePath") or ann_path.with_suffix(".jpg").name
        width = int(data.get("imageWidth"))
        height = int(data.get("imageHeight"))

        image_path = image_dir / file_name
        if not image_path.exists():
            print(f"Warning: image file {image_path} not found", file=sys.stderr)

        images.append(ImageInfo(id=img_id, file_name=file_name, width=width, height=height))

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if not label or label not in categories:
                continue
            points = shape.get("points", [])
            bbox = rectangle_to_bbox(points)
            if not bbox:
                continue
            area = bbox[2] * bbox[3]
            segmentation = bbox_to_segmentation(bbox)
            annotations.append(
                AnnotationInfo(
                    id=ann_id,
                    image_id=img_id,
                    category_id=categories[label],
                    bbox=[round(v, 2) for v in bbox],
                    area=round(area, 2),
                    segmentation=[[round(v, 2) for v in seg] for seg in segmentation],
                )
            )
            ann_id += 1

    coco = {
        "info": {
            "description": "PIDray Explosive subset converted from LabelMe",
            "version": "1.0",
        },
        "licenses": [],
        "images": [image.__dict__ for image in images],
        "annotations": [
            {
                "id": ann.id,
                "image_id": ann.image_id,
                "category_id": ann.category_id,
                "bbox": ann.bbox,
                "area": ann.area,
                "segmentation": ann.segmentation,
                "iscrowd": 0,
            }
            for ann in annotations
        ],
        "categories": [
            {
                "id": cid,
                "name": name,
                "supercategory": "object",
            }
            for name, cid in categories.items()
        ],
    }
    return coco


def main() -> None:
    args = parse_args()
    annotation_paths = list(sorted(args.annotation_dir.glob("*.json")))
    if not annotation_paths:
        raise SystemExit(f"No LabelMe files found in {args.annotation_dir}")

    categories_list = gather_categories(annotation_paths, args.category_file)
    category_to_id = {name: idx + 1 for idx, name in enumerate(categories_list)}

    coco = build_coco_structure(annotation_paths, category_to_id, args.image_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False)

    print(f"Wrote {len(coco['images'])} images and {len(coco['annotations'])} annotations to {args.output}")


if __name__ == "__main__":
    main()
