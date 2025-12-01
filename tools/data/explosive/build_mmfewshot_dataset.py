#!/usr/bin/env python3
"""Build an mmfewshot-ready dataset from PIDray base/novel sources.

This script converts the PIDray base dataset stored in LabelMe format and the
novel explosive dataset stored in COCO format into a unified COCO dataset
layout expected by mmfewshot. Images are symlinked (or optionally copied) into
``data/fewshot_dataset/coco`` and consolidated annotations are written to
``instances_train2014.json`` and ``instances_val2014.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class CocoSplit:
    images: List[Dict]
    annotations: List[Dict]
    next_image_id: int
    next_ann_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ann-dir",
        type=Path,
        default=Path("data/Explosive/Annotations"),
        help="Directory containing LabelMe JSON annotations for base classes.",
    )
    parser.add_argument(
        "--base-img-dir",
        type=Path,
        default=Path("data/Explosive/Images"),
        help="Directory containing images referenced by the base annotations.",
    )
    parser.add_argument(
        "--novel-root",
        type=Path,
        default=Path("data/explosive_coco_1"),
        help="Root directory of the novel dataset in COCO format.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/fewshot_dataset/coco"),
        help="Target directory to populate with the mmfewshot dataset.",
    )
    parser.add_argument(
        "--base-val-ratio",
        type=float,
        default=0.1,
        help="Fraction of base images reserved for validation (0-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed used for the base train/val split.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    return parser.parse_args()


def load_labelme(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gather_base_categories(annotation_paths: Iterable[Path]) -> List[str]:
    categories = set()
    for path in annotation_paths:
        data = load_labelme(path)
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label:
                categories.add(label)
    return sorted(categories)


def gather_novel_categories(annotation_paths: Iterable[Path]) -> List[str]:
    categories = set()
    for path in annotation_paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for cat in data.get("categories", []):
            name = cat.get("name")
            if name:
                categories.add(name)
    return sorted(categories)


def rectangle_to_bbox(points: Sequence[Sequence[float]]) -> List[float]:
    if len(points) < 2:
        return []
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = max(xmax - xmin, 0.0)
    height = max(ymax - ymin, 0.0)
    if width <= 0.0 or height <= 0.0:
        return []
    return [xmin, ymin, width, height]


def bbox_to_segmentation(bbox: Sequence[float]) -> List[List[float]]:
    x, y, w, h = bbox
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def convert_base_split(
    ann_paths: Sequence[Path],
    image_dir: Path,
    split_dir: str,
    cat_to_id: Dict[str, int],
    image_id_start: int,
    ann_id_start: int,
    file_tasks: List[Tuple[Path, Path]],
) -> CocoSplit:
    images: List[Dict] = []
    annotations: List[Dict] = []
    image_id = image_id_start
    ann_id = ann_id_start

    for ann_path in ann_paths:
        data = load_labelme(ann_path)
        image_name = Path(data.get("imagePath") or ann_path.with_suffix(".jpg").name).name
        src_image = image_dir / image_name
        if not src_image.exists():
            raise FileNotFoundError(f"Base image not found: {src_image}")
        dest_rel = Path(split_dir) / image_name
        images.append(
            {
                "id": image_id,
                "file_name": str(dest_rel).replace("\\", "/"),
                "width": int(data.get("imageWidth")),
                "height": int(data.get("imageHeight")),
            }
        )
        file_tasks.append((src_image, dest_rel))

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if not label or label not in cat_to_id:
                continue
            bbox = rectangle_to_bbox(shape.get("points", []))
            if not bbox:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_to_id[label],
                    "bbox": [round(float(v), 2) for v in bbox],
                    "area": round(float(bbox[2] * bbox[3]), 2),
                    "iscrowd": 0,
                    "segmentation": [
                        [round(float(v), 2) for v in segment]
                        for segment in bbox_to_segmentation(bbox)
                    ],
                }
            )
            ann_id += 1

        image_id += 1

    return CocoSplit(images, annotations, image_id, ann_id)


def convert_novel_split(
    coco_data: Dict,
    image_root: Path,
    split_dir: str,
    cat_to_id: Dict[str, int],
    image_id_start: int,
    ann_id_start: int,
    file_tasks: List[Tuple[Path, Path]],
) -> CocoSplit:
    images: List[Dict] = []
    annotations: List[Dict] = []
    image_id = image_id_start
    ann_id = ann_id_start

    cat_id_map = {cat["id"]: cat_to_id[cat["name"]] for cat in coco_data.get("categories", []) if cat.get("name") in cat_to_id}
    image_id_map: Dict[int, int] = {}

    for image in coco_data.get("images", []):
        original_name = Path(image["file_name"])
        src_image = image_root / image["file_name"]
        if not src_image.exists():
            raise FileNotFoundError(f"Novel image not found: {src_image}")
        dest_rel = Path(split_dir) / original_name.name
        images.append(
            {
                "id": image_id,
                "file_name": str(dest_rel).replace("\\", "/"),
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
        )
        file_tasks.append((src_image, dest_rel))
        image_id_map[image["id"]] = image_id
        image_id += 1

    for ann in coco_data.get("annotations", []):
        mapped_image_id = image_id_map.get(ann["image_id"])
        mapped_cat_id = cat_id_map.get(ann["category_id"])
        if mapped_image_id is None or mapped_cat_id is None:
            continue
        annotations.append(
            {
                "id": ann_id,
                "image_id": mapped_image_id,
                "category_id": mapped_cat_id,
                "bbox": [round(float(v), 2) for v in ann.get("bbox", [])],
                "area": round(float(ann.get("area", 0.0)), 2),
                "iscrowd": int(ann.get("iscrowd", 0)),
                "segmentation": ann.get("segmentation", []),
            }
        )
        ann_id += 1

    return CocoSplit(images, annotations, image_id, ann_id)


def safe_symlink_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.samefile(src):
                return
        except FileNotFoundError:
            pass
        if dst.is_dir() and not dst.is_symlink():
            raise IsADirectoryError(f"Destination exists and is a directory: {dst}")
        dst.unlink()

    if copy:
        shutil.copy2(src, dst)
    else:
        rel_src = os.path.relpath(src, dst.parent)
        os.symlink(rel_src, dst)


def prepare_output_dirs(output_root: Path) -> None:
    (output_root / "annotations").mkdir(parents=True, exist_ok=True)
    for split in ("train2014", "val2014"):
        split_dir = output_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    base_ann_paths = sorted(args.base_ann_dir.glob("*.json"))
    if not base_ann_paths:
        raise SystemExit(f"No LabelMe annotations found in {args.base_ann_dir}")
    if not args.base_img_dir.exists():
        raise SystemExit(f"Base image directory not found: {args.base_img_dir}")

    novel_ann_dir = args.novel_root / "annotations"
    novel_train_path = novel_ann_dir / "instances_train.json"
    novel_val_path = novel_ann_dir / "instances_val.json"
    if not novel_train_path.exists() or not novel_val_path.exists():
        raise SystemExit(f"Missing novel dataset annotations under {novel_ann_dir}")

    prepare_output_dirs(args.output_root)

    rng = random.Random(args.seed)
    shuffled_ann_paths = base_ann_paths[:]
    rng.shuffle(shuffled_ann_paths)

    val_count = 0
    if 0 < args.base_val_ratio < 1:
        val_count = max(1, int(len(shuffled_ann_paths) * args.base_val_ratio))
    base_val_paths = shuffled_ann_paths[:val_count]
    base_train_paths = shuffled_ann_paths[val_count:] if val_count else shuffled_ann_paths

    base_categories = gather_base_categories(base_ann_paths)
    with novel_train_path.open("r", encoding="utf-8") as handle:
        novel_train_data = json.load(handle)
    with novel_val_path.open("r", encoding="utf-8") as handle:
        novel_val_data = json.load(handle)
    novel_categories = gather_novel_categories([novel_train_path, novel_val_path])

    cat_to_id: Dict[str, int] = {}
    categories: List[Dict] = []
    for name in base_categories + [c for c in novel_categories if c not in base_categories]:
        if name not in cat_to_id:
            cat_to_id[name] = len(cat_to_id) + 1
            categories.append({"id": cat_to_id[name], "name": name, "supercategory": "object"})

    file_tasks: List[Tuple[Path, Path]] = []

    train_images: List[Dict] = []
    train_annotations: List[Dict] = []
    train_split = convert_base_split(
        base_train_paths,
        args.base_img_dir,
        "train2014",
        cat_to_id,
        image_id_start=1,
        ann_id_start=1,
        file_tasks=file_tasks,
    )
    train_images.extend(train_split.images)
    train_annotations.extend(train_split.annotations)

    novel_train_split = convert_novel_split(
        novel_train_data,
        args.novel_root,
        "train2014",
        cat_to_id,
        image_id_start=train_split.next_image_id,
        ann_id_start=train_split.next_ann_id,
        file_tasks=file_tasks,
    )
    train_images.extend(novel_train_split.images)
    train_annotations.extend(novel_train_split.annotations)

    val_images: List[Dict] = []
    val_annotations: List[Dict] = []
    val_split = convert_base_split(
        base_val_paths,
        args.base_img_dir,
        "val2014",
        cat_to_id,
        image_id_start=1,
        ann_id_start=1,
        file_tasks=file_tasks,
    )
    val_images.extend(val_split.images)
    val_annotations.extend(val_split.annotations)

    novel_val_split = convert_novel_split(
        novel_val_data,
        args.novel_root,
        "val2014",
        cat_to_id,
        image_id_start=val_split.next_image_id,
        ann_id_start=val_split.next_ann_id,
        file_tasks=file_tasks,
    )
    val_images.extend(novel_val_split.images)
    val_annotations.extend(novel_val_split.annotations)

    annotations_dir = args.output_root / "annotations"
    train_out = annotations_dir / "instances_train2014.json"
    val_out = annotations_dir / "instances_val2014.json"

    dataset_info = {
        "info": {"description": "PIDray mmfewshot combined dataset", "version": "1.0"},
        "licenses": [],
        "categories": sorted(categories, key=lambda x: x["id"]),
    }

    with train_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **dataset_info,
                "images": train_images,
                "annotations": train_annotations,
            },
            handle,
            ensure_ascii=False,
        )

    with val_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **dataset_info,
                "images": val_images,
                "annotations": val_annotations,
            },
            handle,
            ensure_ascii=False,
        )

    for src, dest_rel in file_tasks:
        dest_path = args.output_root / dest_rel
        safe_symlink_or_copy(src, dest_path, copy=args.copy_images)

    print(f"Train images: {len(train_images)}, annotations: {len(train_annotations)}")
    print(f"Val images: {len(val_images)}, annotations: {len(val_annotations)}")
    print(f"Categories: {len(categories)} -> {', '.join(cat_to_id.keys())}")
    print(f"Dataset written to {args.output_root}")


if __name__ == "__main__":
    main()
