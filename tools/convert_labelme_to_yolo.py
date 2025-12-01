#!/usr/bin/env python3
"""Convert LabelMe-like per-image JSON annotations to YOLO .txt files.

Usage:
  python convert_labelme_to_yolo.py --ann-dir /path/to/Explosive/Annotations \
      --out-dir /path/to/labels --mode auto
  # or with an existing classes file (one class per line)
  python convert_labelme_to_yolo.py --ann-dir ... --out-dir ... --mode from_classes --classes-file classes.txt

Outputs:
  - For each image JSON: writes a .txt file with same base name (e.g. image.jpg -> image.txt) in --out-dir (or side-by-side with images if out-dir not given)
  - classes.txt in out-dir (if mode auto)
"""
import argparse
import json
import os
from pathlib import Path
from collections import OrderedDict

def discover_labels(ann_dir):
    labels = OrderedDict()
    for p in Path(ann_dir).glob('*.json'):
        try:
            data = json.load(open(p, 'r'))
        except Exception:
            continue
        for s in data.get('shapes', []):
            lab = s.get('label')
            if lab is None:
                continue
            if lab not in labels:
                labels[lab] = None
    # assign ids in discovery order; you can change to sorted(labels) if you prefer alphabetical
    for i, key in enumerate(labels.keys()):
        labels[key] = i
    return labels

def load_classes_file(fp):
    labels = {}
    with open(fp, 'r') as f:
        for i, line in enumerate(f):
            name = line.strip()
            if not name:
                continue
            labels[name] = i
    return labels

def shape_to_bbox(points):
    # expects rectangle points [[x1,y1],[x2,y2]]
    x1,y1 = points[0]
    x2,y2 = points[1]
    xmin = min(x1,x2)
    xmax = max(x1,x2)
    ymin = min(y1,y2)
    ymax = max(y1,y2)
    return xmin, ymin, xmax, ymax

def convert_one_json(jpath, labels_map, out_txt_path):
    try:
        j = json.load(open(jpath, 'r'))
    except Exception as e:
        print(f'Failed to read {jpath}: {e}')
        return
    W = j.get('imageWidth')
    H = j.get('imageHeight')
    shapes = j.get('shapes', [])
    lines = []
    for s in shapes:
        if s.get('shape_type') != 'rectangle':
            # skip polygons/points unless you want other handling
            continue
        pts = s.get('points')
        if not pts or len(pts) < 2:
            continue
        xmin, ymin, xmax, ymax = shape_to_bbox(pts)
        x_c = (xmin + xmax) / 2.0
        y_c = (ymin + ymax) / 2.0
        w = (xmax - xmin)
        h = (ymax - ymin)
        # guard against missing size
        if not W or not H:
            print(f'Missing W/H in {jpath}, skip')
            continue
        x_c_rel = x_c / W
        y_c_rel = y_c / H
        w_rel = w / W
        h_rel = h / H
        label_name = s.get('label')
        if label_name not in labels_map:
            # unknown label (shouldn't happen if you built mapping beforehand)
            print(f'Warning: label {label_name} not in mapping; skipping')
            continue
        cls_id = labels_map[label_name]
        lines.append(f"{cls_id} {x_c_rel:.6f} {y_c_rel:.6f} {w_rel:.6f} {h_rel:.6f}\n")
    # write file
    if lines:
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt_path, 'w') as f:
            f.write(''.join(lines))
    else:
        # create empty file to indicate no objects (optional)
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        open(out_txt_path, 'w').close()

def main():
    parser = argparse.ArgumentParser(description='Convert LabelMe JSON to YOLO txt')
    parser.add_argument('--ann-dir', required=True, help='Directory with per-image JSONs')
    parser.add_argument('--out-dir', default=None, help='Directory to save .txt labels (default: side-by-side with images)')
    parser.add_argument('--mode', choices=['auto', 'from_classes'], default='auto', help='auto discover classes or use existing classes file')
    parser.add_argument('--classes-file', default=None, help='Path to classes.txt (required if mode=from_classes)')
    parser.add_argument('--ext', default='.json', help='annotation file extension to look for')
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None

    if args.mode == 'auto':
        labels_map = discover_labels(ann_dir)
        # write classes.txt
        if out_dir is None:
            classes_fp = ann_dir / 'classes.txt'
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            classes_fp = out_dir / 'classes.txt'
        with open(classes_fp, 'w') as f:
            # mapping is label -> id
            for label_name, idx in labels_map.items():
                f.write(f"{label_name}\n")
        print(f'Wrote classes.txt to {classes_fp} (num classes={len(labels_map)})')
    else:
        if not args.classes_file:
            parser.error('--classes-file required when mode=from_classes')
        labels_map = load_classes_file(args.classes_file)

    # convert each json -> txt
    for jpath in ann_dir.glob(f'*{args.ext}'):
        base = jpath.stem
        # choose output path
        if out_dir is None:
            # guess image is sibling in parent Images folder or same folder; we write side-by-side with json
            out_txt = jpath.with_suffix('.txt')
        else:
            out_txt = out_dir / (base + '.txt')
        convert_one_json(jpath, labels_map, out_txt)

    print('Conversion finished.')

if __name__ == '__main__':
    main()