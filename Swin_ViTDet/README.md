# X-ray Explosive Detection (COCO format)

Two object-detection baselines on your X‑ray luggage dataset (COCO format):
1. **Swin-T + FPN + Faster R-CNN** (strong two-stage baseline)
2. **ViTDet-style Faster R-CNN** (ViT backbone + Simple Feature Pyramid)

Features:
- Full training / validation / testing loops.
- COCO-style mAP (0.5:0.95), AP50, AP75, precision/recall.
- Mixed precision training, gradient clipping.
- Checkpointing every epoch & resume (`--resume`) and best model saving.
- Loss curve auto-plot & metric logs to `outputs/`.
- 5+ visualization samples on the test set with predicted boxes and scores.
- Works with your folder structure:
  ```
  explosive_dataset_coco/
    annotations/   # train.json, val.json, test.json (COCO format)
    train/         # images
    val/
    test/
  ```

> **Important:** Your JSON currently uses `category_id: 0` for "explosive". TorchVision reserves
> `0` for the background class. The code automatically remaps category ids to start at **1**,
> so you can keep your JSON as-is.
