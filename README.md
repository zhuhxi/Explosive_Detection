# 🔥 DINO Training Experiment Summary

## 1️⃣ Training on the Full Test Set (Simple + Occlusion)

Train DINO on the complete explosive dataset, including both Simple and Occlusion scenarios:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
    ./my_configs/dino-4scale_r50_improved_8xb2-1000e_explosive_dataset_coco.py
```

---

## 2️⃣ Simple → Occlusion Transfer Training

Train on the Simple scenario and evaluate the model’s transfer ability to the Occlusion scenario:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
    ./my_configs/dino-4scale_r50_improved_8xb2-12e_explosive_coco.py
```

---

## 3️⃣ Cross-Type Generalization Training (Explosives 1–3 → Unseen Type 4)

Evaluate DINO’s ability to generalize from known explosive types (1–3) to an unseen type (4):

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
    ./my_configs/dino-4scale_r50_improved_8xb2-12e_explosive_coco_1.py
```
