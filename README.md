# 🔥 DINO Training Experiment Summary

The dataset download link is provided at https://kuacae-my.sharepoint.com/:f:/g/personal/100065148_ku_ac_ae/Eu3jYuhl_WlAhPSGQc7HOCMBzlWmYt2SvKI8jzZNEyPTTQ?e=1sIRdy; after downloading, simply place the dataset inside the data/ directory.

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


# 🔍 Few-Shot Learning Pipeline

## 1️⃣ Base Training on VOC Split1 (Train Only Base Classes)

```bash
CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_base-training.py --gpus 1
```

---

## 2️⃣ Initialize BBox Head Using the Base Model (Preparation for FSCE / Few-Shot)

```bash
CUDA_VISIBLE_DEVICES=1 \
python tools/detection/misc/initialize_bbox_head.py \
--src1 work_dirs/fsce_r101_fpn_voc-split1_base-training/latest.pth \
--method random_init --save-dir work_dirs/fsce_r101_fpn_voc-split1_base-training
```

---

## 3️⃣ FSCE Fine-Tuning on Few-Shot Data (1-Shot)

```bash
CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning.py \
--gpus 1
```
