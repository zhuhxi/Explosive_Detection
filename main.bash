########### DINO training script ###########
# 1. train 60 / 20 / 20 train val test split explosive dataset
CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/dino-4scale_r50_improved_8xb2-1000e_explosive_dataset_coco.py

# 2. train on explosive 1, 2, 3, test on explosive 4
CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/dino-4scale_r50_improved_8xb2-12e_explosive_coco.py

# 3. train on simple situations, test on complex situations
CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/dino-4scale_r50_improved_8xb2-12e_explosive_coco_1.py

########### few-shot explosive detection training script ###########
# 1. Base training on split1
CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_base-training.py --gpus 1

# 2. 1-shot fine-tuning on split1_initialize bbox head
CUDA_VISIBLE_DEVICES=1 \
python tools/detection/misc/initialize_bbox_head.py \
--src1 work_dirs/fsce_r101_fpn_voc-split1_base-training/latest.pth \
--method random_init --save-dir work_dirs/fsce_r101_fpn_voc-split1_base-training

# 3. 1-shot fine-tuning on split1
CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning.py \
--gpus 1