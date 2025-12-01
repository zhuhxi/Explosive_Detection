CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_base-training.py --gpus 1

CUDA_VISIBLE_DEVICES=1 \
python tools/detection/misc/initialize_bbox_head.py \
--src1 work_dirs/fsce_r101_fpn_voc-split1_base-training/latest.pth \
--method random_init --save-dir work_dirs/fsce_r101_fpn_voc-split1_base-training

CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning.py \
--gpus 1