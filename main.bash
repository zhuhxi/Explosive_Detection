CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/yolov3_d53_1xb8-_1440-960_-1000e_explosive_dataset_coco.py

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./my_configs/yolov3_d53_1xb8-_1440-960_-1000e_explosive_dataset_coco.py work_dirs/yolov3_d53_1xb8-_1440-960_-1000e_explosive_dataset_coco/epoch_700.pth --show-dir imgs/

CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/detr_r18_1xb16-500e_explosive_dataset_coco.py

CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/yolox_l_8xb8-300e_explosive_dataset_coco.py --resume

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./my_configs/yolox_l_8xb8-300e_explosive_dataset_coco.py work_dirs/yolox_l_8xb8-300e_explosive_dataset_coco/epoch_280.pth --show-dir imgs/

CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/yolox_l_8xb8-1000e_explosive_dataset_coco.py --resume

CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/faster-rcnn_x101-32x4d_fpn_1x_explosive_dataset_coco.py

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./my_configs/faster-rcnn_x101-32x4d_fpn_1x_explosive_dataset_coco.py work_dirs/faster-rcnn_x101-32x4d_fpn_1x_explosive_dataset_coco/epoch_100.pth --show-dir imgs/

CUDA_VISIBLE_DEVICES=1 python tools/train.py /mnt/disk2/zhuhx/PIDray/mmdetection/configs/detr/detr_r101_8xb2-500e_coco.py

CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/detr_r101_8xb2-1000e_explosive_dataset_coco.py

CUDA_VISIBLE_DEVICES=1 python tools/train.py /mnt/disk2/zhuhx/PIDray/mmdetection/configs/dino/dino-4scale_r50_improved_8xb2-12e_coco.py

CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/dino-4scale_r50_improved_8xb2-12e_explosive_dataset_coco.py

CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/dino-4scale_r50_improved_8xb2-1000e_explosive_dataset_coco.py


CUDA_VISIBLE_DEVICES=1 python tools/train.py /mnt/disk2/zhuhx/PIDray/mmdetection/configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py

CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/deformable-detr-refine-twostage_r50_16xb2-1000e_explosive_dataset_coco.py

CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/dino-4scale_r50_improved_8xb2-12e_explosive_coco.py

CUDA_VISIBLE_DEVICES=1 python tools/train.py ./my_configs/dino-4scale_r50_improved_8xb2-12e_explosive_coco_1.py



CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py my_configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_10shot-fine-tuning.py

CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_base-training.py --gpu-id 1
CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py my_configs/fsce_r101_fpn_voc-split1_base-training.py

CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py my_configs/fsce_r101_fpn_voc-split1_base-training.py \
--gpus 1


CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training.py \
--gpus 1


CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_base-training.py \
--gpus 1

CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.py \
--gpus 1

CUDA_VISIBLE_DEVICES=1 \
python tools/detection/misc/initialize_bbox_head.py \
--src1 work_dirs/fsce_r101_fpn_voc-split1_base-training/latest.pth \
--method random_init --save-dir work_dirs/fsce_r101_fpn_voc-split1_base-training

CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning.py \
--gpus 1

CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_10shot-fine-tuning.py \
--gpus 1



CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_2shot-fine-tuning.py \
--gpus 1

CUDA_VISIBLE_DEVICES=1 \
python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_5shot-fine-tuning.py \
--gpus 1


CUDA_VISIBLE_DEVICES=1 \
python tools/detection/test.py \
  configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning.py \
  work_dirs/fsce_r101_fpn_voc-split1_1shot-fine-tuning/latest.pth \
  --eval mAP \
  --show-dir outputs/vis_split1_1shot \
  --show-score-thr 0.85 \
  --gpu-id 0


CUDA_VISIBLE_DEVICES=1 \
python tools/detection/test.py \
  configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning.py \
  work_dirs/fsce_r101_fpn_voc-split1_1shot-fine-tuning/latest.pth \
  --eval mAP \
  --show-dir outputs/vis_split1_1shot \
  --show-score-thr 0.5 \
  --gpu-id 0 \
  --cfg-options \
  model.test_cfg.rpn.nms_pre=600 \
  model.test_cfg.rpn.max_per_img=600 \
  model.test_cfg.rcnn.score_thr=0.5 \
  model.test_cfg.rcnn.nms.iou_threshold=0.3 \
  model.test_cfg.rcnn.max_per_img=50
