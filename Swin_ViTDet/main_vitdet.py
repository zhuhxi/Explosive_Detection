import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.coco_xray import CocoXrayDataset
from src.utils.transforms import make_transforms
from src.utils.engine import train_one_epoch, evaluate
from src.utils.utils import plot_curves
from src.models.vitdet_fasterrcnn import build_vitdet_fasterrcnn

import argparse


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_parser():
    p = argparse.ArgumentParser("ViTDet-style + Faster R-CNN for explosive detection (COCO)")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--ann-train", type=str, default="annotations/train.json")
    p.add_argument("--ann-val", type=str, default="annotations/val.json")
    p.add_argument("--ann-test", type=str, default="annotations/test.json")
    p.add_argument("--img-train", type=str, default="train")
    p.add_argument("--img-val", type=str, default="val")
    p.add_argument("--img-test", type=str, default="test")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--resume", type=str, default="")
    p.add_argument("--output-dir", type=str, default="outputs")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--vis-examples", type=int, default=0)

    p.add_argument("--fixed-size", type=int, default=896, help="letterbox to SxS")
    p.add_argument("--vit-name", type=str, default="vit_base_patch16_224")

    p.add_argument("--eval-score-thr", type=float, default=0.05, help="score threshold for eval/vis (no effect on training)")
    p.add_argument("--save-every", type=int, default=1, help="save checkpoint every N epochs (0 to disable)")
    p.add_argument("--debug-eval", action="store_true", help="print number of predicted boxes per eval batch")
    return p


def main():
    from pycocotools.coco import COCO

    args = get_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets / Dataloaders
    train_set = CocoXrayDataset(
        args.data_root, os.path.join(args.data_root, args.ann_train), os.path.join(args.data_root, args.img_train),
        transforms=make_transforms(True, args.fixed_size),
    )
    val_set = CocoXrayDataset(
        args.data_root, os.path.join(args.data_root, args.ann_val), os.path.join(args.data_root, args.img_val),
        transforms=make_transforms(False, args.fixed_size),
    )
    test_set = CocoXrayDataset(
        args.data_root, os.path.join(args.data_root, args.ann_test), os.path.join(args.data_root, args.img_test),
        transforms=make_transforms(False, args.fixed_size),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, collate_fn=CocoXrayDataset.collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=CocoXrayDataset.collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=CocoXrayDataset.collate_fn)

    num_classes = train_set.num_classes
    model = build_vitdet_fasterrcnn(num_classes=num_classes, vit_name=args.vit_name, pretrained_backbone=True, fixed_size=args.fixed_size)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * max(1, len(train_loader))))
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    out_dir = os.path.join(args.output_dir, "vitdet")
    os.makedirs(out_dir, exist_ok=True)
    history_path = os.path.join(out_dir, "history.json")
    vis_dir = os.path.join(out_dir, "vis")

    start_epoch = 0
    best_map = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        if "scaler" in ckpt and scaler is not None and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_map = ckpt.get("best_map", best_map)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best mAP {best_map:.4f}")

    # --- Test only ---
    if args.test_only:
        coco_api = COCO(os.path.join(args.data_root, args.ann_test))
        # 评估前设置阈值
        old_thr = None
        if hasattr(model.roi_heads, "score_thresh"):
            old_thr = model.roi_heads.score_thresh
            model.roi_heads.score_thresh = args.eval_score_thr
        test_stats = evaluate(model, test_loader, device, coco_api, debug=args.debug_eval)
        if old_thr is not None:
            model.roi_heads.score_thresh = old_thr
        print("[Test] mAP={:.3f} | AP50={:.3f} | AP75={:.3f} | AR1={:.3f} | AR10={:.3f} | AR100={:.3f}".format(
            float(test_stats.get("mAP", 0.0)), float(test_stats.get("AP50", 0.0)), float(test_stats.get("AP75", 0.0)),
            float(test_stats.get("AR1", 0.0)), float(test_stats.get("AR10", 0.0)), float(test_stats.get("AR100", 0.0)),
        ))
        return

    history = {"train_loss": [], "val_map": []}
    coco_val = COCO(os.path.join(args.data_root, args.ann_val))

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler, clip_grad=args.clip_grad, lr_scheduler=lr_scheduler)
        history["train_loss"].append(float(train_stats["loss"]))

        # 评估阈值（只影响评估）
        old_thr = None
        if hasattr(model.roi_heads, "score_thresh"):
            old_thr = model.roi_heads.score_thresh
            model.roi_heads.score_thresh = args.eval_score_thr
        val_stats = evaluate(model, val_loader, device, coco_val, debug=args.debug_eval)
        if old_thr is not None:
            model.roi_heads.score_thresh = old_thr

        print("[Val] mAP={:.3f} | AP50={:.3f} | AP75={:.3f} | AR1={:.3f} | AR10={:.3f} | AR100={:.3f}".format(
            float(val_stats.get("mAP", 0.0)), float(val_stats.get("AP50", 0.0)), float(val_stats.get("AP75", 0.0)),
            float(val_stats.get("AR1", 0.0)), float(val_stats.get("AR10", 0.0)), float(val_stats.get("AR100", 0.0)),
        ))
        history["val_map"].append(float(val_stats.get("mAP", 0.0)))

        # 保存
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "scaler": scaler.state_dict() if scaler is not None else None, "best_map": best_map, "args": vars(args)}
        if args.save_every and ((epoch + 1) % args.save_every == 0):
            torch.save(ckpt, os.path.join(out_dir, f"checkpoint_{epoch+1:03d}.pth"))
        torch.save(ckpt, os.path.join(out_dir, "last.pth"))
        if val_stats.get("mAP", 0.0) > best_map:
            best_map = val_stats["mAP"]
            torch.save(ckpt, os.path.join(out_dir, "best.pth"))
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # 训练完：画曲线 + 测试 + 可视化
    plot_curves(history_path, os.path.join(out_dir, "loss_curve.png"))

    if os.path.exists(os.path.join(out_dir, "best.pth")):
        ckpt = torch.load(os.path.join(out_dir, "best.pth"), map_location=device)
        model.load_state_dict(ckpt["model"])

    from pycocotools.coco import COCO as _COCO
    coco_test = _COCO(os.path.join(args.data_root, args.ann_test))

    old_thr = None
    if hasattr(model.roi_heads, "score_thresh"):
        old_thr = model.roi_heads.score_thresh
        model.roi_heads.score_thresh = args.eval_score_thr
    test_stats = evaluate(model, test_loader, device, coco_test, debug=args.debug_eval)
    if old_thr is not None:
        model.roi_heads.score_thresh = old_thr

    print("[Test] mAP={:.3f} | AP50={:.3f} | AP75={:.3f} | AR1={:.3f} | AR10={:.3f} | AR100={:.3f}".format(
        float(test_stats.get("mAP", 0.0)), float(test_stats.get("AP50", 0.0)), float(test_stats.get("AP75", 0.0)),
        float(test_stats.get("AR1", 0.0)), float(test_stats.get("AR10", 0.0)), float(test_stats.get("AR100", 0.0)),
    ))

    from src.utils.visualize import visualize_predictions
    visualize_predictions(model, test_set, device, out_dir=vis_dir, num_images=max(5, args.vis_examples), score_thr=args.eval_score_thr)
    print(f"Saved visualizations to: {vis_dir}")


if __name__ == "__main__":
    main()
