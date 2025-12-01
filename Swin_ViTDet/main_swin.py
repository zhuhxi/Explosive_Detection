
import os, json, random, numpy as np, torch
from torch.utils.data import DataLoader
from src.datasets.coco_xray import CocoXrayDataset
from src.utils.transforms import make_transforms
from src.utils.engine import train_one_epoch, evaluate
from src.utils.metrics import COCO, CocoEvaluator
from src.utils.utils import plot_curves
from src.utils.visualize import visualize_predictions
from src.utils.visualize import visualize_predictions
from src.utils.visualize import visualize_predictions
from src.utils.cli import get_common_parser
from src.models.swin_fasterrcnn import build_swin_fasterrcnn

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    from src.utils.visualize import visualize_predictions
    parser = get_common_parser("Swin-T + Faster R-CNN")
    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224")
    # 在 parser 定义后面加这一行
    parser.add_argument("--save-every", type=int, default=1,
                        help="每 N 个 epoch 保存一个 checkpoint_###.pth；设为 0 则不保存按epoch的快照")

    args = parser.parse_args()
    set_seed(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = CocoXrayDataset(args.data_root, os.path.join(args.data_root, args.ann_train), os.path.join(args.data_root, args.img_train), transforms=make_transforms(True, args.fixed_size))
    val_set   = CocoXrayDataset(args.data_root, os.path.join(args.data_root, args.ann_val),   os.path.join(args.data_root, args.img_val),   transforms=make_transforms(False, args.fixed_size))
    test_set  = CocoXrayDataset(args.data_root, os.path.join(args.data_root, args.ann_test),  os.path.join(args.data_root, args.img_test),  transforms=make_transforms(False, args.fixed_size))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, collate_fn=CocoXrayDataset.collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=CocoXrayDataset.collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=CocoXrayDataset.collate_fn)

    num_classes = train_set.num_classes
    model = build_swin_fasterrcnn(num_classes=num_classes, backbone_name=args.backbone, pretrained_backbone=True, fixed_size=args.fixed_size)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)))
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    out_dir = os.path.join(args.output_dir, "swin"); os.makedirs(out_dir, exist_ok=True)
    history_path = os.path.join(out_dir, "history.json"); vis_dir = os.path.join(out_dir, "vis")

    start_epoch = 0; best_map = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        if "scaler" in ckpt and scaler is not None and ckpt["scaler"] is not None: scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1; best_map = ckpt.get("best_map", best_map)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best mAP {best_map:.4f}")

    if args.test_only:
        from pycocotools.coco import COCO
        coco_api = COCO(os.path.join(args.data_root, args.ann_test))
        test_stats = evaluate(model, test_loader, device, coco_api)
        print("Test metrics:", test_stats)
        if args.vis_examples > 0:
            from src.utils.visualize import visualize_predictions
            visualize_predictions(model, test_set, device, out_dir=vis_dir, num_images=args.vis_examples, score_thr=0.35)
            print(f"Saved {args.vis_examples} visualizations to: {vis_dir}")
        return

    history = {"train_loss": [], "val_map": []}
    from pycocotools.coco import COCO
    coco_val = COCO(os.path.join(args.data_root, args.ann_val))

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler, clip_grad=args.clip_grad, lr_scheduler=lr_scheduler)
        history["train_loss"].append(float(train_stats["loss"]))

        val_stats = evaluate(model, val_loader, device, coco_val)
        history["val_map"].append(float(val_stats.get("mAP", 0.0)))

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "scaler": scaler.state_dict() if scaler is not None else None, "best_map": best_map, "args": vars(args)}
        if args.save_every and ((epoch + 1) % args.save_every == 0):
            torch.save(ckpt, os.path.join(out_dir, f"checkpoint_{epoch + 1:03d}.pth"))
        torch.save(ckpt, os.path.join(out_dir, f"last.pth"))
        if val_stats.get("mAP", 0.0) > best_map: best_map = val_stats["mAP"]; torch.save(ckpt, os.path.join(out_dir, "best.pth"))
        with open(history_path, "w", encoding="utf-8") as f: json.dump(history, f, indent=2)

    plot_curves(history_path, os.path.join(out_dir, "loss_curve.png"))
    if os.path.exists(os.path.join(out_dir, "best.pth")):
        ckpt = torch.load(os.path.join(out_dir, "best.pth"), map_location=device); model.load_state_dict(ckpt["model"])
    coco_test = COCO(os.path.join(args.data_root, args.ann_test)); test_stats = evaluate(model, test_loader, device, coco_test)
    print("Final test metrics:", test_stats)

    visualize_predictions(model, test_set, device, out_dir=vis_dir, num_images=max(5, args.vis_examples), score_thr=0.35)
    print(f"Saved visualizations to: {vis_dir}")

if __name__ == "__main__": main()
