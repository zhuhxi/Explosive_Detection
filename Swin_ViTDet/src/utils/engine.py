from __future__ import annotations
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import CocoEvaluator


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler=None,
    print_freq: int = 50,
    clip_grad: float = 1.0,
    lr_scheduler=None,
):
    model.train()
    total = 0.0
    count = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for it, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses).backward()
            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if clip_grad and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        total += float(losses.item())
        count += 1
        if it % print_freq == 0:
            pbar.set_postfix(loss=f"{(total / max(1, count)):.4f}")

    return {"loss": (total / max(1, count))}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    coco_api,
    iou_types=("bbox",),
    debug: bool = False,
):
    """
    条件反 letterbox：
      - 若前向输入是等比缩放 + 补边到 S×S（--fixed-size），把预测框从 S×S 反投影回原图坐标再给 COCOeval
      - 若未 letterbox，则不改动
    并且在 evaluator 里把 label -> COCO category_id 的映射做好，避免类别抖动。
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()

    # 从数据集抓取稳定映射
    ds = getattr(data_loader, "dataset", None)
    label_to_coco = getattr(ds, "label_to_coco", None)

    coco_evaluator = CocoEvaluator(coco_api, iou_types, label_to_coco=label_to_coco)

    for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
        pre_sizes = [img.shape[-2:] for img in images]  # (Hs, Ws)

        images = [img.to(device) for img in images]
        outputs = model(images)
        outputs = [{k: v.to("cpu") for k, v in o.items()} for o in outputs]

        fixed_outputs = []
        for o, t, (Hs, Ws) in zip(outputs, targets, pre_sizes):
            H0 = int(t["orig_size"][0].item())
            W0 = int(t["orig_size"][1].item())

            # 是否 letterbox：S×S 且与原图不同
            did_letterbox = (Hs == Ws) and (Hs != H0 or Ws != W0)

            if did_letterbox:
                S = int(Hs)
                s = min(S / H0, S / W0)
                new_h = int(round(H0 * s))
                new_w = int(round(W0 * s))
                pad_y = (S - new_h) // 2
                pad_x = (S - new_w) // 2

                boxes = o["boxes"]
                if boxes.numel() > 0:
                    boxes = boxes.clone()
                    boxes[:, [0, 2]] -= pad_x
                    boxes[:, [1, 3]] -= pad_y
                    boxes /= s
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W0)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H0)
                    o["boxes"] = boxes

            fixed_outputs.append(o)

        if debug:
            num_preds = sum(int(o["boxes"].shape[0]) for o in fixed_outputs)
            print(f"[EvalDebug] batch preds={num_preds}")

        res = {t["image_id"].item(): o for t, o in zip(targets, fixed_outputs)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return stats
