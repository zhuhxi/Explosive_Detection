
import random
from typing import Tuple, Dict, Any, Callable
from PIL import Image
import torchvision.transforms.functional as F
import torch

class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms: image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        # 如果已经是 Tensor，直接返回；否则从 PIL/ndarray 转成 Tensor
        if isinstance(image, torch.Tensor):
            return image, target
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image); w = image.shape[-1]
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes.clone(); boxes[:, [0, 2]] = w - boxes[:, [2, 0]]; target["boxes"] = boxes
        return image, target

class LetterboxResize:
    def __init__(self, size: int): self.size = int(size)
    def __call__(self, image, target):
        if isinstance(image, Image.Image): w0, h0 = image.size; image = F.to_tensor(image)
        else: _, h0, w0 = image.shape
        S = self.size; scale = min(S / h0, S / w0)
        new_h = int(round(h0 * scale)); new_w = int(round(w0 * scale))
        image = F.resize(image, [new_h, new_w])
        pad_y = (S - new_h) // 2; pad_x = (S - new_w) // 2
        padding = [pad_x, pad_y, S - new_w - pad_x, S - new_h - pad_y]
        image = F.pad(image, padding, fill=0)
        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes * torch.tensor([scale, scale, scale, scale], dtype=boxes.dtype)
            boxes[:, [0, 2]] += pad_x; boxes[:, [1, 3]] += pad_y; target["boxes"] = boxes
        return image, target

def make_transforms(train: bool = True, fixed_size: int | None = None):
    tfs = []
    if fixed_size is not None: tfs.append(LetterboxResize(fixed_size))
    tfs.append(ToTensor())
    if train: tfs.append(RandomHorizontalFlip(0.5))
    return Compose(tfs)
