import os
import json
import time
import datetime
from collections import defaultdict, deque

import torch
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque)) if self.deque else torch.tensor([0.0])
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32) if self.deque else torch.tensor([0.0])
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / max(1, self.count)

    @property
    def max(self):
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self):
        return self.deque[-1] if self.deque else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'MetricLogger' object has no attribute '{attr}'")

    def log_every(self, data_loader, print_freq, header=None):
        i = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        for obj in data_loader:
            _ = time.time() - end  # data_time (未打印)
            yield obj
            i += 1
            if i % print_freq == 0:
                eta_seconds = (time.time() - start_time) / i * (len(data_loader) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(f"{header} [{i} / {len(data_loader)}]  eta: {eta_string}  {self}")
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1.0
        alpha = x / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return LambdaLR(optimizer, f)


def plot_curves(log_path: str, out_png: str):
    with open(log_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    plt.figure()
    if "train_loss" in history:
        plt.plot(epochs, history["train_loss"], label="train_loss")
    if "val_map" in history and len(history["val_map"]) == len(epochs):
        plt.plot(epochs, history["val_map"], label="val_mAP")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / mAP")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
