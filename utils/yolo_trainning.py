import torch
from torch import optim

import os
from pathlib import Path


class CustomLR:
    def __init__(
        self, optimizer, warm_up=(0.001, 0.01, 5), T_max=100, eta_min=1e-4, step=1
    ):
        self.optimizer = optimizer
        self.warm_up = warm_up
        self.T_max = T_max
        self.eta_min = eta_min
        self.steper = step
        self.count = 0
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=eta_min
        )

    def step(self):
        self.count += 1
        if self.count % self.steper == 0:
            if self.warm_up[2] != 0 and self.count <= self.warm_up[2]:
                if self.count == self.warm_up[2]:
                    self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=self.T_max, eta_min=self.eta_min
                    )
                else:
                    self._warm_up(self.warm_up[0], self.warm_up[1], self.warm_up[2])
            else:
                self.lr_scheduler.step()

    def _warm_up(self, start_lr=0.001, end_lr=0.01, warmup_step=5):
        scale = (end_lr - start_lr) / warmup_step
        dynamic_lr = start_lr + scale * self.count
        lr = min(dynamic_lr, end_lr)
        for param in self.optimizer.param_groups:
            param["lr"] = lr

    def get_lr(self):
        lr = self.optimizer.param_groups[0]["lr"]
        return lr


def save_best_model(losses, model, optimizer, epoch):
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    current_loss = losses[-1]
    checkpoint = {
        "model": model,
        "optimizer": optimizer,
        "epoch": epoch,
    }

    # 只有当非第一轮，且当前为最优时才保存 best
    if epoch > 0 and len(losses) != 1 and current_loss < min(losses[:-1]):
        torch.save(checkpoint, ".checkpoint.pth")
        os.replace(
            ".checkpoint.pth", weights_dir / f"{current_loss:.4f}_best_{epoch}.pt"
        )
    else:
        torch.save(checkpoint, ".checkpoint.pth")
        os.replace(".checkpoint.pth", weights_dir / f"{current_loss:.4f}_{epoch}.pt")


def continue_train(ckp_path, device):
    ckp = torch.load(ckp_path, weights_only=False, map_location=device)
    model = ckp["model"]
    return model
