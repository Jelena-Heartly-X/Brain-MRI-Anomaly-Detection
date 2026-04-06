"""
training/optimizers.py — Optimizer and LR scheduler factory.

Supports: Adam, AdamW, RMSprop, SGD with momentum, Lookahead wrapper.
Schedulers: CosineAnnealingLR with linear warmup, StepLR, ReduceLROnPlateau.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau,
    LinearLR, SequentialLR
)
from torch.nn import Module


class Lookahead(optim.Optimizer):
    """
    Lookahead optimizer (Zhang et al., 2019).
    Wraps any base optimizer and periodically interpolates fast weights
    toward slow weights, improving convergence stability.
    """

    def __init__(self, base_optimizer: optim.Optimizer,
                 k: int = 5, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        self.param_groups = base_optimizer.param_groups
        # Store slow weights
        self.slow_weights = [
            [p.clone().detach() for p in group["params"]]
            for group in self.param_groups
        ]

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        if self._step_count % self.k == 0:
            for group, slow in zip(self.param_groups, self.slow_weights):
                for fast_p, slow_p in zip(group["params"], slow):
                    slow_p.add_(self.alpha * (fast_p.data - slow_p))
                    fast_p.data.copy_(slow_p)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "base": self.base_optimizer.state_dict(),
            "slow": self.slow_weights,
            "step": self._step_count,
        }

    def load_state_dict(self, state):
        self.base_optimizer.load_state_dict(state["base"])
        self.slow_weights = state["slow"]
        self._step_count = state["step"]


def build_optimizer(model: Module, config) -> optim.Optimizer:
    """
    Build optimizer from config.
    Supports different weight decay for normalization layers (no decay).
    """
    # Separate params: no weight decay for BN/LN/biases
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bn" in name or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    opt_name = config.optimizer.lower()

    if opt_name == "adam":
        base = optim.Adam(param_groups, lr=config.lr, betas=(0.9, 0.999))
    elif opt_name == "adamw":
        base = optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.999))
    elif opt_name == "rmsprop":
        base = optim.RMSprop(param_groups, lr=config.lr, momentum=0.9)
    elif opt_name == "sgd":
        base = optim.SGD(param_groups, lr=config.lr, momentum=0.9, nesterov=True)
    elif opt_name == "lookahead":
        adamw = optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.999))
        base = Lookahead(adamw, k=5, alpha=0.5)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return base


def build_scheduler(optimizer: optim.Optimizer, config,
                    steps_per_epoch: int = None):
    """
    Build LR scheduler with optional linear warmup.
    """
    sched_name = config.scheduler.lower()
    warmup_epochs = config.lr_warmup_epochs

    if sched_name == "cosine":
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs - warmup_epochs,
            eta_min=config.lr * 0.01,
        )
    elif sched_name == "step":
        base_scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    elif sched_name == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=10, min_lr=config.lr * 0.01, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")

    if warmup_epochs > 0:
        # Real optimizer may be Lookahead without standard LR support
        try:
            warmup = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=warmup_epochs
            )
            return SequentialLR(optimizer, [warmup, base_scheduler],
                                milestones=[warmup_epochs])
        except Exception:
            return base_scheduler

    return base_scheduler
