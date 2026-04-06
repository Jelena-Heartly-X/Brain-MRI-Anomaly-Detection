"""
training/trainer.py — Complete Conv-MAE training pipeline.

Features:
  - Mixed precision (FP16) via torch.cuda.amp
  - Gradient clipping
  - Early stopping with patience
  - Cosine LR schedule with warmup
  - Reconstruction visualization every N epochs
  - Checkpoint save/load (best val loss)
  - Classifier fine-tuning phase using pseudo-anomalies
  - Tensorboard / wandb logging hooks (optional)
"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import Config
from models.conv_mae import build_model, ConvMAE
from models.classifier import ErrorMapClassifier
from training.losses import ConvMAELoss, ClassifierLoss
from training.optimizers import build_optimizer, build_scheduler
from data.dataset import get_dataloaders
from data.pseudo_anomaly import augment_batch_with_pseudo_anomalies


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        self.counter += 1
        return self.counter >= self.patience  # True = stop


class Trainer:
    """
    Orchestrates Conv-MAE training and optional classifier fine-tuning.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        print(f"[Trainer] Device: {self.device}")

        # Build model
        self.model = build_model(config).to(self.device)

        # Build loss
        self.criterion = ConvMAELoss(config, self.device)

        # Build optimizer & scheduler
        self.optimizer = build_optimizer(self.model, config)

        # DataLoaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(config)

        self.scheduler = build_scheduler(
            self.optimizer if not hasattr(self.optimizer, "base_optimizer")
            else self.optimizer.base_optimizer,
            config
        )

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=config.mixed_precision)

        # Early stopping
        self.early_stopper = EarlyStopping(patience=config.early_stopping_patience)

        # Tracking
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_mse": [],  "val_mse": [],
            "train_ssim": [], "val_ssim": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.start_epoch = 0

        # Classifier (trained after MAE)
        self.classifier: Optional[ErrorMapClassifier] = None
        self.clf_optimizer = None
        self.clf_criterion = ClassifierLoss()

        # Resume if checkpoint exists
        self._try_resume()

    # ─── Checkpoint I/O ───────────────────────────────────────────────────────

    def _checkpoint_path(self, tag: str = "best") -> str:
        return os.path.join(self.config.checkpoint_dir, f"conv_mae_{tag}.pth")

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
            "config": self.config,
        }
        path = self._checkpoint_path(tag)
        torch.save(state, path)
        print(f"[Trainer] Saved checkpoint: {path} (val_loss={val_loss:.5f})")

    def _try_resume(self):
        path = self._checkpoint_path("last")
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.start_epoch = state["epoch"] + 1
            self.best_val_loss = state["val_loss"]
            self.history = state.get("history", self.history)
            print(f"[Trainer] Resumed from epoch {state['epoch']}, val_loss={state['val_loss']:.5f}")

    # ─── Training Loop ────────────────────────────────────────────────────────

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running = {k: 0.0 for k in ["total", "mse", "ssim", "l1", "aux"]}
        n_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.config.mixed_precision):
                recon, mask, aux_recons = self.model(images)
                loss, loss_dict = self.criterion(images, recon, mask, aux_recons)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                self.scaler.unscale_(
                    self.optimizer if not hasattr(self.optimizer, "base_optimizer")
                    else self.optimizer.base_optimizer
                )
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(
                self.optimizer if not hasattr(self.optimizer, "base_optimizer")
                else self.optimizer.base_optimizer
            )
            self.scaler.update()

            for k in running:
                running[k] += loss_dict.get(k, 0.0)

            if step % self.config.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{loss_dict['total']:.4f}",
                    "mse":  f"{loss_dict['mse']:.4f}",
                    "ssim": f"{loss_dict['ssim']:.4f}",
                })

        return {k: v / n_batches for k, v in running.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        running = {k: 0.0 for k in ["total", "mse", "ssim"]}
        n_batches = len(self.val_loader)

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            recon, mask, aux_recons = self.model(images)
            _, loss_dict = self.criterion(images, recon, mask, aux_recons)
            for k in running:
                running[k] += loss_dict.get(k, 0.0)

        return {k: v / n_batches for k, v in running.items()}

    def run(self):
        """Full training pipeline: Conv-MAE → optional classifier."""
        print(f"\n{'='*60}")
        print(f"  Conv-MAE Training — {self.config.epochs} epochs")
        print(f"{'='*60}")

        for epoch in range(self.start_epoch, self.config.epochs):
            t0 = time.time()

            # Train
            train_metrics = self.train_one_epoch(epoch)
            # Validate
            val_metrics = self.validate()

            # Scheduler step
            val_loss = val_metrics["total"]
            try:
                self.scheduler.step()
            except TypeError:
                self.scheduler.step(val_loss)  # ReduceLROnPlateau

            # Get current LR
            try:
                lr = self.optimizer.base_optimizer.param_groups[0]["lr"]
            except AttributeError:
                lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.history["train_loss"].append(train_metrics["total"])
            self.history["val_loss"].append(val_metrics["total"])
            self.history["train_mse"].append(train_metrics["mse"])
            self.history["val_mse"].append(val_metrics["mse"])
            self.history["train_ssim"].append(train_metrics["ssim"])
            self.history["val_ssim"].append(val_metrics["ssim"])
            self.history["lr"].append(lr)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:03d}/{self.config.epochs} | "
                f"train_loss={train_metrics['total']:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"mse={val_metrics['mse']:.4f} | "
                f"ssim={val_metrics['ssim']:.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, tag="best")

            # Save last every epoch
            self.save_checkpoint(epoch, val_loss, tag="last")

            # Visualize reconstructions periodically
            if (epoch + 1) % 10 == 0:
                self._visualize_reconstructions(epoch)

            # Early stopping
            if self.early_stopper(val_loss):
                print(f"[Trainer] Early stopping at epoch {epoch+1}")
                break

        # Plot training curves
        self._plot_training_curves()
        print("\n[Trainer] Conv-MAE training complete.")

        # ── Classifier training phase ──
        print("\n[Trainer] Training anomaly classifier with pseudo-anomalies...")
        self._train_classifier()

    # ─── Classifier Training ──────────────────────────────────────────────────

    def _train_classifier(self, n_epochs: int = 20):
        """
        Train the ErrorMapClassifier on pseudo-anomaly augmented normal data.
        The Conv-MAE weights are frozen; only the classifier is trained.
        """
        self.classifier = ErrorMapClassifier(in_channels=3).to(self.device)
        self.clf_optimizer = torch.optim.AdamW(
            self.classifier.parameters(), lr=3e-4, weight_decay=1e-4
        )

        # Freeze Conv-MAE
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.classifier.train()
        for epoch in range(n_epochs):
            running_loss, correct, total = 0.0, 0, 0
            for batch in tqdm(self.train_loader,
                               desc=f"Classifier epoch {epoch+1}", leave=False):
                images = batch["image"].to(self.device)

                # Generate pseudo-anomalies
                aug_images, _, labels = augment_batch_with_pseudo_anomalies(
                    images.cpu(), self.config,
                    fraction=self.config.pseudo_anomaly_fraction
                )
                aug_images = aug_images.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    recon = self.model.reconstruct(aug_images)

                self.clf_optimizer.zero_grad(set_to_none=True)
                out = self.classifier(aug_images, recon)
                loss = self.clf_criterion(out["logit"].squeeze(), labels.float().to(self.device))
                loss.backward()
                self.clf_optimizer.step()

                running_loss += loss.item()
                preds = (out["prob"] > 0.5).long()
                correct += (preds == labels).sum().item()
                total += labels.shape[0]

            acc = correct / total
            print(f"  [Classifier] Epoch {epoch+1}/{n_epochs} | "
                  f"loss={running_loss/len(self.train_loader):.4f} | acc={acc:.4f}")

        # Save classifier
        clf_path = os.path.join(self.config.checkpoint_dir, "classifier.pth")
        torch.save(self.classifier.state_dict(), clf_path)
        print(f"[Trainer] Classifier saved: {clf_path}")

        # Unfreeze Conv-MAE
        for p in self.model.parameters():
            p.requires_grad = True

    # ─── Visualization ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _visualize_reconstructions(self, epoch: int, n: int = 4):
        self.model.eval()
        batch = next(iter(self.val_loader))
        images = batch["image"][:n].to(self.device)
        recon, mask, _ = self.model(images)

        images_np = images.cpu().numpy()
        recon_np  = recon.cpu().numpy()
        mask_np   = mask.cpu().numpy()
        error_np  = np.abs(images_np - recon_np)

        fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
        titles = ["Input", "Masked Input", "Reconstruction", "Error Map"]
        masked_input = images.cpu().numpy() * (1 - mask_np)

        for i in range(n):
            for j, (img, title) in enumerate(zip(
                [images_np[i, 0], masked_input[i, 0],
                 recon_np[i, 0],  error_np[i, 0]], titles
            )):
                cmap = "hot" if j == 3 else "gray"
                axes[i, j].imshow(img, cmap=cmap, vmin=0, vmax=1)
                if i == 0:
                    axes[i, j].set_title(title, fontsize=11)
                axes[i, j].axis("off")

        plt.suptitle(f"Epoch {epoch+1} Reconstructions", fontsize=13, y=1.01)
        plt.tight_layout()
        save_path = os.path.join(self.config.output_dir, f"recon_epoch_{epoch+1:03d}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [Viz] Saved reconstruction grid: {save_path}")

    def _plot_training_curves(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.history["train_loss"], label="Train")
        axes[0].plot(self.history["val_loss"],   label="Val")
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history["train_ssim"], label="Train")
        axes[1].plot(self.history["val_ssim"],   label="Val")
        axes[1].set_title("SSIM Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.history["lr"])
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.config.output_dir, "training_curves.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[Trainer] Training curves saved: {save_path}")
