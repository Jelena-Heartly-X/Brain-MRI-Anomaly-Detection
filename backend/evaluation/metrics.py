"""
evaluation/metrics.py — Full evaluation metrics for anomaly detection.

Computes:
  - ROC-AUC (slice-level classification)
  - Precision, Recall, F1 at optimal threshold
  - Dice coefficient (pixel-level segmentation)
  - IoU / Jaccard index
  - Confusion matrix
  - Per-threshold analysis for precision-recall curve
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# ─── Slice-level Classification Metrics ──────────────────────────────────────

def compute_roc_auc(scores: np.ndarray, labels: np.ndarray
                    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC-AUC and ROC curve.
    Returns (auc, fpr, tpr, thresholds).
    """
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return auc, fpr, tpr, thresholds


def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray,
                            metric: str = "f1") -> float:
    """
    Find threshold that maximizes F1 (or other metric) on the score distribution.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if metric == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            f1s = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.nanargmax(f1s[:-1])
        return thresholds[best_idx]
    # Youden's J statistic (TPR - FPR)
    fpr, tpr, thresholds_roc = roc_curve(labels, scores)
    j = tpr - fpr
    return thresholds_roc[np.argmax(j)]


def classification_metrics(scores: np.ndarray, labels: np.ndarray,
                             threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Compute full classification metrics at a given threshold.
    If threshold is None, use the F1-optimal threshold.
    """
    if threshold is None:
        threshold = find_optimal_threshold(scores, labels)

    preds = (scores >= threshold).astype(int)
    auc, _, _, _ = compute_roc_auc(scores, labels)
    ap = average_precision_score(labels, scores)

    return {
        "auc":        float(auc),
        "ap":         float(ap),
        "threshold":  float(threshold),
        "f1":         float(f1_score(labels, preds, zero_division=0)),
        "precision":  float(precision_score(labels, preds, zero_division=0)),
        "recall":     float(recall_score(labels, preds, zero_division=0)),
    }


# ─── Pixel-level Segmentation Metrics ────────────────────────────────────────

def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray,
                      smooth: float = 1e-6) -> float:
    """
    Dice similarity coefficient between predicted and ground-truth binary masks.
    """
    pred = pred_mask.astype(bool).ravel()
    gt   = gt_mask.astype(bool).ravel()
    intersection = (pred & gt).sum()
    return float(2 * intersection / (pred.sum() + gt.sum() + smooth))


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray,
              smooth: float = 1e-6) -> float:
    """Intersection over Union (Jaccard index)."""
    pred = pred_mask.astype(bool).ravel()
    gt   = gt_mask.astype(bool).ravel()
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection / (union + smooth))


def pixel_level_metrics(pred_masks: np.ndarray,
                         gt_masks: np.ndarray) -> Dict[str, float]:
    """
    Compute aggregate pixel-level metrics across all samples.

    Args:
        pred_masks: (N, H, W) binary predicted masks
        gt_masks:   (N, H, W) binary ground-truth masks

    Returns:
        dict with mean dice, iou, pixel precision, pixel recall
    """
    dices, ious, precs, recs = [], [], [], []

    for pred, gt in zip(pred_masks, gt_masks):
        if gt.sum() == 0 and pred.sum() == 0:
            dices.append(1.0)
            ious.append(1.0)
        elif gt.sum() == 0:
            dices.append(0.0)
            ious.append(0.0)
        else:
            dices.append(dice_coefficient(pred, gt))
            ious.append(iou_score(pred, gt))

        p = pred.astype(bool).ravel()
        g = gt.astype(bool).ravel()
        tp = (p & g).sum()
        precs.append(tp / (p.sum() + 1e-6))
        recs.append(tp / (g.sum() + 1e-6))

    return {
        "dice":           float(np.mean(dices)),
        "iou":            float(np.mean(ious)),
        "pixel_precision": float(np.mean(precs)),
        "pixel_recall":    float(np.mean(recs)),
    }


# ─── Full Evaluation Report ───────────────────────────────────────────────────

class Evaluator:
    """
    Complete evaluation pipeline for Conv-MAE anomaly detection.
    Runs on test set, computes all metrics, saves plots and report.
    """

    def __init__(self, config):
        self.config = config

    def evaluate(
        self,
        model,
        test_loader,
        device,
        output_dir: str,
    ) -> Dict[str, float]:
        """
        Run full evaluation on test_loader.
        Saves ROC curve, PR curve, and prints report.
        """
        from evaluation.anomaly_score import EnsembleAnomalyScorer
        from evaluation.heatmap import AnomalyHeatmapGenerator
        import torch

        scorer = EnsembleAnomalyScorer(self.config)
        heatmap_gen = AnomalyHeatmapGenerator(self.config)

        all_scores, all_labels, all_error_maps = [], [], []
        all_gt_masks, all_pred_masks = [], []

        model.eval()
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch.get("label", torch.zeros(images.shape[0]))
            gt_masks = batch.get("mask", torch.zeros_like(images))

            with torch.no_grad():
                recon = model.reconstruct(images)

            results = scorer.score_batch(images, recon)

            all_scores.append(results["score"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_error_maps.append(results["error_map"].cpu().squeeze(1).numpy())

            # Generate heatmaps and predicted masks for first N batches
            if batch_idx < 20:
                for i in range(images.shape[0]):
                    img_np  = images[i, 0].cpu().numpy()
                    rec_np  = recon[i, 0].cpu().numpy()
                    err_np  = results["error_map"][i, 0].cpu().numpy()
                    gt_np   = gt_masks[i, 0].cpu().numpy().astype(np.uint8)

                    proc = heatmap_gen.process(img_np, rec_np, err_np, gt_np)
                    all_pred_masks.append(proc["binary_mask"])
                    all_gt_masks.append(gt_np)

                    # Save visualization for first 5 slices per batch
                    if i < 3:
                        save_path = os.path.join(
                            output_dir, f"heatmap_b{batch_idx:02d}_s{i}.png"
                        )
                        heatmap_gen.visualize(
                            img_np, rec_np, err_np, gt_np,
                            save_path=save_path,
                            title=f"Sample {batch_idx*test_loader.batch_size + i} | "
                                  f"Label: {'Anomaly' if labels[i].item() else 'Normal'}"
                        )

        scores = np.concatenate(all_scores)
        labels_np = np.concatenate(all_labels).astype(int)

        # ── Slice-level metrics ──
        clf_metrics = classification_metrics(scores, labels_np)
        print("\n" + "="*50)
        print("SLICE-LEVEL ANOMALY DETECTION")
        print("="*50)
        for k, v in clf_metrics.items():
            print(f"  {k:20s}: {v:.4f}")

        # ── Pixel-level segmentation metrics ──
        seg_metrics = {}
        if all_gt_masks:
            preds_arr = np.array(all_pred_masks)
            gts_arr   = np.array(all_gt_masks)
            seg_metrics = pixel_level_metrics(preds_arr, gts_arr)
            print("\nPIXEL-LEVEL SEGMENTATION")
            print("="*50)
            for k, v in seg_metrics.items():
                print(f"  {k:20s}: {v:.4f}")

        # ── Save plots ──
        self._plot_roc_curve(scores, labels_np, output_dir)
        self._plot_pr_curve(scores, labels_np, output_dir)
        self._plot_score_distribution(scores, labels_np, output_dir)

        all_metrics = {**clf_metrics, **seg_metrics}
        return all_metrics

    def _plot_roc_curve(self, scores, labels, output_dir):
        auc, fpr, tpr, _ = compute_roc_auc(scores, labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Conv-MAE Anomaly Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_pr_curve(self, scores, labels, output_dir):
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_score_distribution(self, scores, labels, output_dir):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(scores[labels == 0], bins=60, alpha=0.7, label="Normal (IXI)", color="#3B8BD4")
        ax.hist(scores[labels == 1], bins=60, alpha=0.7, label="Anomaly (BraTS)", color="#E24B4A")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.set_title("Anomaly Score Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
