"""Per-class test accuracy across all 32 classes (Stream 3).

Expected pattern: improvement monotonically larger for rarer classes,
approximately no difference for the most common classes.
"""

import torch
import numpy as np
from scipy.stats import pearsonr


@torch.no_grad()
def per_class_accuracy(model, test_loader, device, n_classes: int = 32):
    """Compute per-class accuracy for all classes.

    Args:
        model: Trained PlayEmbedder.
        test_loader: DataLoader for test set.
        device: torch device.
        n_classes: Number of classes.

    Returns:
        numpy array of shape [n_classes] with accuracy per class.
    """
    correct = torch.zeros(n_classes)
    total = torch.zeros(n_classes)

    model.eval()
    for batch in test_loader:
        seq = batch['sequence'].to(device)
        logits, _ = model(seq)
        preds = logits.argmax(dim=-1).cpu()
        labels = batch['label']

        for c in range(n_classes):
            mask = labels == c
            correct[c] += (preds[mask] == c).sum()
            total[c] += mask.sum()

    return (correct / total.clamp(min=1)).numpy()


def rank_improvement_correlation(acc_c1, acc_c4, n_classes: int = 32):
    """Compute Pearson correlation between accuracy improvement and Zipf rank.

    Expected: r > 0.80 — improvement is rank-ordered by rarity.

    Args:
        acc_c1: Per-class accuracy for C1 (baseline).
        acc_c4: Per-class accuracy for C4 (Zipf+SW).

    Returns:
        Dict with 'r', 'p_value', and 'improvement' array.
    """
    improvement = acc_c4 - acc_c1
    ranks = np.arange(1, n_classes + 1)  # 1 = most common, 32 = rarest

    r, p = pearsonr(ranks, improvement)
    return {
        'r': r,
        'p_value': p,
        'improvement': improvement,
    }
