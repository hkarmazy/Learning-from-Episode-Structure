"""Rare-class recognition metrics — PRIMARY evaluation (Stream 1).

Bottom 8 of 32 classes by Zipf rank (classes 24-31, 0-indexed).
At α=2.5, these collectively account for the smallest share of training samples.
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

RARE_CLASSES = list(range(24, 32))  # bottom 8 by Zipf rank


@torch.no_grad()
def rare_class_f1(model, test_loader, device, rare_classes=None):
    """Compute macro F1 score for rare classes only.

    Args:
        model: Trained PlayEmbedder.
        test_loader: DataLoader for test set.
        device: torch device.
        rare_classes: List of rare class indices (default: 24-31).

    Returns:
        Macro F1 score over rare classes.
    """
    if rare_classes is None:
        rare_classes = RARE_CLASSES

    all_preds, all_labels = [], []
    model.eval()

    for batch in test_loader:
        seq = batch['sequence'].to(device)
        logits, _ = model(seq)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch['label'].tolist())

    # Filter to rare classes only
    rare_mask = [l in rare_classes for l in all_labels]
    rare_preds = [p for p, m in zip(all_preds, rare_mask) if m]
    rare_labels = [l for l, m in zip(all_labels, rare_mask) if m]

    if len(rare_labels) == 0:
        return 0.0

    return f1_score(rare_labels, rare_preds, average='macro',
                    labels=rare_classes, zero_division=0)


@torch.no_grad()
def rare_class_precision_recall(model, test_loader, device, rare_classes=None):
    """Compute precision and recall for rare classes.

    Returns:
        Dict with 'precision' and 'recall' (macro-averaged over rare classes).
    """
    if rare_classes is None:
        rare_classes = RARE_CLASSES

    all_preds, all_labels = [], []
    model.eval()

    for batch in test_loader:
        seq = batch['sequence'].to(device)
        logits, _ = model(seq)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch['label'].tolist())

    rare_mask = [l in rare_classes for l in all_labels]
    rare_preds = [p for p, m in zip(all_preds, rare_mask) if m]
    rare_labels = [l for l, m in zip(all_labels, rare_mask) if m]

    if len(rare_labels) == 0:
        return {'precision': 0.0, 'recall': 0.0}

    return {
        'precision': precision_score(rare_labels, rare_preds, average='macro',
                                     labels=rare_classes, zero_division=0),
        'recall': recall_score(rare_labels, rare_preds, average='macro',
                               labels=rare_classes, zero_division=0),
    }


@torch.no_grad()
def all_class_f1(model, test_loader, device, n_classes: int = 32):
    """Compute macro F1 across all 32 classes."""
    all_preds, all_labels = [], []
    model.eval()

    for batch in test_loader:
        seq = batch['sequence'].to(device)
        logits, _ = model(seq)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch['label'].tolist())

    return f1_score(all_labels, all_preds, average='macro',
                    labels=list(range(n_classes)), zero_division=0)
