"""Standard cross-entropy loss wrapper with uniform interface.

Used for C1 (uniform baseline) and provides a consistent API
matching AttractorLoss so the trainer doesn't need condition-specific logic.
"""

import torch
import torch.nn as nn


class StandardLoss(nn.Module):
    """Baseline cross-entropy loss with the same forward signature as AttractorLoss."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, embeddings, targets, centroids=None):
        """Compute standard cross-entropy (ignores embeddings/centroids).

        Args:
            logits: [B, n_classes] classification logits.
            embeddings: [B, embed_dim] — unused, for API compatibility.
            targets: [B] class labels.
            centroids: [n_classes, embed_dim] — unused, for API compatibility.

        Returns:
            Scalar loss tensor.
        """
        return self.ce(logits, targets)

    def update_ema_centroids(self, embeddings, targets):
        """No-op for standard loss — centroids unused."""
        pass

    def warm_start_rare(self, rare_classes):
        """No-op for standard loss."""
        pass

    def get_component_losses(self, logits, embeddings, targets, centroids=None):
        """Return component losses for logging (only L_cls for baseline)."""
        return {
            'loss_cls': self.ce(logits, targets).item(),
            'loss_depth': 0.0,
            'loss_attract': 0.0,
            'surprise_value': 1.0,
        }
