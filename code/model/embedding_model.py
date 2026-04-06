"""PlayEmbedder — lightweight 32-class toy embedding network.

Intentionally simple so that performance differences are attributable
to the loss function rather than model capacity. The GRU context encoder
captures sequential structure of each episode.
"""

import torch
import torch.nn as nn


class PlayEmbedder(nn.Module):
    """32-class toy embedding network.

    Input:  sequence of toy indices [B, T=180]
    Output: class logits [B, 32]  +  embedding [B, embed_dim]
    """

    def __init__(self, n_classes: int = 32, embed_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.context = nn.GRU(embed_dim, hidden, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, seq):
        """Forward pass.

        Args:
            seq: [B, T] long tensor of toy indices.

        Returns:
            logits: [B, n_classes] classification logits.
            proj: [B, embed_dim] projected embedding (last hidden state).
        """
        emb = self.embedding(seq)           # [B, T, embed_dim]
        ctx, _ = self.context(emb)          # [B, T, hidden]
        proj = self.proj(ctx[:, -1, :])     # [B, embed_dim] — last hidden state
        logits = self.classifier(proj)      # [B, n_classes]
        return logits, proj

    @torch.no_grad()
    def get_centroids(self, loader, device, n_classes: int = None, embed_dim: int = None):
        """Compute per-class centroids from a data loader.

        Args:
            loader: DataLoader yielding dicts with 'sequence' and 'label'.
            device: torch device.
            n_classes: Override for number of classes.
            embed_dim: Override for embedding dimension.

        Returns:
            Centroids tensor [n_classes, embed_dim].
        """
        nc = n_classes or self.n_classes
        ed = embed_dim or self.embed_dim

        sums = torch.zeros(nc, ed, device=device)
        counts = torch.zeros(nc, device=device)

        self.eval()
        for batch in loader:
            seq = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            _, emb = self.forward(seq)
            for c in range(nc):
                mask = labels == c
                if mask.any():
                    sums[c] += emb[mask].sum(0)
                    counts[c] += mask.sum()

        # Avoid division by zero for classes with no samples
        return sums / counts.clamp(min=1).unsqueeze(1)
