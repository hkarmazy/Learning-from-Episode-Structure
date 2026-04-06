"""VisualPlayEmbedder — current-token classification with SW transition prior.

Classifies the image at position t using:
  - Full sequence history via GRU (hidden state encodes tokens 0..t)
  - Explicit recent context: projected embeddings of t-1 and t-2
  - SW transition prior: a 32-d probability vector computed by soft-assigning
    emb(t-1) to class centroids and walking the SW transition graph one step.
    This gives the classifier an explicit map of the attractor landscape.

v4: The SW prior is the key mechanism that lets the model navigate the
structured embedding space the Attractor Loss builds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualPlayEmbedder(nn.Module):
    """Visual embedding model with SW transition prior.

    Input:  pre-extracted CNN features [B, T=180, 512]
    Output: current-token logits [B, T, 32] + embeddings [B, T, 64]
    """

    def __init__(self, n_classes: int = 32, embed_dim: int = 64,
                 hidden: int = 256, feature_dim: int = 512,
                 projection_hidden: int = 128, context_window: int = 2,
                 sw_prior_temp: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.context_window = context_window
        self.sw_prior_temp = sw_prior_temp

        # Replace nn.Embedding with a projection from CNN features
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_hidden),
            nn.ReLU(),
            nn.Linear(projection_hidden, embed_dim),
        )

        self.context = nn.GRU(embed_dim, hidden, batch_first=True)

        # Input: GRU hidden [hidden] + recent embeddings [W * embed_dim] + SW prior [n_classes]
        combined_dim = hidden + context_window * embed_dim + n_classes
        self.proj = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

        self.classifier = nn.Linear(embed_dim, n_classes)

        # Buffers for SW prior (set by set_sw_prior_data)
        self.register_buffer('_sw_transition', None)
        self.register_buffer('_ema_centroids', None)
        self.register_buffer('_ema_initialized', None)

    def set_sw_prior_data(self, transition_matrix, ema_centroids, ema_initialized):
        """Update the SW prior data from the loss function's state.

        Call before each forward pass during training.

        Args:
            transition_matrix: [n_classes, n_classes] row-stochastic.
            ema_centroids: [n_classes, embed_dim] EMA centroids.
            ema_initialized: [n_classes] bool mask.
        """
        self._sw_transition = transition_matrix
        self._ema_centroids = ema_centroids
        self._ema_initialized = ema_initialized

    def _compute_sw_prior(self, emb_prev):
        """Compute SW transition prior from previous token embeddings.

        Step 1: Soft-assign emb_prev to class centroids (distance softmax)
        Step 2: Walk the SW transition graph one step

        Args:
            emb_prev: [B, T, embed_dim] previous token embeddings (zero-padded at t=0).
        Returns:
            [B, T, n_classes] prior probability distribution.
        """
        if self._sw_transition is None or self._ema_centroids is None:
            # No SW data available — return uniform prior
            B, T, _ = emb_prev.shape
            return torch.ones(B, T, self.n_classes, device=emb_prev.device) / self.n_classes

        centroids = self._ema_centroids  # [C, D]
        B, T, D = emb_prev.shape

        # Flatten to [B*T, D] for distance computation
        emb_flat = emb_prev.reshape(-1, D)  # [B*T, D]

        # Step 1: Distance to each centroid → soft assignment
        dists = torch.cdist(emb_flat, centroids)  # [B*T, C]

        # Mask uninitialized centroids
        if self._ema_initialized is not None:
            mask = (~self._ema_initialized).float() * 1e9
            dists = dists + mask.unsqueeze(0)

        soft_assign = F.softmax(-dists / self.sw_prior_temp, dim=-1)  # [B*T, C]

        # Step 2: One-step SW graph walk
        prior = soft_assign @ self._sw_transition  # [B*T, C]

        return prior.reshape(B, T, self.n_classes)

    def _build_recent_context(self, emb):
        """Gather projected embeddings at t-1, t-2, ... for each position t."""
        B, T, D = emb.shape
        W = self.context_window
        padded = F.pad(emb, (0, 0, W, 0))  # [B, T + W, D]
        parts = []
        for offset in range(1, W + 1):
            parts.append(padded[:, W - offset: T + W - offset, :])
        return torch.cat(parts, dim=-1)  # [B, T, W*D]

    def _get_prev_embedding(self, emb):
        """Get the previous token embedding (t-1) for SW prior, zero-padded at t=0."""
        B, T, D = emb.shape
        padded = F.pad(emb, (0, 0, 1, 0))  # [B, T+1, D]
        return padded[:, :T, :]  # [B, T, D] — emb at t-1 for each t

    def forward(self, features):
        """Forward pass with SW transition prior.

        Args:
            features: [B, T, feature_dim] pre-extracted CNN features.

        Returns:
            logits: [B, T, n_classes] current-token classification logits.
            proj: [B, T, embed_dim] projected embeddings at each timestep.
        """
        emb = self.projection(features)       # [B, T, embed_dim]
        ctx, _ = self.context(emb)            # [B, T, hidden]

        # Recent context (t-1, t-2 embeddings)
        recent = self._build_recent_context(emb)  # [B, T, W*embed_dim]

        # SW transition prior from previous embedding
        emb_prev = self._get_prev_embedding(emb.detach())  # [B, T, embed_dim]
        sw_prior = self._compute_sw_prior(emb_prev)  # [B, T, n_classes]

        # Concatenate: GRU state + recent + SW prior
        combined = torch.cat([ctx, recent, sw_prior], dim=-1)  # [B, T, 416]

        proj = self.proj(combined)            # [B, T, embed_dim]
        logits = self.classifier(proj)        # [B, T, n_classes]
        return logits, proj

    @torch.no_grad()
    def get_centroids(self, loader, device, n_classes: int = None,
                      embed_dim: int = None):
        """Compute per-class centroids from token-level embeddings."""
        nc = n_classes or self.n_classes
        ed = embed_dim or self.embed_dim

        sums = torch.zeros(nc, ed, device=device)
        counts = torch.zeros(nc, device=device)

        self.eval()
        for batch in loader:
            features = batch['features'].to(device)
            class_seq = batch['class_seq'].to(device)

            _, emb = self.forward(features)
            targets = class_seq

            emb_flat = emb.reshape(-1, ed)
            tgt_flat = targets.reshape(-1)

            for c in range(nc):
                mask = tgt_flat == c
                if mask.any():
                    sums[c] += emb_flat[mask].sum(0)
                    counts[c] += mask.sum()

        return sums / counts.clamp(min=1).unsqueeze(1)
