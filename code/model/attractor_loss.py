"""Attractor Loss — basin depth + SW coupling + surprise modulation.

Four components, each grounded in a natural learning principle:
  L_cls     — baseline supervised signal (cross-entropy)
  L_depth   — rare experiences get deeper encoding (Zipf)
  L_attract — semantically related objects share attractor slopes (SW)
  surprise  — novel events trigger cascading reorganization (multiplicative gate)

Combined: L_total = L_cls + surprise * (λ1·L_depth + λ2·L_attract)

v4: Pure EMA centroid stabilization (no SW-neighbor blending).
EMA centroids serve as stable landmarks for the SW transition prior
computed in the model's forward pass.

v7: Centroid-velocity mass gating. Per-class centroid velocity tracks
"representational mass" — stable centroids (low velocity) resist surprise,
erratic centroids (high velocity) remain plastic. Immune to global
embedding expansion confound that broke v6's variance-based approach.
"""

import torch
import torch.nn as nn


class AttractorLoss(nn.Module):
    """Full attractor landscape loss with EMA centroid tracking.

    Args:
        freq_vec: Normalized frequency vector [n_classes].
        transition_matrix: Row-stochastic transition matrix [n_classes, n_classes].
        alpha_loss: Basin depth power (default 1.8).
        beta_surprise: Surprise exponent (default 1.5).
        lambda1: Basin depth weight (default 0.4).
        lambda2: Neighbor attraction weight (default 0.3).
        ema_decay: EMA decay for surprise and centroid computation (default 0.95).
        sw_edge_threshold: Threshold for SW edge detection (default 0.02).
        embed_dim: Embedding dimension for centroid buffer (default 64).
    """

    def __init__(self, freq_vec, transition_matrix,
                 alpha_loss: float = 1.8, beta_surprise: float = 1.5,
                 lambda1: float = 0.4, lambda2: float = 0.3,
                 ema_decay: float = 0.95, sw_edge_threshold: float = 0.02,
                 embed_dim: int = 64, use_ema: bool = True,
                 velocity_gating: bool = False, vel_ema_decay: float = 0.7,
                 surprise_floor: float = 1.3):
        super().__init__()
        self.register_buffer('freq', torch.tensor(freq_vec, dtype=torch.float32))
        self.register_buffer('T', torch.tensor(transition_matrix, dtype=torch.float32))

        self.alpha = alpha_loss
        self.beta = beta_surprise
        self.lam1 = lambda1
        self.lam2 = lambda2
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.ema = 1.0
        self.ce = nn.CrossEntropyLoss(reduction='none')

        # Centroid-velocity mass gating (v8: floored surprise)
        self.velocity_gating = velocity_gating
        self.vel_ema_decay = vel_ema_decay
        self.surprise_floor = surprise_floor

        n_classes = len(freq_vec)
        self.n_classes = n_classes

        # Precompute SW edge pairs for L_attract
        sw_pairs = (self.T + self.T.T) > sw_edge_threshold
        edges_i, edges_j = [], []
        weights = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                if sw_pairs[i, j]:
                    edges_i.append(i)
                    edges_j.append(j)
                    weights.append((self.T[i, j] + self.T[j, i]).item())
        self.register_buffer('edge_i', torch.tensor(edges_i, dtype=torch.long))
        self.register_buffer('edge_j', torch.tensor(edges_j, dtype=torch.long))
        self.register_buffer('edge_w', torch.tensor(weights, dtype=torch.float32))

        # SW adjacency for warm-start neighbor lookup
        sw_adj = self.T + self.T.T
        sw_adj[sw_adj < sw_edge_threshold] = 0
        sw_adj.fill_diagonal_(0)
        self.register_buffer('sw_adj', sw_adj)

        # EMA centroid buffer
        self.register_buffer('ema_centroids',
                             torch.zeros(n_classes, embed_dim))
        self.register_buffer('_ema_initialized',
                             torch.zeros(n_classes, dtype=torch.bool))

        # Per-class centroid velocity for mass gating (v7)
        self.register_buffer('ema_velocity', torch.zeros(n_classes))
        self.register_buffer('prev_centroids', torch.zeros(n_classes, embed_dim))
        self.velocity_initialized = False

    @torch.no_grad()
    def update_ema_centroids(self, embeddings, targets):
        """Update per-class EMA centroids from batch embeddings.

        Args:
            embeddings: [B, embed_dim] projected embeddings (detached).
            targets: [B] class labels.
        """
        for c in range(self.n_classes):
            mask = (targets == c)
            if mask.sum() == 0:
                continue

            batch_centroid = embeddings[mask].mean(dim=0)

            if not self._ema_initialized[c]:
                self.ema_centroids[c] = batch_centroid
                self._ema_initialized[c] = True
            elif self.use_ema:
                self.ema_centroids[c] = (
                    self.ema_decay * self.ema_centroids[c]
                    + (1.0 - self.ema_decay) * batch_centroid
                )
            else:
                self.ema_centroids[c] = batch_centroid

    @torch.no_grad()
    def update_centroid_velocity(self):
        """Compute per-class centroid velocity and update the EMA velocity buffer.

        Called once per epoch after all batch centroid updates are done.
        Velocity[c] = ||centroid_now - centroid_prev||.
        Low velocity = stable centroid = high mass = anchor.
        High velocity = erratic centroid = low mass = plastic.
        """
        if not self.velocity_initialized:
            self.prev_centroids = self.ema_centroids.detach().clone()
            self.velocity_initialized = True
            return

        displacement = torch.norm(
            self.ema_centroids - self.prev_centroids, dim=1
        )  # [n_classes]

        zero_mask = (self.ema_velocity == 0.0)
        self.ema_velocity = torch.where(
            zero_mask,
            displacement,
            self.vel_ema_decay * self.ema_velocity
            + (1.0 - self.vel_ema_decay) * displacement
        )

        self.prev_centroids = self.ema_centroids.detach().clone()

    def compute_per_class_surprise(self, surprise_global):
        """Gate global surprise by per-class centroid velocity.

        Low velocity (stable centroid, high mass) -> surprise near 1.0.
        High velocity (erratic centroid, low mass) -> full surprise_global.

        Returns: [n_classes] tensor of per-class surprise weights.
        """
        if not self.velocity_gating or not self.velocity_initialized:
            return torch.full(
                (self.n_classes,), surprise_global,
                device=self.ema_velocity.device
            )
        max_vel = self.ema_velocity.max()
        if max_vel < 1e-8:
            return torch.full(
                (self.n_classes,), surprise_global,
                device=self.ema_velocity.device
            )
        vel_norm = self.ema_velocity / (max_vel + 1e-7)
        # v8: floor-to-ceiling interpolation (anchors bend but don't freeze)
        return self.surprise_floor + vel_norm * (surprise_global - self.surprise_floor)

    def warm_start_rare(self, rare_classes):
        """Initialize uninitialized rare-class centroids from SW neighbors.

        Call after epoch 1, when common-class centroids are established.
        """
        with torch.no_grad():
            for c in rare_classes:
                if not self._ema_initialized[c]:
                    weights = self.sw_adj[c]
                    valid = self._ema_initialized & (weights > 0)
                    if valid.any():
                        valid_weights = weights * valid.float()
                        valid_weights = valid_weights / valid_weights.sum()
                        self.ema_centroids[c] = (
                            valid_weights.unsqueeze(1) * self.ema_centroids
                        ).sum(dim=0)
                    self._ema_initialized[c] = True

    def forward(self, logits, embeddings, targets, centroids=None):
        """Compute the full attractor loss using EMA centroids."""
        centroids = self.ema_centroids

        # 1. Base cross-entropy
        L_cls = self.ce(logits, targets)  # [B]

        # 2. Basin depth amplification (Zipf component)
        initialized_mask = self._ema_initialized[targets]
        if initialized_mask.any():
            freq_c = self.freq[targets[initialized_mask]]
            depth = torch.log(1.0 + 1.0 / (freq_c + 1e-7)) ** self.alpha
            diff = embeddings[initialized_mask] - centroids[targets[initialized_mask]]
            L_depth = (depth * diff.pow(2).sum(-1)).mean()
        else:
            L_depth = torch.tensor(0.0, device=logits.device)

        # 3. Neighbor attraction (SW coupling) — vectorized
        L_attract = torch.tensor(0.0, device=logits.device)
        if len(self.edge_i) > 0:
            both_init = self._ema_initialized[self.edge_i] & self._ema_initialized[self.edge_j]
            if both_init.any():
                ei = self.edge_i[both_init]
                ej = self.edge_j[both_init]
                ew = self.edge_w[both_init]
                ci = centroids[ei]
                cj = centroids[ej]
                d2 = (ci - cj).pow(2).sum(-1)
                overlap = torch.exp(-d2 / (2 * 0.5 ** 2))
                L_attract = (ew * d2 * (1.0 - overlap)).mean()

        # 4. Multiplicative surprise modulation
        mean_loss = L_cls.mean().item()
        self.ema = self.ema_decay * self.ema + (1 - self.ema_decay) * mean_loss
        surprise_global = (mean_loss / (self.ema + 1e-7)) ** self.beta
        surprise_global = max(0.5, min(surprise_global, 3.0))

        # v7: Velocity-gated per-class surprise for L_depth
        if self.velocity_gating and initialized_mask.any():
            per_class_surp = self.compute_per_class_surprise(surprise_global)
            per_sample_surp = per_class_surp[targets[initialized_mask]]
            freq_c = self.freq[targets[initialized_mask]]
            depth = torch.log(1.0 + 1.0 / (freq_c + 1e-7)) ** self.alpha
            diff = embeddings[initialized_mask] - centroids[targets[initialized_mask]]
            L_depth_gated = (per_sample_surp * depth * diff.pow(2).sum(-1)).mean()
            return L_cls.mean() + self.lam1 * L_depth_gated + surprise_global * self.lam2 * L_attract
        else:
            return L_cls.mean() + surprise_global * (
                self.lam1 * L_depth + self.lam2 * L_attract
            )

    def get_surprise(self) -> float:
        return self.ema

    def get_component_losses(self, logits, embeddings, targets, centroids=None):
        """Compute individual loss components for logging."""
        centroids = self.ema_centroids
        L_cls = self.ce(logits, targets).mean()

        initialized_mask = self._ema_initialized[targets]
        if initialized_mask.any():
            freq_c = self.freq[targets[initialized_mask]]
            depth = torch.log(1.0 + 1.0 / (freq_c + 1e-7)) ** self.alpha
            diff = embeddings[initialized_mask] - centroids[targets[initialized_mask]]
            L_depth = (depth * diff.pow(2).sum(-1)).mean()
        else:
            L_depth = torch.tensor(0.0, device=logits.device)

        L_attract = torch.tensor(0.0, device=logits.device)
        if len(self.edge_i) > 0:
            both_init = self._ema_initialized[self.edge_i] & self._ema_initialized[self.edge_j]
            if both_init.any():
                ei = self.edge_i[both_init]
                ej = self.edge_j[both_init]
                ew = self.edge_w[both_init]
                ci = centroids[ei]
                cj = centroids[ej]
                d2 = (ci - cj).pow(2).sum(-1)
                overlap = torch.exp(-d2 / (2 * 0.5 ** 2))
                L_attract = (ew * d2 * (1.0 - overlap)).mean()

        mean_loss = L_cls.item()
        surprise = (mean_loss / (self.ema + 1e-7)) ** self.beta

        return {
            'loss_cls': L_cls.item(),
            'loss_depth': L_depth.item(),
            'loss_attract': L_attract.item(),
            'surprise_value': surprise,
        }
