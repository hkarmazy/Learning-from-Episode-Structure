"""Visual training loop — v5 full experiment with grokking metrics.

Current-token classification with SW transition prior, EMA centroids,
effective dimensionality tracking, and per-class learning onset detection.

Usage:
    python -m training.visual_trainer --condition zipf_sw --seed 0
    python -m training.visual_trainer --condition zipf_sw --seed 0 --epochs 400
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cifar_generator import CIFARPlaySequenceDataset
from data.cifar_class_selection import CIFAR100_CLASS_NAMES
from data.synthetic_generator import zipf_frequencies
from data.transition_graph import build_sw_graph, sw_transition_matrix
from model.visual_embedding_model import VisualPlayEmbedder
from model.attractor_loss import AttractorLoss
from model.loss_standard import StandardLoss


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def effective_dimensionality(embeddings):
    """Compute effective dimensionality via PCA participation ratio.

    (sum of eigenvalues)^2 / sum(eigenvalues^2)
    High = rote/spread encoding, Low = compressed/schema encoding.
    """
    if embeddings is None or len(embeddings) < 4:
        return float('nan')

    emb = embeddings.float()
    emb = emb - emb.mean(dim=0, keepdim=True)

    try:
        _, S, _ = torch.linalg.svd(emb, full_matrices=False)
    except RuntimeError:
        try:
            _, S, _ = torch.linalg.svd(emb.cpu(), full_matrices=False)
        except Exception:
            return float('nan')

    eigenvalues = (S ** 2) / max(len(emb) - 1, 1)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return float('nan')

    sum_sq = (eigenvalues ** 2).sum()
    if sum_sq < 1e-10:
        return float(len(eigenvalues))

    return (eigenvalues.sum() ** 2 / sum_sq).item()


def build_loss(condition: str, cfg: dict, device: torch.device, use_ema: bool = True):
    """Build loss function with EMA centroid support."""
    if condition == 'uniform':
        return StandardLoss().to(device), None

    n_classes = cfg['data']['n_classes']
    alpha = cfg['data']['zipf_alpha']
    sw_beta = cfg['data']['sw_beta']
    acfg = cfg['attractor_loss']
    embed_dim = cfg['model']['embed_dim']

    freq = zipf_frequencies(n_classes, alpha)
    graph = build_sw_graph(n_classes, beta=sw_beta)
    T = sw_transition_matrix(graph)

    common_kwargs = dict(
        alpha_loss=acfg['alpha_loss'],
        beta_surprise=acfg['beta_surprise'],
        ema_decay=acfg['ema_decay'],
        embed_dim=embed_dim,
        use_ema=use_ema,
        velocity_gating=acfg.get('velocity_gating', False),
        vel_ema_decay=acfg.get('vel_ema_decay', 0.7),
        surprise_floor=acfg.get('surprise_floor', 1.3),
    )

    if condition == 'zipf':
        loss = AttractorLoss(
            freq, T, lambda1=acfg['lambda1'], lambda2=0.0,
            **common_kwargs,
        ).to(device)
    elif condition == 'sw':
        freq_uniform = np.ones(n_classes) / n_classes
        loss = AttractorLoss(
            freq_uniform, T, lambda1=0.0, lambda2=acfg['lambda2'],
            **common_kwargs,
        ).to(device)
    elif condition == 'zipf_sw':
        loss = AttractorLoss(
            freq, T, lambda1=acfg['lambda1'], lambda2=acfg['lambda2'],
            **common_kwargs,
        ).to(device)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    T_tensor = torch.tensor(T, dtype=torch.float32, device=device)
    return loss, T_tensor


def _sync_prior_data(model, loss_fn, T_tensor):
    """Push EMA centroid state from loss to model for SW prior computation."""
    if T_tensor is not None and hasattr(loss_fn, 'ema_centroids'):
        model.set_sw_prior_data(T_tensor, loss_fn.ema_centroids, loss_fn._ema_initialized)


def train_one_epoch(model, loader, loss_fn, optimizer, device, T_tensor):
    """Train for one epoch with per-batch EMA centroids and SW prior."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    comp_losses = {'loss_cls': 0, 'loss_depth': 0, 'loss_attract': 0, 'surprise_value': 0}
    n_batches = 0

    for batch in loader:
        features = batch['features'].to(device)
        class_seq = batch['class_seq'].to(device)

        _sync_prior_data(model, loss_fn, T_tensor)

        logits, embeddings = model(features)
        targets = class_seq

        B, T, C = logits.shape
        logits_flat = logits.reshape(-1, C)
        emb_flat = embeddings.reshape(-1, embeddings.shape[-1])
        tgt_flat = targets.reshape(-1)

        loss_fn.update_ema_centroids(emb_flat.detach(), tgt_flat)

        loss = loss_fn(logits_flat, emb_flat, tgt_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B * T
        total_tokens += B * T

        with torch.no_grad():
            cl = loss_fn.get_component_losses(logits_flat, emb_flat, tgt_flat)
            for k in comp_losses:
                comp_losses[k] += cl[k]
            n_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    for k in comp_losses:
        comp_losses[k] /= max(n_batches, 1)

    return avg_loss, comp_losses


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int = 32, **kwargs):
    """Evaluate current-token classification accuracy."""
    model.eval()
    correct = torch.zeros(n_classes)
    total = torch.zeros(n_classes)

    for batch in loader:
        features = batch['features'].to(device)
        class_seq = batch['class_seq'].to(device)

        logits, _ = model(features)
        targets = class_seq
        preds = logits.argmax(dim=-1).cpu()
        targets_cpu = targets.cpu()

        for c in range(n_classes):
            mask = targets_cpu == c
            correct[c] += (preds[mask] == c).sum()
            total[c] += mask.sum()

    per_class_acc = (correct / total.clamp(min=1)).numpy()
    all_acc = correct.sum() / total.sum()

    rt = kwargs.get('rare_threshold', 16)
    rare_classes = list(range(rt, n_classes))
    common_classes = list(range(rt))
    rare_acc = correct[rare_classes].sum() / total[rare_classes].clamp(min=1).sum()
    common_acc = correct[common_classes].sum() / total[common_classes].clamp(min=1).sum()

    return {
        'acc_all': all_acc.item(),
        'acc_rare': rare_acc.item(),
        'acc_common': common_acc.item(),
        'per_class_acc': per_class_acc,
    }


@torch.no_grad()
def evaluate_with_embeddings(model, loader, device, n_classes: int = 32, **kwargs):
    """Evaluate accuracy AND collect embeddings for effective dimensionality."""
    model.eval()
    correct = torch.zeros(n_classes)
    total = torch.zeros(n_classes)
    all_embs = []
    all_labs = []

    for batch in loader:
        features = batch['features'].to(device)
        class_seq = batch['class_seq'].to(device)

        logits, embeddings = model(features)
        targets = class_seq
        preds = logits.argmax(dim=-1).cpu()
        targets_cpu = targets.cpu()

        for c in range(n_classes):
            mask = targets_cpu == c
            correct[c] += (preds[mask] == c).sum()
            total[c] += mask.sum()

        # Collect embeddings on CPU
        all_embs.append(embeddings.cpu().reshape(-1, embeddings.shape[-1]))
        all_labs.append(targets_cpu.reshape(-1))

    per_class_acc = (correct / total.clamp(min=1)).numpy()
    all_acc = correct.sum() / total.sum()

    rt = kwargs.get('rare_threshold', 16)
    rare_classes = list(range(rt, n_classes))
    common_classes = list(range(rt))
    rare_acc = correct[rare_classes].sum() / total[rare_classes].clamp(min=1).sum()
    common_acc = correct[common_classes].sum() / total[common_classes].clamp(min=1).sum()

    # Compute effective dimensionality
    all_embs_t = torch.cat(all_embs)
    all_labs_t = torch.cat(all_labs)
    rt = kwargs.get('rare_threshold', 16)
    rare_mask = all_labs_t >= rt
    common_mask = all_labs_t < rt

    eff_dim_rare = effective_dimensionality(all_embs_t[rare_mask])
    eff_dim_common = effective_dimensionality(all_embs_t[common_mask])
    eff_dim_all = effective_dimensionality(all_embs_t)

    return {
        'acc_all': all_acc.item(),
        'acc_rare': rare_acc.item(),
        'acc_common': common_acc.item(),
        'per_class_acc': per_class_acc,
        'eff_dim_rare': eff_dim_rare,
        'eff_dim_common': eff_dim_common,
        'eff_dim_all': eff_dim_all,
    }


def main():
    parser = argparse.ArgumentParser(description='Visual Attractor Landscape Training v5')
    parser.add_argument('--condition', type=str, required=True,
                        choices=['uniform', 'zipf', 'sw', 'zipf_sw'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output_dir', type=str, default='runs_visual')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override training epochs from config')
    parser.add_argument('--disable_ema', action='store_true', default=False,
                        help='Disable EMA centroid stabilization (use batch centroids)')
    parser.add_argument('--zipf_alpha', type=float, default=None,
                        help='Override Zipf alpha from config')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               args.config)
    cfg = load_config(config_path)
    if args.epochs is not None:
        cfg['training']['epochs'] = args.epochs
    if args.zipf_alpha is not None:
        cfg['data']['zipf_alpha'] = args.zipf_alpha
        print(f"[Config] Zipf alpha overridden to {args.zipf_alpha}")

    set_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Condition: {args.condition}, Seed: {args.seed}")

    # Output directory
    run_name = f"{args.condition}_seed{args.seed}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)

    # Data
    dcfg = cfg['data']
    n_classes = dcfg['n_classes']
    cache_dir = dcfg.get('feature_cache_dir', 'data/cifar_cache')

    train_ds = CIFARPlaySequenceDataset(
        args.condition, dcfg['n_train_episodes'],
        n_classes=n_classes, episode_len=dcfg['episode_len'],
        zipf_alpha=dcfg['zipf_alpha'], sw_beta=dcfg['sw_beta'],
        seed=args.seed, feature_cache_dir=cache_dir, split='train',
    )
    val_ds = CIFARPlaySequenceDataset(
        args.condition, dcfg['n_val_episodes'],
        n_classes=n_classes, episode_len=dcfg['episode_len'],
        zipf_alpha=dcfg['zipf_alpha'], sw_beta=dcfg['sw_beta'],
        seed=args.seed + 1000, feature_cache_dir=cache_dir, split='test',
    )
    test_ds = CIFARPlaySequenceDataset(
        args.condition, dcfg['n_test_episodes'],
        n_classes=n_classes, episode_len=dcfg['episode_len'],
        zipf_alpha=dcfg['zipf_alpha'], sw_beta=dcfg['sw_beta'],
        seed=args.seed + 2000, feature_cache_dir=cache_dir, split='test',
    )

    tcfg = cfg['training']
    train_loader = DataLoader(train_ds, batch_size=tcfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=tcfg['batch_size'])

    # Model
    mcfg = cfg['model']
    acfg = cfg['attractor_loss']
    model = VisualPlayEmbedder(
        n_classes=n_classes,
        embed_dim=mcfg['embed_dim'],
        hidden=mcfg['hidden_dim'],
        feature_dim=mcfg.get('feature_dim', 512),
        projection_hidden=mcfg.get('projection_hidden', 128),
        context_window=mcfg.get('context_window', 2),
        sw_prior_temp=acfg.get('sw_prior_temp', 1.0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss + transition matrix for SW prior
    use_ema = not args.disable_ema
    loss_fn, T_tensor = build_loss(args.condition, cfg, device, use_ema=use_ema)
    ema_status = "ENABLED" if use_ema else "DISABLED (batch centroids)"
    print(f"[Config] EMA centroid stabilization: {ema_status}")

    # Save SW adjacency for onset correlation figure
    if T_tensor is not None:
        sw_adj_path = os.path.join(run_dir, 'sw_adj.npy')
        np.save(sw_adj_path, T_tensor.cpu().numpy())

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg['lr'])

    # Logging
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tb'))
    csv_path = os.path.join(run_dir, 'metrics.csv')
    csv_fields = [
        'epoch', 'train_loss', 'train_acc_all', 'train_acc_rare', 'train_acc_common',
        'val_acc_all', 'val_acc_rare', 'val_acc_common',
        'loss_cls', 'loss_depth', 'loss_attract', 'surprise_value',
        'eff_dim_rare', 'eff_dim_common', 'eff_dim_all',
        'mean_vel_common', 'mean_vel_rare', 'vel_ratio',
    ]
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    checkpoint_epochs = set(cfg.get('checkpoints', {}).get('save_epochs', [50, 100, 200, 300, 400]))

    # Rare/common split from config
    rare_threshold = cfg.get('evaluation', {}).get('rare_rank_threshold', 16)
    rare_classes = list(range(rare_threshold, n_classes))
    print(f"[Config] Rare threshold: {rare_threshold} (common=0-{rare_threshold-1}, rare={rare_threshold}-{n_classes-1})")

    # Per-class learning onset tracking
    ecfg = cfg.get('evaluation', {})
    onset_threshold = ecfg.get('onset_threshold', 0.40)
    class_onset_epoch = {}
    class_names = CIFAR100_CLASS_NAMES

    # Early stopping setup
    es_cfg = cfg.get('early_stopping', {})
    es_patience = es_cfg.get('patience', 100)
    es_min_delta = es_cfg.get('min_delta', 0.002)
    es_metric = es_cfg.get('metric', 'rare_acc')
    best_es_value = -float('inf')
    best_es_epoch = 0
    es_no_improve = 0
    best_model_path = os.path.join(run_dir, 'model_best.pt')
    print(f"[EarlyStopping] metric={es_metric}  patience={es_patience}  min_delta={es_min_delta}")

    print(f"\nTraining for up to {tcfg['epochs']} epochs (current-token + SW prior, visual)...")
    start_time = time.time()

    for epoch in range(1, tcfg['epochs'] + 1):
        train_loss, comp = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, T_tensor
        )

        # SW warm-start: after epoch 1, initialize uninitialized rare centroids
        if epoch == 1:
            loss_fn.warm_start_rare(rare_classes)

        # Sync prior data for evaluation
        _sync_prior_data(model, loss_fn, T_tensor)

        train_metrics = evaluate(model, train_loader, device, n_classes, rare_threshold=rare_threshold)
        val_metrics = evaluate_with_embeddings(model, val_loader, device, n_classes, rare_threshold=rare_threshold)

        # Per-class onset tracking
        for c in range(n_classes):
            if c not in class_onset_epoch:
                if val_metrics['per_class_acc'][c] >= onset_threshold:
                    class_onset_epoch[c] = epoch
                    if epoch % 5 == 0 or epoch <= 10:
                        print(f"  [Onset] Class {c} ({class_names[c]}) "
                              f"first reached {onset_threshold:.0%} at epoch {epoch}")

        # v7: Update centroid velocity for mass gating
        mean_vel_common = 0.0
        mean_vel_rare = 0.0
        vel_ratio = 1.0
        if hasattr(loss_fn, 'update_centroid_velocity'):
            loss_fn.update_centroid_velocity()
            ema_vel_cpu = loss_fn.ema_velocity.cpu()
            mean_vel_common = ema_vel_cpu[:16].mean().item()
            mean_vel_rare = ema_vel_cpu[16:].mean().item()
            vel_ratio = mean_vel_rare / (mean_vel_common + 1e-8)

        # TensorBoard
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/cls', comp['loss_cls'], epoch)
        writer.add_scalar('loss/depth', comp['loss_depth'], epoch)
        writer.add_scalar('loss/attract', comp['loss_attract'], epoch)
        writer.add_scalar('loss/surprise', comp['surprise_value'], epoch)
        writer.add_scalar('acc/train_all', train_metrics['acc_all'], epoch)
        writer.add_scalar('acc/train_rare', train_metrics['acc_rare'], epoch)
        writer.add_scalar('acc/val_all', val_metrics['acc_all'], epoch)
        writer.add_scalar('acc/val_rare', val_metrics['acc_rare'], epoch)
        writer.add_scalar('acc/val_common', val_metrics['acc_common'], epoch)
        writer.add_scalar('val/eff_dim_rare', val_metrics['eff_dim_rare'], epoch)
        writer.add_scalar('val/eff_dim_common', val_metrics['eff_dim_common'], epoch)
        writer.add_scalar('val/eff_dim_all', val_metrics['eff_dim_all'], epoch)
        writer.add_scalar('velocity/mean_vel_common', mean_vel_common, epoch)
        writer.add_scalar('velocity/mean_vel_rare', mean_vel_rare, epoch)
        writer.add_scalar('velocity/vel_ratio', vel_ratio, epoch)

        # CSV
        row = {
            'epoch': epoch,
            'train_loss': f"{train_loss:.4f}",
            'train_acc_all': f"{train_metrics['acc_all']:.4f}",
            'train_acc_rare': f"{train_metrics['acc_rare']:.4f}",
            'train_acc_common': f"{train_metrics['acc_common']:.4f}",
            'val_acc_all': f"{val_metrics['acc_all']:.4f}",
            'val_acc_rare': f"{val_metrics['acc_rare']:.4f}",
            'val_acc_common': f"{val_metrics['acc_common']:.4f}",
            'loss_cls': f"{comp['loss_cls']:.4f}",
            'loss_depth': f"{comp['loss_depth']:.4f}",
            'loss_attract': f"{comp['loss_attract']:.4f}",
            'surprise_value': f"{comp['surprise_value']:.4f}",
            'eff_dim_rare': f"{val_metrics['eff_dim_rare']:.4f}",
            'eff_dim_common': f"{val_metrics['eff_dim_common']:.4f}",
            'eff_dim_all': f"{val_metrics['eff_dim_all']:.4f}",
            'mean_vel_common': f"{mean_vel_common:.6f}",
            'mean_vel_rare': f"{mean_vel_rare:.6f}",
            'vel_ratio': f"{vel_ratio:.4f}",
        }
        csv_writer.writerow(row)
        csv_file.flush()

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{tcfg['epochs']} | "
                  f"loss={train_loss:.4f} | "
                  f"val_all={val_metrics['acc_all']:.3f} | "
                  f"val_rare={val_metrics['acc_rare']:.3f} | "
                  f"eff_dim_rare={val_metrics['eff_dim_rare']:.1f} | "
                  f"surprise={comp['surprise_value']:.2f} | "
                  f"time={elapsed:.0f}s")

        if epoch in checkpoint_epochs:
            ckpt_path = os.path.join(run_dir, 'checkpoints', f'model_ep{epoch}.pt')
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'condition': args.condition,
                'seed': args.seed,
                'metrics': {
                    'val_acc': val_metrics['acc_all'],
                    'rare_acc': val_metrics['acc_rare'],
                    'eff_dim_rare': val_metrics['eff_dim_rare'],
                    'eff_dim_all': val_metrics['eff_dim_all'],
                },
            }
            if hasattr(loss_fn, 'ema_centroids'):
                ckpt_data['ema_centroids'] = loss_fn.ema_centroids.cpu()
                ckpt_data['centroid_initialized'] = loss_fn._ema_initialized.cpu()
            if hasattr(loss_fn, 'ema_velocity'):
                ckpt_data['ema_velocity'] = loss_fn.ema_velocity.cpu()
                ckpt_data['prev_centroids'] = loss_fn.prev_centroids.cpu()
                ckpt_data['velocity_initialized'] = loss_fn.velocity_initialized
            torch.save(ckpt_data, ckpt_path)

        # Early stopping check
        es_current = val_metrics['acc_rare']
        if es_current > best_es_value + es_min_delta:
            best_es_value = es_current
            best_es_epoch = epoch
            es_no_improve = 0
            # Save best model
            best_ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'condition': args.condition,
                'seed': args.seed,
            }
            if hasattr(loss_fn, 'ema_centroids'):
                best_ckpt['ema_centroids'] = loss_fn.ema_centroids.cpu()
                best_ckpt['centroid_initialized'] = loss_fn._ema_initialized.cpu()
            torch.save(best_ckpt, best_model_path)
            if epoch % 5 == 0 or epoch <= 10:
                print(f"  [EarlyStopping] New best {es_metric}={es_current:.4f} at epoch {epoch}")
        else:
            es_no_improve += 1
            if es_no_improve >= es_patience:
                print(f"\n{'='*60}")
                print(f"[EarlyStopping] STOPPING at epoch {epoch}")
                print(f"  Best {es_metric}: {best_es_value:.4f} at epoch {best_es_epoch}")
                print(f"{'='*60}\n")
                break

    # Load best model for final evaluation
    if os.path.exists(best_model_path):
        best_ckpt = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt['model_state_dict'], strict=False)
        print(f"Loaded best model from epoch {best_es_epoch} for final evaluation")

    # Final test evaluation — dual test: uniform + zipf_sw episodes
    print("\n--- Final Test Evaluation ---")
    _sync_prior_data(model, loss_fn, T_tensor)

    test_metrics = evaluate(model, test_loader, device, n_classes, rare_threshold=rare_threshold)
    print(f"Test (own condition) — all: {test_metrics['acc_all']:.4f}  "
          f"rare: {test_metrics['acc_rare']:.4f}  common: {test_metrics['acc_common']:.4f}")

    test_ds_uniform = CIFARPlaySequenceDataset(
        'uniform', dcfg['n_test_episodes'],
        n_classes=n_classes, episode_len=dcfg['episode_len'],
        zipf_alpha=dcfg['zipf_alpha'], sw_beta=dcfg['sw_beta'],
        seed=args.seed + 3000, feature_cache_dir=cache_dir, split='test',
    )
    test_loader_uniform = DataLoader(test_ds_uniform, batch_size=tcfg['batch_size'])
    test_uniform = evaluate(model, test_loader_uniform, device, n_classes, rare_threshold=rare_threshold)
    print(f"Test (uniform eps)   — all: {test_uniform['acc_all']:.4f}  "
          f"rare: {test_uniform['acc_rare']:.4f}  common: {test_uniform['acc_common']:.4f}")

    test_ds_zipfsw = CIFARPlaySequenceDataset(
        'zipf_sw', dcfg['n_test_episodes'],
        n_classes=n_classes, episode_len=dcfg['episode_len'],
        zipf_alpha=dcfg['zipf_alpha'], sw_beta=dcfg['sw_beta'],
        seed=args.seed + 4000, feature_cache_dir=cache_dir, split='test',
    )
    test_loader_zipfsw = DataLoader(test_ds_zipfsw, batch_size=tcfg['batch_size'])
    test_zipfsw = evaluate(model, test_loader_zipfsw, device, n_classes, rare_threshold=rare_threshold)
    print(f"Test (zipf+sw eps)   — all: {test_zipfsw['acc_all']:.4f}  "
          f"rare: {test_zipfsw['acc_rare']:.4f}  common: {test_zipfsw['acc_common']:.4f}")

    results = {
        'condition': args.condition,
        'seed': args.seed,
        'test_acc_all': test_metrics['acc_all'],
        'test_acc_rare': test_metrics['acc_rare'],
        'test_acc_common': test_metrics['acc_common'],
        'per_class_acc': test_metrics['per_class_acc'].tolist(),
        'test_uniform': {
            'acc_all': test_uniform['acc_all'],
            'acc_rare': test_uniform['acc_rare'],
            'acc_common': test_uniform['acc_common'],
            'per_class_acc': test_uniform['per_class_acc'].tolist(),
        },
        'test_zipfsw': {
            'acc_all': test_zipfsw['acc_all'],
            'acc_rare': test_zipfsw['acc_rare'],
            'acc_common': test_zipfsw['acc_common'],
            'per_class_acc': test_zipfsw['per_class_acc'].tolist(),
        },
        'class_onset_epochs': {str(k): v for k, v in class_onset_epoch.items()},
        'per_class_acc_final': val_metrics['per_class_acc'].tolist(),
        'early_stopping': {
            'stopped_epoch': epoch,
            'best_epoch': best_es_epoch,
            'best_rare_acc': best_es_value,
            'patience_used': es_patience,
            'stopped_early': (es_no_improve >= es_patience),
        },
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(run_dir, 'model_final.pt'))

    writer.close()
    csv_file.close()
    print(f"\nDone. Results saved to {run_dir}/")


if __name__ == '__main__':
    main()
