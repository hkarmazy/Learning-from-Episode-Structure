"""Main training loop for Attractor Landscape experiments.

Usage:
    python -m training.trainer --condition zipf_sw --seed 0
    python -m training.trainer --condition uniform --seed 0 --config config.yaml
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import PlaySequenceGenerator, zipf_frequencies
from data.transition_graph import build_sw_graph, sw_transition_matrix
from model.embedding_model import PlayEmbedder
from model.attractor_loss import AttractorLoss
from model.loss_standard import StandardLoss
from evaluation.rare_class_metrics import rare_class_f1, rare_class_precision_recall


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_loss(condition: str, cfg: dict, device: torch.device):
    """Build the appropriate loss function for the given condition.

    C1 (uniform): StandardLoss (CE only)
    C2 (zipf):    AttractorLoss with lambda2=0 (no SW coupling)
    C3 (sw):      AttractorLoss with lambda1=0 (no basin depth)
    C4 (zipf_sw): Full AttractorLoss
    """
    if condition == 'uniform':
        return StandardLoss().to(device)

    # Need freq vector and transition matrix for attractor loss
    n_classes = cfg['data']['n_classes']
    alpha = cfg['data']['zipf_alpha']
    sw_beta = cfg['data']['sw_beta']
    acfg = cfg['attractor_loss']

    freq = zipf_frequencies(n_classes, alpha)
    graph = build_sw_graph(n_classes, beta=sw_beta)
    T = sw_transition_matrix(graph)

    if condition == 'zipf':
        return AttractorLoss(
            freq, T,
            alpha_loss=acfg['alpha_loss'],
            beta_surprise=acfg['beta_surprise'],
            lambda1=acfg['lambda1'],
            lambda2=0.0,  # No SW coupling for C2
        ).to(device)
    elif condition == 'sw':
        # Uniform frequencies for C3
        freq_uniform = np.ones(n_classes) / n_classes
        return AttractorLoss(
            freq_uniform, T,
            alpha_loss=acfg['alpha_loss'],
            beta_surprise=acfg['beta_surprise'],
            lambda1=0.0,  # No basin depth for C3
            lambda2=acfg['lambda2'],
        ).to(device)
    elif condition == 'zipf_sw':
        return AttractorLoss(
            freq, T,
            alpha_loss=acfg['alpha_loss'],
            beta_surprise=acfg['beta_surprise'],
            lambda1=acfg['lambda1'],
            lambda2=acfg['lambda2'],
        ).to(device)
    else:
        raise ValueError(f"Unknown condition: {condition}")


def train_one_epoch(model, loader, loss_fn, optimizer, device, centroids):
    """Train for one epoch, return average loss and component losses."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    comp_losses = {'loss_cls': 0, 'loss_depth': 0, 'loss_attract': 0, 'surprise_value': 0}
    n_batches = 0

    for batch in loader:
        seq = batch['sequence'].to(device)
        labels = batch['label'].to(device)

        logits, embeddings = model(seq)
        loss = loss_fn(logits, embeddings, labels, centroids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seq.size(0)
        total_samples += seq.size(0)

        # Log component losses
        with torch.no_grad():
            cl = loss_fn.get_component_losses(logits, embeddings, labels, centroids)
            for k in comp_losses:
                comp_losses[k] += cl[k]
            n_batches += 1

    avg_loss = total_loss / max(total_samples, 1)
    for k in comp_losses:
        comp_losses[k] /= max(n_batches, 1)

    return avg_loss, comp_losses


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int = 32):
    """Evaluate model: return accuracy (all, rare, common) and per-class accuracy."""
    model.eval()
    correct = torch.zeros(n_classes)
    total = torch.zeros(n_classes)

    for batch in loader:
        seq = batch['sequence'].to(device)
        labels = batch['label']
        logits, _ = model(seq)
        preds = logits.argmax(dim=-1).cpu()

        for c in range(n_classes):
            mask = labels == c
            correct[c] += (preds[mask] == c).sum()
            total[c] += mask.sum()

    per_class_acc = (correct / total.clamp(min=1)).numpy()
    all_acc = correct.sum() / total.sum()

    rare_classes = list(range(24, 32))
    common_classes = list(range(0, 8))
    rare_acc = correct[rare_classes].sum() / total[rare_classes].clamp(min=1).sum()
    common_acc = correct[common_classes].sum() / total[common_classes].clamp(min=1).sum()

    return {
        'acc_all': all_acc.item(),
        'acc_rare': rare_acc.item(),
        'acc_common': common_acc.item(),
        'per_class_acc': per_class_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='Attractor Landscape Training')
    parser.add_argument('--condition', type=str, required=True,
                        choices=['uniform', 'zipf', 'sw', 'zipf_sw'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output_dir', type=str, default='runs')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               args.config)
    cfg = load_config(config_path)

    # Setup
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
    focal_boost = dcfg.get('focal_boost', 0.1)
    train_ds = PlaySequenceGenerator(
        args.condition, dcfg['n_train_episodes'],
        n_classes=dcfg['n_classes'], episode_len=dcfg['episode_len'],
        zipf_alpha=dcfg['zipf_alpha'], sw_beta=dcfg['sw_beta'],
        focal_boost=focal_boost, seed=args.seed,
    )
    val_ds = PlaySequenceGenerator(
        args.condition, dcfg['n_val_episodes'],
        n_classes=dcfg['n_classes'], episode_len=dcfg['episode_len'],
        zipf_alpha=dcfg['zipf_alpha'], sw_beta=dcfg['sw_beta'],
        focal_boost=focal_boost, seed=args.seed + 1000,
    )
    test_ds = PlaySequenceGenerator(
        args.condition, dcfg['n_test_episodes'],
        n_classes=dcfg['n_classes'], episode_len=dcfg['episode_len'],
        zipf_alpha=dcfg['zipf_alpha'], sw_beta=dcfg['sw_beta'],
        focal_boost=focal_boost, seed=args.seed + 2000,
    )

    tcfg = cfg['training']
    train_loader = DataLoader(train_ds, batch_size=tcfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=tcfg['batch_size'])

    # Model
    mcfg = cfg['model']
    model = PlayEmbedder(
        n_classes=dcfg['n_classes'],
        embed_dim=mcfg['embed_dim'],
        hidden=mcfg['hidden_dim'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss
    loss_fn = build_loss(args.condition, cfg, device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg['lr'])

    # Logging
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tb'))
    csv_path = os.path.join(run_dir, 'metrics.csv')
    csv_fields = [
        'epoch', 'train_loss', 'train_acc_all', 'train_acc_rare', 'train_acc_common',
        'val_acc_all', 'val_acc_rare', 'val_acc_common',
        'loss_cls', 'loss_depth', 'loss_attract', 'surprise_value',
    ]
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Checkpoint epochs
    checkpoint_epochs = {10, 25, 50, 100}

    # Training loop
    print(f"\nTraining for {tcfg['epochs']} epochs...")
    start_time = time.time()

    for epoch in range(1, tcfg['epochs'] + 1):
        # Compute centroids at start of each epoch
        centroids = model.get_centroids(train_loader, device)

        # Train
        train_loss, comp = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, centroids
        )

        # Evaluate
        train_metrics = evaluate(model, train_loader, device, dcfg['n_classes'])
        val_metrics = evaluate(model, val_loader, device, dcfg['n_classes'])

        # Log to TensorBoard
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

        # Log to CSV
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
        }
        csv_writer.writerow(row)
        csv_file.flush()

        # Progress
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{tcfg['epochs']} | "
                  f"loss={train_loss:.4f} | "
                  f"val_all={val_metrics['acc_all']:.3f} | "
                  f"val_rare={val_metrics['acc_rare']:.3f} | "
                  f"surprise={comp['surprise_value']:.2f} | "
                  f"time={elapsed:.0f}s")

        # Checkpoint
        if epoch in checkpoint_epochs:
            ckpt_path = os.path.join(run_dir, 'checkpoints', f'epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'condition': args.condition,
                'seed': args.seed,
            }, ckpt_path)

    # Final evaluation
    print("\n--- Final Test Evaluation ---")
    test_metrics = evaluate(model, test_loader, device, dcfg['n_classes'])
    test_f1 = rare_class_f1(model, test_loader, device)
    test_pr = rare_class_precision_recall(model, test_loader, device)

    print(f"Test Accuracy (all):    {test_metrics['acc_all']:.4f}")
    print(f"Test Accuracy (rare):   {test_metrics['acc_rare']:.4f}")
    print(f"Test Accuracy (common): {test_metrics['acc_common']:.4f}")
    print(f"Rare-class F1:          {test_f1:.4f}")
    print(f"Rare-class Precision:   {test_pr['precision']:.4f}")
    print(f"Rare-class Recall:      {test_pr['recall']:.4f}")

    # Save final results
    results = {
        'condition': args.condition,
        'seed': args.seed,
        'test_acc_all': test_metrics['acc_all'],
        'test_acc_rare': test_metrics['acc_rare'],
        'test_acc_common': test_metrics['acc_common'],
        'rare_class_f1': test_f1,
        'rare_class_precision': test_pr['precision'],
        'rare_class_recall': test_pr['recall'],
        'per_class_acc': test_metrics['per_class_acc'].tolist(),
    }
    import json
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save final model
    torch.save(model.state_dict(), os.path.join(run_dir, 'model_final.pt'))

    writer.close()
    csv_file.close()
    print(f"\nDone. Results saved to {run_dir}/")


if __name__ == '__main__':
    main()
