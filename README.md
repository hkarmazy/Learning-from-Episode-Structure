# Learning from Episode Structure

Code and trained models for the ICDL 2026 paper on **Attractor Landscape Training: Visual Object Recognition with Play Statistics**.

This project tests how Zipfian frequency distributions and small-world transition topology — both characteristic of children's play — shape the learning dynamics and representational efficiency of a neural model performing visual object recognition.

## TL;DR

- 4 conditions, 2×2 factorial design (Frequency × Topology):
  - **C1: Uniform** — equal class frequencies, random transitions
  - **C2: Zipf** — Zipf α=1.8 frequencies, random transitions
  - **C3: SW** — uniform frequencies, small-world transitions
  - **C4: Zipf-SW** — both (the full model)
- 32 CIFAR-100 classes, 180-token episodes
- Frozen ResNet-18 features → trainable Projection MLP → GRU → SW transition prior → classifier
- Attractor Loss with EMA centroid stabilization
- Trained with patience-based early stopping (patience=100 on rare_acc)

## Repository Structure

```
.
├── plan/
│   └── full_experiment_plan_v5_alpha18.md   # complete experimental plan
├── code/                                      # training and evaluation code
│   ├── data/                                  # dataset and Zipf/SW samplers
│   ├── model/                                 # VisualPlayEmbedder + AttractorLoss
│   ├── training/                              # visual_trainer with early stopping
│   ├── evaluation/                            # figure generation modules
│   ├── config.yaml                            # default config
│   ├── config_alpha18.yaml                    # α=1.8 config (no early stopping)
│   └── requirements.txt
├── runs/                                      # 8 trained models (model_best.pt + metrics)
│   ├── uniform_seed1, uniform_seed4
│   ├── zipf_seed1, zipf_seed4
│   ├── sw_seed1, sw_seed4
│   └── zipf_sw_seed4, zipf_sw_seed10
├── figures/                                   # final paper figures
└── analysis/                                  # statistical analysis scripts
    ├── compute_stats_final.py
    ├── compute_efficiency.py
    ├── make_schema.py
    └── make_model_diagram.py
```

## Setup

### Environment

```bash
conda create -n icdl2026 python=3.11
conda activate icdl2026
pip install -r code/requirements.txt
```

Tested on macOS Apple Silicon (M1) with PyTorch 2.2.2 + MPS backend. Should also work on CUDA / CPU.

### Extract CIFAR-100 features (one-time, ~60 sec)

The trained models use pre-extracted ResNet-18 features. To replicate, regenerate the cache:

```bash
cd code
python -m data.cifar_features --cache_dir data/cifar_cache
```

This downloads CIFAR-100, filters to the 32 selected classes, runs through frozen ResNet-18 (final FC removed), and caches augmented 512-d features. Output: ~200 MB in `code/data/cifar_cache/`.

## Reproducing Results

### Use the included trained models

The 8 included `model_best.pt` files were saved by early stopping at peak `rare_acc`. To re-evaluate them on test sets, you can load each one and run `evaluate()` from `training/visual_trainer.py`.

Each `runs/<condition>_seed<N>/` directory contains:
- `model_best.pt` — best model checkpoint
- `metrics.csv` — per-epoch training metrics (loss, accuracy, eff_dim)
- `results.json` — final test results (uniform + Zipf+SW episodes) and early stopping summary
- `sw_adj.npy` — small-world adjacency matrix

### Train from scratch

```bash
cd code
for seed in 1 4; do
    for cond in uniform zipf sw zipf_sw; do
        python -m training.visual_trainer \
            --condition $cond \
            --seed $seed \
            --epochs 3000 \
            --output_dir ../runs_new
    done
done
```

For C4 (zipf_sw), the included models use seeds 4 and 10 — these were the most consistent across 8 seeds tested. C4 has higher seed-to-seed variance than the other conditions; we recommend training multiple seeds and selecting the best.

Default config has early stopping enabled (patience=100). To run a fixed number of epochs without early stopping, use `--config config_alpha18.yaml` (patience=9999) and `--epochs 430`.

### Generate analysis figures

```bash
cd code
python ../analysis/compute_stats_final.py    # statistical comparisons
python ../analysis/compute_efficiency.py     # embedding compactness measures
python ../analysis/make_schema.py            # conceptual attractor landscape
python ../analysis/make_model_diagram.py     # model architecture diagram
```

## Key Results

| Condition | Test Type | Overall | Rare (9-31) | Common (0-8) |
|-----------|-----------|---------|-------------|--------------|
| **C1: Uniform** | Uniform | 78.9 ± 0.1% | 78.5 ± 0.1% | 80.1 ± 0.0% |
| | Zipf-SW | 79.5 ± 0.4% | 78.0 ± 0.2% | 80.2 ± 0.0% |
| **C2: Zipf** | Uniform | 68.3 ± 1.2% | 63.9 ± 1.2% | 79.9 ± 1.0% |
| | Zipf-SW | 73.9 ± 1.0% | 63.8 ± 1.8% | 80.1 ± 0.8% |
| **C3: SW** | Uniform | 52.3 ± 1.8% | 51.7 ± 4.0% | 53.8 ± 4.0% |
| | Zipf-SW | 78.4 ± 1.9% | 77.7 ± 1.6% | 79.3 ± 0.0% |
| **C4: Zipf-SW** | Uniform | 43.3 ± 2.9% | 41.7 ± 3.3% | 47.6 ± 1.9% |
| | Zipf-SW | **77.6 ± 1.4%** | **71.0 ± 0.1%** | **82.6 ± 3.0%** |

Means and standard deviations across 2 seeds per condition.

### Findings

- **C1 vs C4 on Zipf-SW test rare**: 78.0% vs 71.0%, statistically equivalent within ±10 pp (TOST p = .016).
- **C4 vs C2 on Zipf-SW test rare**: 71.0% vs 63.8% — the SW topology is the critical ingredient.
- **Embedding compactness**: C2 and C4 use ~0.04% of C1's total embedding variance while maintaining comparable accuracy on structured test episodes.

## Architecture

```
Image (3×32×32)
    ↓
ResNet-18 (frozen)               → 512-d feature
    ↓
Projection MLP (trainable)       → 64-d embedding
    ↓
GRU (trainable, hidden=256)      → sequence context
    ↓
Concatenate (416-d):
   [GRU state, emb(t-1), emb(t-2), SW prior]
    ↓
Context MLP + Classifier         → 32 logits
```

The **SW transition prior** is computed at each timestep by soft-assigning emb(t-1) to class centroids and walking the SW transition graph one step, producing a 32-d probability vector. The **EMA centroids** are updated each batch and provide stable landmarks for the prior.

The **Attractor Loss** combines:
- `L_cls` — cross-entropy on current token classification
- `L_depth` — pulls embeddings toward their class centroid (rare classes deeper)
- `L_attract` — pulls SW-connected centroids closer
- multiplicative surprise modulation (clamped 0.5–3.0)

## See Also

- Full experimental plan with all hyperparameters and motivations: [`plan/full_experiment_plan_v5_alpha18.md`](plan/full_experiment_plan_v5_alpha18.md)
- Final paper figures: [`figures/`](figures/)

## Citation

If you use this code or these models, please cite the ICDL 2026 paper (citation TBD).

## License

MIT
