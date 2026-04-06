import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
frozen_color = '#d9d9d9'
trainable_color = '#4C72B0'
loss_color = '#C44E52'
arrow_color = '#333333'

def box(x, y, w, h, label, color, fontsize=10, sublabel=None, alpha=0.85):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                           facecolor=color, edgecolor='#333333', linewidth=1.2, alpha=alpha)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h*0.6, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white' if color != frozen_color else '#333333')
        ax.text(x + w/2, y + h*0.28, sublabel, ha='center', va='center',
                fontsize=fontsize - 2, color='white' if color != frozen_color else '#555555', style='italic')
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white' if color != frozen_color else '#333333')

def arrow(x1, y1, x2, y2, label=None, color=arrow_color):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=8, color='#555555', style='italic', va='center')

# ===== ROW 1: Feature extraction (frozen) =====
box(0.5, 8.2, 2.5, 1.2, 'CIFAR-100\nImages', frozen_color, fontsize=10, sublabel='32×32 RGB')
arrow(3.0, 8.8, 4.0, 8.8, '[B,T,3,32,32]')
box(4.0, 8.2, 3.0, 1.2, 'ResNet-18', frozen_color, fontsize=11, sublabel='frozen, pretrained')
arrow(7.0, 8.8, 8.0, 8.8, '[B,T,512]')
box(8.0, 8.2, 2.5, 1.2, 'Feature\nVectors', frozen_color, fontsize=10, sublabel='512-d cached')

# Frozen label
ax.text(5.5, 9.6, 'Phase 1: Feature Extraction (frozen, one-time)', fontsize=11,
        fontweight='bold', ha='center', color='#666666')

# ===== ROW 2: Trainable projection =====
arrow(9.25, 8.2, 9.25, 7.2)
box(7.5, 5.8, 3.5, 1.2, 'Projection MLP', trainable_color, fontsize=11,
    sublabel='512 → 128 → ReLU → 64')
ax.text(9.25, 7.5, '[B,T,512]', fontsize=8, color='#555555', style='italic', ha='center')

# ===== ROW 3: Sequential processing =====
arrow(9.25, 5.8, 9.25, 5.0)
ax.text(9.25, 5.3, '[B,T,64]', fontsize=8, color='#555555', style='italic', ha='center')

# GRU
box(7.5, 3.6, 3.5, 1.2, 'GRU', trainable_color, fontsize=12,
    sublabel='64 → 256 hidden')

# ===== Context assembly (wide box) =====
arrow(9.25, 3.6, 9.25, 2.8)

# Show the three inputs merging
box(2.0, 1.4, 12.0, 1.2, 'Context Assembly', trainable_color, fontsize=11,
    sublabel='GRU[256] + emb(t-1)[64] + emb(t-2)[64] + SW Prior[32] → 416-d')

# Recent context arrow from projection
arrow(7.5, 6.4, 3.5, 6.4)
box(1.0, 5.8, 2.5, 1.2, 'Recent\nContext', trainable_color, fontsize=9,
    sublabel='emb(t-1), emb(t-2)')
arrow(2.25, 5.8, 2.25, 2.6)

# SW Prior
box(12.0, 5.8, 2.5, 1.2, 'SW Transition\nPrior', trainable_color, fontsize=9,
    sublabel='32-d probability')
arrow(11.0, 6.4, 12.0, 6.4)
arrow(13.25, 5.8, 13.25, 2.6)

# EMA Centroids feeding SW Prior
box(12.0, 7.5, 2.5, 0.9, 'EMA\nCentroids', loss_color, fontsize=9)
arrow(13.25, 7.5, 13.25, 7.0, '[32,64]')

# ===== ROW 4: Classifier =====
arrow(8.0, 1.4, 8.0, 0.6)
box(6.0, -0.2, 4.0, 0.7, 'Classifier  [64 → 32]', trainable_color, fontsize=10)

# Output
arrow(8.0, -0.2, 8.0, -0.8)
ax.text(8.0, -1.0, 'Class prediction at position t', fontsize=10,
        ha='center', fontweight='bold', color='#333333')

# ===== Attractor Loss (side) =====
box(0.3, 2.8, 2.8, 1.8, 'Attractor\nLoss', loss_color, fontsize=11,
    sublabel='L_cls + surprise×\n(λ₁·L_depth + λ₂·L_attract)')
arrow(3.1, 3.7, 6.0, 2.0, color=loss_color)
ax.text(4.2, 3.1, 'gradients', fontsize=8, color=loss_color, style='italic', rotation=-25)

# Phase 2 label
ax.text(8.0, 10.0, 'Phase 2: Trainable Model (per condition)', fontsize=12,
        fontweight='bold', ha='center', color=trainable_color)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=frozen_color, edgecolor='#333', label='Frozen (pretrained)'),
    mpatches.Patch(facecolor=trainable_color, edgecolor='#333', label='Trainable'),
    mpatches.Patch(facecolor=loss_color, edgecolor='#333', label='Loss / EMA state'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9,
          bbox_to_anchor=(1.0, 0.98))

fig.savefig('figures_v5es/fig_model_architecture.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved figures_v5es/fig_model_architecture.png')
