import json, numpy as np, pandas as pd, torch
from scipy import stats
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader
from data.cifar_generator import CIFARPlaySequenceDataset
from model.visual_embedding_model import VisualPlayEmbedder

RUNS = 'runs_alpha18_final'
RARE = list(range(9, 32)); COMMON = list(range(9))
CONDS = [
    ('C1', 'uniform', [1, 4]),
    ('C2', 'zipf', [1, 4]),
    ('C3', 'sw', [1, 4]),
    ('C4', 'zipf_sw', [4, 10]),
]

def get_vals(cond, seeds, test_key):
    a, r, c = [], [], []
    for s in seeds:
        d = json.load(open(f'{RUNS}/{cond}_seed{s}/results.json'))
        t = d[test_key]
        a.append(t['acc_all'])
        r.append(np.array(t['per_class_acc'])[RARE].mean())
        c.append(np.array(t['per_class_acc'])[COMMON].mean())
    return np.array(a), np.array(r), np.array(c)

def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = abs(a.mean() - b.mean()) / (np.sqrt((a.std()**2 + b.std()**2) / 2) + 1e-10)
    return t, p, d

def tost(a, b, margin=0.05):
    diff = a.mean() - b.mean()
    se = np.sqrt(a.std()**2/len(a) + b.std()**2/len(b))
    df = min(len(a), len(b)) - 1
    t_upper = (diff - margin) / (se + 1e-10)
    p_upper = stats.t.cdf(t_upper, df=df)
    t_lower = (diff + margin) / (se + 1e-10)
    p_lower = 1 - stats.t.cdf(t_lower, df=df)
    return max(p_upper, p_lower)

# ===== UNIFORM TEST =====
print('=' * 80)
print('UNIFORM TEST — All Conditions')
print('=' * 80)
for metric_name, idx in [('Overall', 0), ('Common (0-8)', 2), ('Rare (9-31)', 1)]:
    print(f'\n--- {metric_name} ---')
    vals = {}
    for label, cond, seeds in CONDS:
        v = get_vals(cond, seeds, 'test_uniform')[idx]
        vals[label] = v
        print(f'  {label}: {v.mean()*100:.1f} +/- {v.std()*100:.1f}%')
    # Pairwise tests with C1
    for label in ['C2', 'C3', 'C4']:
        t, p, d = welch(vals['C1'], vals[label])
        diff = (vals['C1'].mean() - vals[label].mean()) * 100
        print(f'  C1 vs {label}: diff={diff:+.1f}pp, t={t:.3f}, p={p:.4f}, d={d:.3f}')

# ===== ZIPF+SW TEST =====
print('\n' + '=' * 80)
print('ZIPF+SW TEST — All Conditions')
print('=' * 80)
for metric_name, idx in [('Overall', 0), ('Common (0-8)', 2), ('Rare (9-31)', 1)]:
    print(f'\n--- {metric_name} ---')
    vals = {}
    for label, cond, seeds in CONDS:
        v = get_vals(cond, seeds, 'test_zipfsw')[idx]
        vals[label] = v
        print(f'  {label}: {v.mean()*100:.1f} +/- {v.std()*100:.1f}%')
    for label in ['C2', 'C3', 'C4']:
        t, p, d = welch(vals['C1'], vals[label])
        diff = (vals['C1'].mean() - vals[label].mean()) * 100
        print(f'  C1 vs {label}: diff={diff:+.1f}pp, t={t:.3f}, p={p:.4f}, d={d:.3f}')
    # TOST for C4 rare
    if idx == 1:
        p_tost = tost(vals['C1'], vals['C4'], margin=0.10)
        print(f'  TOST C1 vs C4 rare (margin=10pp): p={p_tost:.4f} ({"equivalent" if p_tost < 0.05 else "not equivalent"})')
        p_tost5 = tost(vals['C1'], vals['C4'], margin=0.05)
        print(f'  TOST C1 vs C4 rare (margin=5pp): p={p_tost5:.4f} ({"equivalent" if p_tost5 < 0.05 else "not equivalent"})')
    # C2 vs C4
    if label == 'C4':
        t24, p24, d24 = welch(vals['C2'], vals['C4'])
        diff24 = (vals['C4'].mean() - vals['C2'].mean()) * 100
        print(f'  C2 vs C4: diff={diff24:+.1f}pp (C4 advantage), t={t24:.3f}, p={p24:.4f}')

# ===== EMBEDDING COMPACTNESS =====
print('\n' + '=' * 80)
print('EMBEDDING COMPACTNESS — All Conditions')
print('=' * 80)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
res = {}
for label, cond, seeds in CONDS:
    res[label] = {'total_var': [], 'rare_spread': [], 'common_spread': [], 'inter_dist': []}
    for s in seeds:
        model = VisualPlayEmbedder(n_classes=32, context_window=2).to(device)
        ckpt = torch.load(f'{RUNS}/{cond}_seed{s}/model_best.pt', map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        ds = CIFARPlaySequenceDataset(cond, 50, n_classes=32, episode_len=180, seed=9999, feature_cache_dir='data/cifar_cache', split='test')
        loader = DataLoader(ds, batch_size=50)
        embs, clss = [], []
        with torch.no_grad():
            for batch in loader:
                _, emb = model(batch['features'].to(device))
                embs.append(emb.cpu().reshape(-1, emb.shape[-1]))
                clss.append(batch['class_seq'].reshape(-1))
        embs = torch.cat(embs).numpy(); clss = torch.cat(clss).numpy()
        res[label]['total_var'].append(np.trace(np.cov(embs.T)))
        rs, cs = [], []
        for c in range(32):
            m = clss == c
            if m.sum() > 1:
                v = embs[m].std(axis=0).mean()
                (rs if c in RARE else cs).append(v)
        res[label]['rare_spread'].append(np.mean(rs))
        res[label]['common_spread'].append(np.mean(cs))
        centroids = np.array([embs[clss == c].mean(axis=0) for c in range(32)])
        res[label]['inter_dist'].append(pdist(centroids).mean())

for name, key, interp in [
    ('Total Embedding Variance', 'total_var', 'lower = more compact'),
    ('Rare Within-Class Spread', 'rare_spread', 'lower = tighter'),
    ('Common Within-Class Spread', 'common_spread', 'lower = tighter'),
    ('Inter-Centroid Distance', 'inter_dist', 'class separation'),
]:
    print(f'\n{name} ({interp}):')
    for label, _, _ in CONDS:
        v = np.array(res[label][key])
        print(f'  {label}: {v.mean():.4f} +/- {v.std():.4f}')
    c1v = np.array(res['C1'][key]); c4v = np.array(res['C4'][key])
    t, p, d = welch(c1v, c4v)
    print(f'  C1 vs C4: ratio={c4v.mean()/c1v.mean():.4f}x, t={t:.3f}, p={p:.4f}')
    c2v = np.array(res['C2'][key])
    t2, p2, _ = welch(c1v, c2v)
    print(f'  C1 vs C2: ratio={c2v.mean()/c1v.mean():.4f}x, t={t2:.3f}, p={p2:.4f}')

# ===== EFF DIM AT BEST EPOCH =====
print('\n' + '=' * 80)
print('EFFECTIVE DIMENSIONALITY AT BEST EPOCH')
print('=' * 80)
for label, cond, seeds in CONDS:
    vals_r, vals_c, vals_a = [], [], []
    for s in seeds:
        r = json.load(open(f'{RUNS}/{cond}_seed{s}/results.json'))
        df = pd.read_csv(f'{RUNS}/{cond}_seed{s}/metrics.csv')
        best_ep = r['early_stopping']['best_epoch']
        row = df[df['epoch'] == best_ep].iloc[0]
        vals_r.append(float(row['eff_dim_rare']))
        vals_c.append(float(row['eff_dim_common']))
        vals_a.append(float(row['eff_dim_all']))
    print(f'{label}: eff_dim_rare={np.mean(vals_r):.2f}+/-{np.std(vals_r):.2f}  '
          f'eff_dim_common={np.mean(vals_c):.2f}+/-{np.std(vals_c):.2f}  '
          f'eff_dim_all={np.mean(vals_a):.2f}+/-{np.std(vals_a):.2f}')
