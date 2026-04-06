import json, numpy as np, pandas as pd, torch
from scipy import stats
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader
from data.cifar_generator import CIFARPlaySequenceDataset
from model.visual_embedding_model import VisualPlayEmbedder

RUNS = 'runs_visual_v5es'
RARE = list(range(16, 32))
C1_SEEDS = [0, 1, 2, 3, 4]
C4_SEEDS = [1, 4]

print('=' * 75)
print('1. EFFECTIVE DIMENSIONALITY AT BEST EPOCH')
print('=' * 75)

c1_eff = {'rare': [], 'common': [], 'all': []}
c4_eff = {'rare': [], 'common': [], 'all': []}

for cond, seeds, store in [('uniform', C1_SEEDS, c1_eff), ('zipf_sw', C4_SEEDS, c4_eff)]:
    for s in seeds:
        r = json.load(open(f'{RUNS}/{cond}_seed{s}/results.json'))
        df = pd.read_csv(f'{RUNS}/{cond}_seed{s}/metrics.csv')
        best_ep = r['early_stopping']['best_epoch']
        row = df[df['epoch'] == best_ep].iloc[0]
        store['rare'].append(float(row['eff_dim_rare']))
        store['common'].append(float(row['eff_dim_common']))
        store['all'].append(float(row['eff_dim_all']))

for name, key in [('All classes', 'all'), ('Common (0-15)', 'common'), ('Rare (16-31)', 'rare')]:
    c1 = np.array(c1_eff[key]); c4 = np.array(c4_eff[key])
    t_stat, p_val = stats.ttest_ind(c1, c4, equal_var=False)
    d = abs(c1.mean() - c4.mean()) / (np.sqrt((c1.std()**2 + c4.std()**2) / 2) + 1e-10)
    higher = "C1 higher" if c1.mean() > c4.mean() else "C4 higher"
    print(f'\n{name}:')
    print(f'  C1: {c1.mean():.2f} +/- {c1.std():.2f}  C4: {c4.mean():.2f} +/- {c4.std():.2f}')
    print(f'  {higher} by {abs(c1.mean() - c4.mean()):.2f} dims')
    print(f'  Welch t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}')

print('\n' + '=' * 75)
print('2. EMBEDDING SPACE COMPACTNESS')
print('=' * 75)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

res = {}
for label, pairs in [('C1', [('uniform', s) for s in C1_SEEDS]), ('C4', [('zipf_sw', s) for s in C4_SEEDS])]:
    res[label] = {'total_var': [], 'rare_spread': [], 'common_spread': [], 'inter_dist': [], 'acc_per_dim': []}
    for condition, s in pairs:
        model = VisualPlayEmbedder(n_classes=32, context_window=2).to(device)
        ckpt = torch.load(f'{RUNS}/{condition}_seed{s}/model_best.pt', map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        ds = CIFARPlaySequenceDataset(condition, 50, n_classes=32, episode_len=180, seed=9999, feature_cache_dir='data/cifar_cache', split='test')
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

        r = json.load(open(f'{RUNS}/{condition}_seed{s}/results.json'))
        df = pd.read_csv(f'{RUNS}/{condition}_seed{s}/metrics.csv')
        best_ep = r['early_stopping']['best_epoch']
        eff = float(df[df['epoch'] == best_ep].iloc[0]['eff_dim_rare'])
        racc = np.array(r['test_zipfsw']['per_class_acc'])[RARE].mean()
        res[label]['acc_per_dim'].append(racc / (eff + 1e-10))

for name, key, interp in [
    ('Total Embedding Variance', 'total_var', 'lower = more compact'),
    ('Rare Within-Class Spread', 'rare_spread', 'lower = tighter rare clusters'),
    ('Common Within-Class Spread', 'common_spread', 'lower = tighter common clusters'),
    ('Mean Inter-Centroid Distance', 'inter_dist', 'class separation'),
    ('Rare Acc / Eff Dim (efficiency)', 'acc_per_dim', 'higher = more efficient'),
]:
    c1 = np.array(res['C1'][key]); c4 = np.array(res['C4'][key])
    t_stat, p_val = stats.ttest_ind(c1, c4, equal_var=False)
    ratio = c4.mean() / (c1.mean() + 1e-10)
    print(f'\n{name} ({interp}):')
    print(f'  C1: {c1.mean():.4f} +/- {c1.std():.4f}')
    print(f'  C4: {c4.mean():.4f} +/- {c4.std():.4f}')
    print(f'  C4/C1 ratio: {ratio:.2f}x  Welch t={t_stat:.3f}, p={p_val:.4f}')

print('\n' + '=' * 75)
print('3. REPRESENTATIONAL EFFICIENCY SUMMARY')
print('=' * 75)
for key, name in [('total_var', 'Total variance'), ('rare_spread', 'Rare spread'),
                  ('inter_dist', 'Inter-centroid dist'), ('acc_per_dim', 'Acc per dim')]:
    c1m = np.mean(res['C1'][key]); c4m = np.mean(res['C4'][key])
    print(f'  {name}: C4 = {c4m / c1m * 100:.0f}% of C1 (C1={c1m:.4f}, C4={c4m:.4f})')
