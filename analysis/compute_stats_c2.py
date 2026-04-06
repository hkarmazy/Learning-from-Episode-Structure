import json, numpy as np, pandas as pd, torch
from scipy import stats
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader
from data.cifar_generator import CIFARPlaySequenceDataset
from model.visual_embedding_model import VisualPlayEmbedder

RUNS = 'runs_visual_v5es'
RARE = list(range(16, 32)); COMMON = list(range(16))
C1_SEEDS = [0, 1, 2, 3, 4]
C2_SEEDS = [1, 4]
C4_SEEDS = [1, 4]

def get_test_vals(cond, seeds, test_key):
    all_v, rare_v, common_v = [], [], []
    for s in seeds:
        r = json.load(open(f'{RUNS}/{cond}_seed{s}/results.json'))
        t = r[test_key]
        all_v.append(t['acc_all'])
        rare_v.append(np.array(t['per_class_acc'])[RARE].mean())
        common_v.append(np.array(t['per_class_acc'])[COMMON].mean())
    return np.array(all_v), np.array(rare_v), np.array(common_v)

def report(name, c1, c2, c4):
    print(f'\n--- {name} ---')
    print(f'  C1: {c1.mean()*100:.1f} +/- {c1.std()*100:.1f}%  (n={len(c1)})')
    print(f'  C2: {c2.mean()*100:.1f} +/- {c2.std()*100:.1f}%  (n={len(c2)})')
    print(f'  C4: {c4.mean()*100:.1f} +/- {c4.std()*100:.1f}%  (n={len(c4)})')
    for label, a, b in [('C1 vs C2', c1, c2), ('C1 vs C4', c1, c4), ('C2 vs C4', c2, c4)]:
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        diff = (a.mean() - b.mean()) * 100
        pooled = np.sqrt((a.std()**2 + b.std()**2) / 2) + 1e-10
        d = abs(a.mean() - b.mean()) / pooled
        print(f'  {label}: diff={diff:+.1f}pp, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}')

# ===== UNIFORM TEST =====
print('=' * 75)
print('UNIFORM TEST — C1 vs C2 vs C4')
print('=' * 75)
for name, idx_fn in [('Overall', 0), ('Common (0-15)', 2), ('Rare (16-31)', 1)]:
    c1 = get_test_vals('uniform', C1_SEEDS, 'test_uniform')[idx_fn]
    c2 = get_test_vals('zipf', C2_SEEDS, 'test_uniform')[idx_fn]
    c4 = get_test_vals('zipf_sw', C4_SEEDS, 'test_uniform')[idx_fn]
    report(name, c1, c2, c4)

# ===== ZIPF+SW TEST =====
print('\n' + '=' * 75)
print('ZIPF+SW TEST — C1 vs C2 vs C4')
print('=' * 75)
for name, idx_fn in [('Overall', 0), ('Common (0-15)', 2), ('Rare (16-31)', 1)]:
    c1 = get_test_vals('uniform', C1_SEEDS, 'test_zipfsw')[idx_fn]
    c2 = get_test_vals('zipf', C2_SEEDS, 'test_zipfsw')[idx_fn]
    c4 = get_test_vals('zipf_sw', C4_SEEDS, 'test_zipfsw')[idx_fn]
    report(name, c1, c2, c4)

# ===== TOST for C4 vs C1 rare on ZSW (already done, include for completeness) =====
print('\n' + '=' * 75)
print('TOST EQUIVALENCE TESTS (margin = 5pp)')
print('=' * 75)
c1_rare_zsw = get_test_vals('uniform', C1_SEEDS, 'test_zipfsw')[1]
c2_rare_zsw = get_test_vals('zipf', C2_SEEDS, 'test_zipfsw')[1]
c4_rare_zsw = get_test_vals('zipf_sw', C4_SEEDS, 'test_zipfsw')[1]
margin = 0.05
for label, a, b in [('C1 vs C4 rare ZSW', c1_rare_zsw, c4_rare_zsw),
                     ('C1 vs C2 rare ZSW', c1_rare_zsw, c2_rare_zsw)]:
    diff = a.mean() - b.mean()
    se = np.sqrt(a.std()**2/len(a) + b.std()**2/len(b))
    df_approx = min(len(a), len(b)) - 1
    t_upper = (diff - margin) / (se + 1e-10)
    p_upper = stats.t.cdf(t_upper, df=df_approx)
    t_lower = (diff + margin) / (se + 1e-10)
    p_lower = 1 - stats.t.cdf(t_lower, df=df_approx)
    p_tost = max(p_upper, p_lower)
    print(f'{label}: diff={diff*100:.1f}pp, TOST p={p_tost:.4f} ({"equivalent" if p_tost < 0.05 else "not equivalent"})')

# ===== EMBEDDING COMPACTNESS C2 =====
print('\n' + '=' * 75)
print('EMBEDDING COMPACTNESS — C1 vs C2 vs C4')
print('=' * 75)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

res = {}
for label, pairs in [('C1', [('uniform', s) for s in C1_SEEDS]),
                      ('C2', [('zipf', s) for s in C2_SEEDS]),
                      ('C4', [('zipf_sw', s) for s in C4_SEEDS])]:
    res[label] = {'total_var': [], 'rare_spread': [], 'common_spread': [], 'inter_dist': []}
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

for name, key, interp in [
    ('Total Embedding Variance', 'total_var', 'lower = more compact'),
    ('Rare Within-Class Spread', 'rare_spread', 'lower = tighter'),
    ('Common Within-Class Spread', 'common_spread', 'lower = tighter'),
    ('Inter-Centroid Distance', 'inter_dist', 'class separation'),
]:
    print(f'\n{name} ({interp}):')
    for label in ['C1', 'C2', 'C4']:
        v = np.array(res[label][key])
        print(f'  {label}: {v.mean():.4f} +/- {v.std():.4f}')
    for a_label, b_label in [('C1', 'C2'), ('C1', 'C4'), ('C2', 'C4')]:
        a = np.array(res[a_label][key]); b = np.array(res[b_label][key])
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        ratio = b.mean() / (a.mean() + 1e-10)
        print(f'  {a_label} vs {b_label}: ratio={ratio:.4f}x, t={t_stat:.3f}, p={p_val:.4f}')

# ===== EFF DIM at best epoch =====
print('\n' + '=' * 75)
print('EFFECTIVE DIMENSIONALITY AT BEST EPOCH')
print('=' * 75)
for label, cond, seeds in [('C1', 'uniform', C1_SEEDS), ('C2', 'zipf', C2_SEEDS), ('C4', 'zipf_sw', C4_SEEDS)]:
    vals = []
    for s in seeds:
        r = json.load(open(f'{RUNS}/{cond}_seed{s}/results.json'))
        df = pd.read_csv(f'{RUNS}/{cond}_seed{s}/metrics.csv')
        best_ep = r['early_stopping']['best_epoch']
        row = df[df['epoch'] == best_ep].iloc[0]
        vals.append(float(row['eff_dim_rare']))
    v = np.array(vals)
    print(f'{label} eff_dim_rare: {v.mean():.2f} +/- {v.std():.2f}')
