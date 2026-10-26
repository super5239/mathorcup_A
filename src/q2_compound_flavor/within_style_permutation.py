"""Q2 敏感性实验：全局置换 vs 风格内置换（同一测试折、同一模型、同一评价样本集合）。

沿用 Q1 正式 OOF 的 15 个划分与 Extra Trees 固定配置重新拟合 15 次（不搜索超参数）；
不改动原 OOF 预测与 100 次 bootstrap，不重算入选率，不替换原 Top-3 主名单。
风格内置换只支持“在已记录风格分组内仍存在预测依赖”，不构成化学因果结论。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
FEAS = ROOT / 'analysis' / 'A_feasibility'
Q2 = ROOT / 'analysis' / 'q2'
OUT = ROOT / 'results' / 'q2'
R, SEED, BATCH = 3, 20261027, 12
threadpool_limits(2)

x = pd.read_csv(ROOT / '1.csv').set_index('beer')
y = pd.read_csv(ROOT / '4.csv').set_index('beer').loc[x.index]
X = x.drop(columns=['beer_id', 'tasting_category_fine'])
Y = y.drop(columns=['beer_id', 'tasting_category_fine'])
style = x.tasting_category_fine
fields, targets = list(X), list(Y)
mapping = pd.read_csv(Q2 / 'chemical_mapping_and_modules.csv', encoding='utf-8-sig').set_index('field').loc[fields]
eligible = np.where(mapping.eligible_primary_top3.to_numpy())[0]
top3 = pd.read_csv(Q2 / 'top3_chemical_candidates.csv', encoding='utf-8-sig')
splits = json.loads((FEAS / 'floor_ablation' / 'split_and_floor_log.json').read_text(encoding='utf-8'))
oof = np.load(FEAS / 'floor_ablation' / 'baseline_oof.npz', allow_pickle=True)
assert list(oof['fields']) == targets and (oof['beer'] == X.index.to_numpy()).all()
pos = {b: i for i, b in enumerate(X.index)}

imp = {k: np.zeros((15, len(targets), len(fields))) for k in ['global', 'within']}
log, max_oof_diff = [], 0.0
for idx, s in enumerate(splits):
    tr = np.array([pos[b] for b in s['train_beer']])
    te = np.array([pos[b] for b in s['test_beer']])
    imputer = SimpleImputer(strategy='median')
    at = imputer.fit_transform(X.to_numpy(float)[tr])
    bt = imputer.transform(X.to_numpy(float)[te])
    et = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=3, max_features=.7,
                             random_state=20261025 + idx, n_jobs=2).fit(at, Y.to_numpy()[tr])
    pred_all = et.predict(bt)
    max_oof_diff = max(max_oof_diff, float(np.abs(pred_all - oof['prediction'][s['repeat']][te]).max()))
    # 评价集合：测试折内同风格至少2款的样本（两种置换共用）
    st = style.to_numpy()[te]
    counts = pd.Series(st).value_counts()
    keep = np.isin(st, counts.index[counts >= 2])
    E, yE, groups = bt[keep], Y.to_numpy()[te][keep], st[keep]
    n = len(E)
    base_err = np.abs(et.predict(E) - yE).mean(0)
    rng = np.random.default_rng(SEED + idx)
    gidx = [np.where(groups == g)[0] for g in np.unique(groups)]
    for first in range(0, len(fields), BATCH):
        cols = list(range(first, min(first + BATCH, len(fields))))
        batch = np.tile(E, (len(cols) * R * 2, 1))
        for u, j in enumerate(cols):
            for r in range(R):
                pg = rng.permutation(n)
                pw = np.arange(n)
                for g in gidx:
                    pw[g] = rng.permutation(g)
                for k, perm in enumerate([pg, pw]):
                    sl = slice(((u * R + r) * 2 + k) * n, ((u * R + r) * 2 + k + 1) * n)
                    batch[sl, j] = E[perm, j]
        P = et.predict(batch).reshape(len(cols), R, 2, n, len(targets))
        d = np.abs(P - yE[None, None, None]).mean(3).mean(1) - base_err[None, None]   # (cols, 2, targets)
        imp['global'][idx][:, cols] = d[:, 0].T
        imp['within'][idx][:, cols] = d[:, 1].T
    log.append({'split': idx, 'repeat': s['repeat'], 'fold': s['fold'], 'n_test': len(te), 'n_evaluable': int(n),
                'n_styles_evaluable': len(gidx), 'excluded_styles': counts.index[counts < 2].tolist()})
    print(f'split {idx + 1}/15: evaluable {n}/{len(te)}', flush=True)

G, W = imp['global'].mean(0), imp['within'].mean(0)
pd.DataFrame(G, index=targets, columns=fields).to_csv(OUT / 'perm_global_importance_mean15.csv', encoding='utf-8-sig')
pd.DataFrame(W, index=targets, columns=fields).to_csv(OUT / 'perm_within_style_importance_mean15.csv', encoding='utf-8-sig')


def rank_among_eligible(vec, j):
    order = eligible[np.argsort(-vec[eligible], kind='stable')]
    return int(np.where(order == j)[0][0]) + 1


rows = []
for _, r in top3.iterrows():
    t, j = targets.index(r.target), fields.index(r.field)
    g, w = imp['global'][:, t, j], imp['within'][:, t, j]
    rows.append({'target': r.target, 'rank_original': r['rank'], 'field': r.field,
                 'global_imp': G[t, j], 'within_imp': W[t, j],
                 'retention_ratio': W[t, j] / G[t, j] if G[t, j] > 0 else np.nan,
                 'within_positive_splits': int((w > 0).sum()), 'paired_within_minus_global_mean': float((w - g).mean()),
                 'global_rank_217': rank_among_eligible(G[t], j), 'within_rank_217': rank_among_eligible(W[t], j)})
res = pd.DataFrame(rows)
res['within_top3'] = res.within_rank_217 <= 3
res['within_top10'] = res.within_rank_217 <= 10
res.to_csv(OUT / 'q2_within_style_permutation_top3.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(log).to_csv(OUT / 'q2_within_style_permutation_log.csv', index=False, encoding='utf-8-sig')

per_field = res.groupby('field').agg(n_targets=('target', 'size'), median_retention=('retention_ratio', 'median'),
                                     share_within_positive=('within_imp', lambda v: float((v > 0).mean())),
                                     share_within_top10=('within_top10', 'mean')).sort_values('n_targets', ascending=False)
per_field.to_csv(OUT / 'q2_within_style_permutation_by_field.csv', encoding='utf-8-sig')

# 原 Top-3 的排序在风格内置换下是否保留（同一目标的3个候选之间）
kept = []
for t, g in res.groupby('target'):
    kept.append(list(g.sort_values('rank_original').within_imp.rank(ascending=False).astype(int)) == [1, 2, 3])
summary = {
    'refit_models': 15, 'refit_matches_saved_OOF_max_abs_diff': max_oof_diff,
    'evaluable_samples_total': int(sum(l['n_evaluable'] for l in log)),
    'test_samples_total': int(sum(l['n_test'] for l in log)),
    'top3_entries': len(res),
    'share_global_imp_positive': float((res.global_imp > 0).mean()),
    'share_within_imp_positive': float((res.within_imp > 0).mean()),
    'median_retention_ratio': float(res.retention_ratio.median()),
    'share_within_top10_of_217': float(res.within_top10.mean()),
    'share_within_top3_of_217': float(res.within_top3.mean()),
    'targets_with_original_order_kept_within': int(sum(kept)),
    'top_fields': per_field.head(6).round(3).to_dict('index'),
    'permutation_repeats': R,
}
json.dump(summary, open(OUT / 'within_style_permutation_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
