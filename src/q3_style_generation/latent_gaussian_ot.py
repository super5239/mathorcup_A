"""v3.1 第3、4项：经验边际潜在高斯生成 + 共同空间高斯最优传输迁移。

  x --G--> z --(风格条件收缩高斯 / OT)--> z' --G^-1--> x' --缺陷投影--> x''

G 由训练集全体风格共用的逐列经验 CDF 构建（中间秩处理并列，逆变换保留并列点质量）。
生成器与 analysis/q3 首版基线使用相同的三折划分、相同的随机数，并在相同的标准化坐标中评价。
规则见 docs/解题思路.md 第3节。
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm, wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'analysis' / 'q3'
OUT = ROOT / 'results' / 'q3'
OUT.mkdir(parents=True, exist_ok=True)
SEED, NGEN, STYLES = 20261026, 300, ['Tripel', 'Kriek', 'Lambic']   # 与首版基线一致
ALPHAS = [0, 0.25, 0.5, 0.75, 1.0]
CENTER_GAIN, NN_GAIN = 0.20, 0.10                                     # 冻结验收门槛
FLOOR_SHARE_MIN = 0.10                                                # 统计“下限点质量”的列：训练下限占比≥10%

x = pd.read_csv(ROOT / '1.csv').set_index('beer')
X = x.drop(columns=['beer_id', 'tasting_category_fine'])
labels = x.tasting_category_fine.to_numpy()
fields = list(X)
p = len(fields)
config = json.loads((BASE / 'defect_thresholds.json').read_text(encoding='utf-8'))
DEFECTS = {k: v['field'] for k, v in config['measured_defects'].items()}


# ---------------------------------------------------------------- 共同映射 G
class EmpiricalMarginals:
    """逐列经验 CDF。并列值取中间秩；逆变换对并列块返回常数（保留点质量），单点之间线性插值。"""
    EPS = 1e-9

    def fit(self, B):
        self.fwd, self.inv = [], []
        n = B.shape[0]
        for j in range(B.shape[1]):
            vals, cnt = np.unique(B[:, j], return_counts=True)
            b = np.cumsum(cnt) / n
            a = b - cnt / n
            mid = (a + b) / 2
            self.fwd.append((vals, mid))
            U, V = [], []
            for v, c, lo, hi, m in zip(vals, cnt, a, b, mid):
                if c > 1:
                    U += [lo + self.EPS, hi - self.EPS]
                    V += [v, v]
                else:
                    U.append(m)
                    V.append(v)
            self.inv.append((np.array(U), np.array(V)))
        return self

    def cdf(self, B):
        return np.column_stack([np.interp(B[:, j], *self.fwd[j]) for j in range(B.shape[1])])

    def transform(self, B):
        return norm.ppf(self.cdf(B))

    def inverse(self, Z):
        Uq = norm.cdf(Z)
        return np.column_stack([np.interp(Uq[:, j], *self.inv[j]) for j in range(Z.shape[1])])


class LatentStyles:
    def fit(self, A, styles):
        self.imp = SimpleImputer(strategy='median')
        B = self.imp.fit_transform(np.asarray(A, dtype=float))
        self.G = EmpiricalMarginals().fit(B)
        self.Z = self.G.transform(B)
        self.models = {'__all__': LedoitWolf().fit(self.Z)}
        for s in np.unique(styles):
            self.models[s] = LedoitWolf().fit(self.Z[styles == s])
        self.roots = {}
        for k, m in self.models.items():
            v, Q = np.linalg.eigh(m.covariance_)
            self.roots[k] = Q * np.sqrt(np.maximum(v, 0))[None, :]
        return self

    def sample(self, key, normals):
        return self.G.inverse(normals @ self.roots[key].T + self.models[key].location_)

    def transform(self, A):
        return self.G.transform(self.imp.transform(np.asarray(A, dtype=float)))


class BaselineStyles:
    """首版基线（analysis/q3/run_gaussian_baseline.py 的 GaussianStyles），用于同折配对比较与评价坐标。"""

    def fit(self, A, styles):
        self.imp = SimpleImputer(strategy='median')
        B = self.imp.fit_transform(np.asarray(A, dtype=float))
        self.scaler = StandardScaler().fit(B)
        self.Z = self.scaler.transform(B)
        self.low, self.high = B.min(axis=0), B.max(axis=0)
        self.models = {'__all__': LedoitWolf().fit(self.Z)}
        for s in np.unique(styles):
            self.models[s] = LedoitWolf().fit(self.Z[styles == s])
        self.roots = {}
        for k, m in self.models.items():
            v, Q = np.linalg.eigh(m.covariance_)
            self.roots[k] = Q * np.sqrt(np.maximum(v, 0))[None, :]
        return self

    def sample(self, key, normals):
        m = self.models[key]
        B = self.scaler.inverse_transform(normals @ self.roots[key].T + m.location_)
        return np.clip(B, self.low, self.high)

    def transform(self, A):
        return self.scaler.transform(self.imp.transform(np.asarray(A, dtype=float)))


def corr(A):
    B = A - A.mean(0)
    den = np.sqrt((B * B).sum(0))
    return (B.T @ B) / np.maximum(np.outer(den, den), 1e-15), den > 1e-12


def offdiag_share(cov):
    d = np.sqrt(np.diag(cov))
    C = cov / np.outer(d, d)
    return float((C ** 2).sum() - (np.diag(C) ** 2).sum()) / float((C ** 2).sum())


# ---------------------------------------------------------------- 生成：三折留出保真（与基线配对）
splits = json.loads((BASE / 'generator_holdout_splits.json').read_text(encoding='utf-8'))
rows, corr_rows = [], []
for fold, (train, test) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, labels)):
    assert X.index[train].tolist() == splits[fold]['train_beer']
    base = BaselineStyles().fit(X.iloc[train], labels[train])        # 评价坐标 = 基线的训练折标准化
    lat = LatentStyles().fit(X.iloc[train], labels[train])
    Btr = base.imp.transform(X.iloc[train].to_numpy(float))
    floor = Btr.min(axis=0)
    floor_cols = np.where((Btr == floor).mean(0) >= FLOOR_SHARE_MIN)[0]
    rng = np.random.default_rng(SEED + fold)
    centered = {m: ([], []) for m in ['latent_style', 'baseline_style', 'baseline_unconditional']}
    for style in STYLES:
        at = test[labels[test] == style]
        testz = base.transform(X.iloc[at])
        trainz = base.Z[labels[train] == style]
        Bte = base.imp.transform(X.iloc[at].to_numpy(float))
        noise = rng.normal(size=(NGEN, p))
        realnn = cdist(testz, trainz).min(1).mean() / np.sqrt(p)
        for method, B in [('latent_style', lat.sample(style, noise)),
                          ('baseline_style', base.sample(style, noise)),
                          ('baseline_unconditional', base.sample('__all__', noise))]:
            Z = base.transform(B)
            dnn = cdist(Z, trainz).min(1).mean() / np.sqrt(p)
            syn_floor = (B[:, floor_cols] == floor[floor_cols]).mean(0)
            real_floor = (Bte[:, floor_cols] == floor[floor_cols]).mean(0)
            j4 = fields.index('X4VG')
            rows.append({
                'fold': fold, 'style': style, 'method': method, 'n_test_style': len(at),
                'mean_coordinate_MAE_standardized': float(abs(Z.mean(0) - testz.mean(0)).mean()),
                'median_coordinate_MAE_standardized': float(np.median(abs(Z.mean(0) - testz.mean(0)))),
                'mean_coordinate_Wasserstein_standardized': float(np.mean(
                    [wasserstein_distance(Z[:, j], testz[:, j]) for j in range(p)])),
                'synthetic_to_real_NN_ratio': float(dnn / realnn),
                'floor_fraction_MAE_floor_columns': float(abs(syn_floor - real_floor).mean()),
                'n_floor_columns': len(floor_cols),
                'synthetic_4VG_floor_fraction': float((B[:, j4] == floor[j4]).mean()),
                'holdout_4VG_floor_fraction': float((Bte[:, j4] == floor[j4]).mean()),
            })
            centered[method][0].append(Z - Z.mean(0))
            centered[method][1].append(testz - testz.mean(0))
    for method, (aa, bb) in centered.items():
        Cg, vg = corr(np.vstack(aa))
        Cr, vr = corr(np.vstack(bb))
        ids = np.where(vg & vr)[0]
        i, j = np.triu_indices(len(ids), 1)
        corr_rows.append({'fold': fold, 'method': method, 'pooled_within_style_correlation_RMSE': float(
            np.sqrt(np.mean((Cg[np.ix_(ids, ids)][i, j] - Cr[np.ix_(ids, ids)][i, j]) ** 2)))})
    print(f'generator fold {fold + 1}/3 done', flush=True)

gm = pd.DataFrame(rows)
gc = pd.DataFrame(corr_rows)
gm.to_csv(OUT / 'generator_holdout_metrics_paired.csv', index=False, encoding='utf-8-sig')
gc.to_csv(OUT / 'generator_holdout_correlation_paired.csv', index=False, encoding='utf-8-sig')

# 复现核对：本脚本重跑的基线应与首版保存的数值一致
saved = pd.read_csv(BASE / 'generator_holdout_metrics.csv', encoding='utf-8-sig')
saved = saved[saved.method == 'style_conditional'].set_index(['fold', 'style'])
mine = gm[gm.method == 'baseline_style'].set_index(['fold', 'style'])
baseline_reproduced = float((saved.mean_coordinate_MAE_standardized - mine.mean_coordinate_MAE_standardized).abs().max())

# ---------------------------------------------------------------- 最终生成（全数据拟合）
fit = LatentStyles().fit(X, labels)
rng = np.random.default_rng(SEED + 100)
gen, desc = [], []
Bfull = fit.imp.transform(X.to_numpy(float))
for style in STYLES:
    B = fit.sample(style, rng.normal(size=(NGEN, p)))
    df = pd.DataFrame(B, columns=fields)
    df.insert(0, 'style', style)
    df.insert(0, 'candidate_id', [f'{style}_{i + 1:03}' for i in range(NGEN)])
    gen.append(df)
    real = fit.Z[labels == style]
    syn = fit.G.transform(B)
    nn = cdist(syn, real).min(1) / np.sqrt(p)
    m = fit.models[style]
    j4 = fields.index('X4VG')
    desc.append({
        'style': style, 'n_real': int((labels == style).sum()), 'n_generated': NGEN,
        'LedoitWolf_shrinkage': float(m.shrinkage_),
        'offdiag_share_of_correlation_Frobenius': offdiag_share(m.covariance_),
        'exact_training_copy_count': int((nn < 1e-12).sum()),
        'real_4VG_floor_fraction': float((Bfull[labels == style, j4] == Bfull[:, j4].min()).mean()),
        'synthetic_4VG_floor_fraction': float((B[:, j4] == Bfull[:, j4].min()).mean()),
    })
gen = pd.concat(gen, ignore_index=True)
gen.to_csv(OUT / 'generated_latent_231_coordinates.csv', index=False, encoding='utf-8-sig')
gdesc = pd.DataFrame(desc)
gdesc.to_csv(OUT / 'generator_final_descriptive.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 迁移：共同空间高斯 OT + 投影
def sqrtm_psd(S):
    v, Q = np.linalg.eigh(S)
    return (Q * np.sqrt(np.maximum(v, 0))) @ Q.T


src_mask, tgt_mask = labels == 'Pils/Lager', labels == 'Tripel'
src_m, tgt_m = fit.models['Pils/Lager'], fit.models['Tripel']
mu_s, mu_t = src_m.location_, tgt_m.location_
Ss, St = src_m.covariance_, tgt_m.covariance_
Ss_h = sqrtm_psd(Ss)
Ss_ih = np.linalg.inv(Ss_h)
A = Ss_ih @ sqrtm_psd(Ss_h @ St @ Ss_h) @ Ss_ih
A = (A + A.T) / 2
ot_check = float(np.abs(A @ Ss @ A - St).max() / np.abs(St).max())   # 闭式映射应满足 A Σs A = Σt

real_tripel_z = fit.Z[tgt_mask]
sources = X.loc[src_mask]

# 陈化方向指数：附件8阈值表 aging 列固定名单
th = pd.read_excel(ROOT / '8.xlsx', sheet_name='Sensory thresholds')
J_plus = [f for f in th.loc[th.aging == 'increase', 'field_name_R'] if f in fields]
J_minus = [f for f in th.loc[th.aging == 'decrease', 'field_name_R'] if f in fields]
jp, jm = [fields.index(f) for f in J_plus], [fields.index(f) for f in J_minus]


def aging_index(raw_row):
    """raw_row: 原储存尺度（含 NaN）。任一名单项缺失 -> NaN（不可完整评价）。"""
    if np.isnan(raw_row[jp + jm]).any():
        return np.nan
    F = fit.G.cdf(raw_row[None, :])[0]
    return 0.5 * (F[jp].mean() + (1 - F[jm]).mean())


def evaluate(beer, source_raw, cand_raw, z0, alpha, method):
    """cand_raw: 投影后的候选（原储存尺度，无缺失）。距离在共同正态分数空间中计算。"""
    zc = fit.G.transform(cand_raw[None, :])[0]
    d0c = np.linalg.norm(z0 - mu_t) / np.sqrt(p)
    d1c = np.linalg.norm(zc - mu_t) / np.sqrt(p)
    d0n = cdist(z0[None], real_tripel_z).min() / np.sqrt(p)
    d1n = cdist(zc[None], real_tripel_z).min() / np.sqrt(p)
    defect_ok, evaluable, floor_kept = [], [], []
    for name, f in DEFECTS.items():
        j = fields.index(f)
        a, b = source_raw[j], cand_raw[j]
        if np.isnan(a):
            evaluable.append(False)
            continue
        evaluable.append(True)
        defect_ok.append(bool(b <= a))
        if a == Bfull[:, j].min():
            floor_kept.append(bool(b == a))
    a0, a1 = aging_index(source_raw), aging_index(cand_raw)
    c1 = all(evaluable) and all(defect_ok) and all(floor_kept)
    c2 = 1 - d1c / d0c >= CENTER_GAIN
    c3 = 1 - d1n / d0n >= NN_GAIN
    c4_eval = not (np.isnan(a0) or np.isnan(a1))
    c4 = c4_eval and a1 <= a0 + 1e-12
    complete = all(evaluable) and c4_eval
    return {
        'method': method, 'beer': int(beer), 'alpha': alpha,
        'center_dist_source': d0c, 'center_dist': d1c, 'center_reduction': 1 - d1c / d0c,
        'nn_dist_source': d0n, 'nn_dist': d1n, 'nn_reduction': 1 - d1n / d0n,
        'source_change_RMS_latent': float(np.sqrt(np.mean((zc - z0) ** 2))),
        'four_defects_evaluable': all(evaluable), 'observed_defects_nonincrease': all(defect_ok),
        'floor_states_kept': all(floor_kept), 'n_floor_defects': len(floor_kept),
        'aging_index_source': a0, 'aging_index_candidate': a1, 'aging_evaluable': c4_eval,
        'C1_defects': c1, 'C2_center': bool(c2), 'C3_nn': bool(c3), 'C4_aging': bool(c4),
        'complete_evaluation': complete,
        'all_criteria': bool(c1 and c2 and c3 and c4),
    }


fwd = joblib.load(BASE / 'sensory_forward_model.joblib')
assert fwd['fields'] == fields
stale_j = fwd['targets'].index('stale_hops')

mig_rows, mig_coords = [], []
for beer, srow in sources.iterrows():
    source_raw = srow.to_numpy(float)
    z0 = fit.transform(srow.to_frame().T)[0]
    Tz = mu_t + A @ (z0 - mu_s)
    for alpha in ALPHAS:
        za = z0 + alpha * (Tz - z0)
        cand = fit.G.inverse(za[None, :])[0]                 # 已在观测范围内
        for name, f in DEFECTS.items():                       # 缺陷非增投影（源缺失则无法约束）
            j = fields.index(f)
            if not np.isnan(source_raw[j]):
                cand[j] = min(cand[j], source_raw[j])
        r = evaluate(beer, source_raw, cand, z0, alpha, 'latent_OT')
        mig_rows.append(r)
        rec = dict(zip(fields, cand))
        rec.update({'beer': int(beer), 'alpha': alpha})
        mig_coords.append(rec)

# 现有优化基线在相同标准下评价
bm = pd.read_csv(BASE / 'migration_231_coordinates.csv', encoding='utf-8-sig')
for _, r in bm.iterrows():
    beer = int(r.beer)
    source_raw = sources.loc[beer].to_numpy(float)
    z0 = fit.transform(sources.loc[[beer]])[0]
    mig_rows.append(evaluate(beer, source_raw, r[fields].to_numpy(float), z0, float(r.alpha), 'baseline_optimizer'))

mr = pd.DataFrame(mig_rows)
mc = pd.DataFrame(mig_coords)
# Stale hops 预测：仅参考展示
src_imp = fwd['imputer'].transform(sources[fields])
src_stale = dict(zip(sources.index, fwd['model'].predict(src_imp)[:, stale_j]))
cand_stale = fwd['model'].predict(fwd['imputer'].transform(mc[fields]))[:, stale_j]
mr.loc[mr.method == 'latent_OT', 'stale_hops_pred_change_reference'] = cand_stale - mc.beer.map(src_stale).to_numpy()
mr.to_csv(OUT / 'migration_acceptance_all.csv', index=False, encoding='utf-8-sig')
mc.to_csv(OUT / 'migration_latent_OT_231_coordinates.csv', index=False, encoding='utf-8-sig')


def select(df):
    out = []
    for (method, beer), g in df[df.alpha > 0].sort_values('alpha').groupby(['method', 'beer']):
        ok = g[g.all_criteria]
        if not g.complete_evaluation.all():
            status = '不可完整验收'
        elif len(ok):
            status = '达标'
        else:
            status = '未达到预设标准'
        out.append({'method': method, 'beer': beer, 'status': status,
                    'selected_alpha': float(ok.alpha.iloc[0]) if status == '达标' else np.nan})
    return pd.DataFrame(out)


sel = select(mr)
sel.to_csv(OUT / 'migration_selected_candidates.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 感官画像（Q1 最终前向模型 + 冻结锚点）
anchors = pd.read_json(ROOT / 'results' / 'q1' / 'q1_frozen_anchors.json', orient='index')
support = pd.read_csv(ROOT / 'results' / 'q1' / 'q1_prediction_support_tiers.csv', encoding='utf-8-sig', index_col=0)
targets = fwd['targets']
pred = pd.DataFrame(fwd['model'].predict(fwd['imputer'].transform(gen[fields])), columns=targets)
pred.insert(0, 'style', gen['style'])
y_real = pd.read_csv(ROOT / '4.csv').set_index('beer').loc[x.index, targets]
prof = []
for style in STYLES:
    P = pred[pred['style'] == style][targets]
    for t in targets:
        row = {'style': style, 'target': t, 'type': support.loc[t, 'type'], 'support_tier': support.loc[t, 'support_tier'],
               'skill_flag': '' if support.loc[t, 'skill_positive'] else '未优于风格均值',
               'pred_mean_standardized': float(P[t].mean()),
               'real_style_mean_standardized': float(y_real.loc[labels == style, t].mean())}
        if t in anchors.index:
            q05, q95 = anchors.loc[t, 'q05_oof'], anchors.loc[t, 'q95_oof']
            s = np.clip(6 * (P[t] - q05) / (q95 - q05), 0, 6).round()
            row['relative_score_median'] = float(s.median())
        prof.append(row)
prof = pd.DataFrame(prof)
prof.to_csv(OUT / 'generated_sensory_profile.csv', index=False, encoding='utf-8-sig')
pred.to_csv(OUT / 'generated_sensory_standardized.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 汇总
agg = gm.groupby('method')[['mean_coordinate_MAE_standardized', 'median_coordinate_MAE_standardized',
                            'mean_coordinate_Wasserstein_standardized', 'synthetic_to_real_NN_ratio',
                            'floor_fraction_MAE_floor_columns']].mean()
agg['pooled_within_style_correlation_RMSE'] = gc.groupby('method').pooled_within_style_correlation_RMSE.mean()
by_alpha = mr.groupby(['method', 'alpha'])[['center_reduction', 'nn_reduction', 'source_change_RMS_latent',
                                            'C1_defects', 'C2_center', 'C3_nn', 'C4_aging', 'all_criteria']].mean()
summary = {
    'baseline_reproduction_max_abs_diff': baseline_reproduced,
    'generator_holdout_mean': agg.round(4).to_dict('index'),
    'generator_final': gdesc.round(4).to_dict('records'),
    'ot_covariance_check_rel_err': ot_check,
    'aging_J_plus': J_plus, 'aging_J_minus': J_minus,
    'migration_by_alpha': {f'{m}|{a}': v for (m, a), v in by_alpha.round(3).to_dict('index').items()},
    'migration_status_counts': sel.groupby(['method', 'status']).size().to_dict(),
    'migration_selected': sel.to_dict('records'),
}
summary['migration_status_counts'] = {f'{k[0]}|{k[1]}': int(v) for k, v in summary['migration_status_counts'].items()}
json.dump(summary, open(OUT / 'summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
