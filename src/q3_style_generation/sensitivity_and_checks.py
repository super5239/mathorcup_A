"""Q3 补充（结果出现后的敏感性分析与核查，不替换冻结主结果）。

1. 判据2替换为目标风格经验容许域：d(x'', mu_T) <= q95{ d(x_i, mu_T,-i) }（28款真实Tripel留一中心距离），其余三条不变。
2. 迁移群体离散度：防止“全部收缩到中心”被当作风格保真。
3. 两法共同达标样本上、各自首次达标点的配对改变量（描述性；两法 alpha 网格不同）。
4. 首版基线拟合参数核对 + 运行环境记录。
"""
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_info

import latent_gaussian_ot as L   # 复用共同映射与模型（导入即重跑主脚本，结果确定）

OUT = L.OUT
fit, X, labels, p = L.fit, L.X, L.labels, L.p
mr = pd.read_csv(OUT / 'migration_acceptance_all.csv', encoding='utf-8-sig')
mc = pd.read_csv(OUT / 'migration_latent_OT_231_coordinates.csv', encoding='utf-8-sig')
bm = pd.read_csv(L.BASE / 'migration_231_coordinates.csv', encoding='utf-8-sig')

# ---------- 1. 留一经验容许域 ----------
Zt = fit.Z[labels == 'Tripel']
loo = np.array([np.linalg.norm(Zt[i] - np.delete(Zt, i, 0).mean(0)) for i in range(len(Zt))]) / np.sqrt(p)
r95 = float(np.quantile(loo, 0.95))
mr['C2_envelope'] = mr.center_dist <= r95
mr['all_criteria_envelope'] = mr.C1_defects & mr.C2_envelope & mr.C3_nn & mr.C4_aging


def select(df, flag):
    out = []
    for (method, beer), g in df[df.alpha > 0].sort_values('alpha').groupby(['method', 'beer']):
        ok = g[g[flag]]
        status = '不可完整验收' if not g.complete_evaluation.all() else ('达标' if len(ok) else '未达到预设标准')
        out.append({'method': method, 'beer': beer, 'status': status,
                    'selected_alpha': float(ok.alpha.iloc[0]) if status == '达标' else np.nan})
    return pd.DataFrame(out)


frozen = select(mr, 'all_criteria')
envelope = select(mr, 'all_criteria_envelope')
comp = frozen.merge(envelope, on=['method', 'beer'], suffixes=('_frozen', '_envelope'))
comp.to_csv(OUT / 'sensitivity_envelope_selection.csv', index=False, encoding='utf-8-sig')
mr.to_csv(OUT / 'migration_acceptance_with_envelope.csv', index=False, encoding='utf-8-sig')

# ---------- 2. 迁移群体离散度（共同正态分数空间） ----------
def disp(Z):
    return {'mean_dist_to_Tripel_center': float(np.linalg.norm(Z - fit.models['Tripel'].location_, axis=1).mean() / np.sqrt(p)),
            'mean_pairwise_dist': float(pdist(Z).mean() / np.sqrt(p))}


rows = [dict(group='real_Tripel(28)', **disp(Zt)),
        dict(group='source_Pils/Lager(11)', **disp(fit.Z[labels == 'Pils/Lager']))]
for alpha in sorted(mc.alpha.unique()):
    rows.append(dict(group=f'latent_OT alpha={alpha}', **disp(fit.G.transform(mc[mc.alpha == alpha][L.fields].to_numpy(float)))))
for alpha in sorted(bm.alpha.unique()):
    rows.append(dict(group=f'baseline_optimizer alpha={alpha}', **disp(fit.G.transform(bm[bm.alpha == alpha][L.fields].to_numpy(float)))))
dispersion = pd.DataFrame(rows)
dispersion.to_csv(OUT / 'migration_dispersion.csv', index=False, encoding='utf-8-sig')

# ---------- 3. 配对改变量（冻结判据下两法共同达标样本） ----------
both = frozen.pivot(index='beer', columns='method', values='selected_alpha').dropna()
chg = {}
for method in ['latent_OT', 'baseline_optimizer']:
    sub = mr[mr.method == method].set_index(['beer', 'alpha']).source_change_RMS_latent
    chg[method] = [float(sub.loc[(b, both.loc[b, method])]) for b in both.index]
paired = pd.DataFrame({'beer': both.index, 'alpha_OT': both.latent_OT.values, 'alpha_baseline': both.baseline_optimizer.values,
                       'change_OT': chg['latent_OT'], 'change_baseline': chg['baseline_optimizer']})
paired.to_csv(OUT / 'paired_change_at_first_pass.csv', index=False, encoding='utf-8-sig')

# ---------- 4. 首版基线参数核对 + 环境 ----------
saved = joblib.load(L.BASE / 'style_gaussians.joblib')
refit = L.BaselineStyles().fit(X, labels)
param = {
    'scaler_mean_max_abs_diff': float(np.abs(saved['scaler'].mean_ - refit.scaler.mean_).max()),
    'scaler_scale_max_abs_diff': float(np.abs(saved['scaler'].scale_ - refit.scaler.scale_).max()),
    'style_location_max_abs_diff': float(max(np.abs(saved['style_models'][k].location_ - refit.models[k].location_).max()
                                             for k in refit.models)),
    'style_covariance_max_abs_diff': float(max(np.abs(saved['style_models'][k].covariance_ - refit.models[k].covariance_).max()
                                               for k in refit.models)),
}
env = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
       'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__,
       'blas': [{k: i.get(k) for k in ('internal_api', 'version', 'threading_layer', 'num_threads')} for i in threadpool_info()]}
json.dump(env, open(OUT / 'environment_versions.json', 'w', encoding='utf-8'), indent=1)

summary = {
    'loo_center_dist_real_Tripel': {'mean': float(loo.mean()), 'min': float(loo.min()), 'max': float(loo.max()), 'q95_radius': r95},
    'status_frozen': {f'{m}|{st}': int(v) for (m, st), v in frozen.groupby(['method', 'status']).size().items()},
    'status_envelope': {f'{m}|{st}': int(v) for (m, st), v in envelope.groupby(['method', 'status']).size().items()},
    'selected_alpha_envelope': {f'{m}|{a}': int(v) for (m, a), v in
                                envelope.dropna().groupby('method').selected_alpha.value_counts().items()},
    'paired_first_pass_change': {'n': len(paired), 'OT_mean': float(paired.change_OT.mean()),
                                 'baseline_mean': float(paired.change_baseline.mean()),
                                 'OT_smaller_count': int((paired.change_OT < paired.change_baseline).sum())},
    'dispersion': dispersion.round(3).to_dict('records'),
    'baseline_parameter_check': param,
}
json.dump(summary, open(OUT / 'sensitivity_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print(comp.to_string())
