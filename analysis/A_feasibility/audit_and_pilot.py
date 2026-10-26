"""Exploratory audit only; not a final contest model.

AI assistance: OpenAI Codex, 2026-10-24; model version unavailable in runtime.
All reported metrics use out-of-fold predictions with fold-local preprocessing.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = next(ROOT.glob('2026年第七届*/2026*赛道A*')) / '附件'
Xraw = pd.read_csv(DATA / '1.csv').set_index('beer')
Yraw = pd.read_csv(DATA / '4.csv').set_index('beer').loc[Xraw.index]
X = Xraw.drop(columns=['beer_id', 'tasting_category_fine'])
Y = Yraw.drop(columns=['beer_id', 'tasting_category_fine'])
ref = pd.read_excel(DATA / '8.xlsx', sheet_name=None)
cal = ref['Calibrated chemical dataset'].set_index('beer').loc[X.index]

transforms = []
for col in cal.columns:
    if col not in X:
        continue
    a = X[col].to_numpy(float)
    b = cal[col].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    candidates = [
        ('identity', b), ('ln', np.log(np.where(b > 0, b, np.nan))),
        ('log10', np.log10(np.where(b > 0, b, np.nan)))
    ]
    errors = {name: float(np.median(abs(a[ok]-v[ok]))) for name, v in candidates}
    best = min(errors, key=errors.get)
    expected = dict(candidates)[best]
    max_error = float(np.max(abs(a[ok]-expected[ok])))
    transforms.append({'column': col, 'n_positive_pairs': int(ok.sum()),
        'best_transform_by_median': best, 'median_error': errors[best],
        'max_error': max_error, 'all_positive_pairs_verified':max_error < 1e-7,
        'agreement_fraction_1e_7':float(np.mean(abs(a[ok]-expected[ok]) < 1e-7)),
        'raw_zero_count': int((b == 0).sum()),
        'stored_values_when_raw_zero': np.unique(a[b == 0]).tolist()[:8]})
pd.DataFrame(transforms).to_csv(OUT / 'verified_transforms.csv', index=False, encoding='utf-8-sig')

threshold = ref['Sensory thresholds']
name_map = dict(zip(threshold.field_name_R, threshold.chemical_name_short))
name_map.update(dict(zip(threshold.field_name_R, threshold.chemical_name)))
matrix2 = pd.read_csv(DATA / '2.csv', index_col=0)
mapped = {c: name_map[c] for c in X if c in name_map and name_map[c] in matrix2.index}
corr_errors = {}
for method in ['pearson', 'spearman']:
    corr = X[list(mapped)].corr(method=method).to_numpy()
    supplied = matrix2.loc[list(mapped.values()),list(mapped.values())].to_numpy()
    ij = np.triu_indices_from(corr, k=1)
    corr_errors[method] = {'pairs': len(ij[0]), 'MAE': float(np.nanmean(abs(corr[ij]-supplied[ij]))),
        'max_abs_error': float(np.nanmax(abs(corr[ij]-supplied[ij])))}

matrix5 = pd.read_csv(DATA / '5.csv', index_col=0)
sn = ref['Sensory names']
ymap = dict(zip(sn.R_name, sn.CleanName_PanelFeedback))
validy = {c: ymap[c] for c in Y if c in ymap and ymap[c] in matrix5.index}
corr5_errors = {}
for method in ['pearson', 'spearman']:
    a = Y[list(validy)].corr(method=method).to_numpy()
    b = matrix5.loc[list(validy.values()), list(validy.values())].to_numpy()
    ij = np.triu_indices_from(a, k=1)
    corr5_errors[method] = {'matched_dimensions':len(validy), 'MAE':float(np.nanmean(abs(a[ij]-b[ij])))}

wb = load_workbook(DATA / '3.xlsx', data_only=False, read_only=True)
headers = [v for v in next(wb['summary'].values) if v is not None]
styles = Xraw.tasting_category_fine
models = {
    'training_mean': make_pipeline(SimpleImputer(strategy='median'), DummyRegressor()),
    'ridge_alpha100': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=100)),
    'pls_8': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), PLSRegression(n_components=8, scale=False)),
    'ridge_quantile_alpha100': make_pipeline(SimpleImputer(strategy='median'),
        QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=20261024), Ridge(alpha=100)),
    'pls_quantile_8': make_pipeline(SimpleImputer(strategy='median'),
        QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=20261024), PLSRegression(n_components=8, scale=False)),
    'extra_trees_leaf3': make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
        n_estimators=160, min_samples_leaf=3, max_features=0.7, random_state=20261024, n_jobs=2))
}
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=20261024).split(X, styles))
records = []
summary = {}
for name, model in models.items():
    pred = np.empty(Y.shape)
    for train, test in folds:
        fit = clone(model).fit(X.iloc[train], Y.iloc[train])
        pred[test] = fit.predict(X.iloc[test])
    scores = r2_score(Y, pred, multioutput='raw_values')
    for j, col in enumerate(Y):
        records.append({'model':name, 'sensory':col, 'oof_R2':float(scores[j]),
            'oof_MAE':float(mean_absolute_error(Y.iloc[:,j],pred[:,j]))})
    summary[name] = {'mean_oof_R2_50':float(scores.mean()),
        'mean_oof_R2_aroma_flavour_30':float(scores[:30].mean()),
        'positive_R2_count':int((scores > 0).sum()),
        'lowest5':dict(sorted(zip(Y.columns, scores), key=lambda p:p[1])[:5]),
        'highest5':dict(sorted(zip(Y.columns, scores), key=lambda p:p[1], reverse=True)[:5])}
    print(name, json.dumps(summary[name],ensure_ascii=False), flush=True)
pd.DataFrame(records).to_csv(OUT / 'pilot_oof_metrics.csv', index=False, encoding='utf-8-sig')
audit = {'status':'exploratory; fixed hyperparameters; not nested CV; no original-score validation',
    'n':len(X),'p':X.shape[1],'q':Y.shape[1],
    'chemical_missing':X.isna().sum()[lambda v:v>0].to_dict(),
    'chemical_missing_fraction':float(X.isna().sum().sum()/X.size),
    'sensory_missing':int(Y.isna().sum().sum()),
    'styles':styles.value_counts().to_dict(),
    'template_summary_columns':headers,
    'correlation2_audit':corr_errors,'correlation5_audit':corr5_errors,
    'transform_counts_by_median_only':pd.Series([x['best_transform_by_median'] for x in transforms]).value_counts().to_dict(),
    'fully_verified_transform_counts':pd.Series([x['best_transform_by_median'] for x in transforms if x['all_positive_pairs_verified']]).value_counts().to_dict(),
    'pilot':summary}
(OUT / 'audit_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('AUDIT',json.dumps({k:v for k,v in audit.items() if k!='pilot'},ensure_ascii=False),flush=True)
