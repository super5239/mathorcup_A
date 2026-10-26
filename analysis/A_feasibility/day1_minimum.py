"""Day-one minimal analysis; outputs are intermediate data and audit records.

AI assistance: OpenAI Codex, 2026-10-24; exact model version unavailable.
Score mappings are explicit conventions, not recovered raw taster ratings.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / 'day1'
OUT.mkdir(exist_ok=True)
DATA = next(ROOT.glob('2026年第七届*/2026*赛道A*')) / '附件'
x = pd.read_csv(DATA / '1.csv').set_index('beer')
y = pd.read_csv(DATA / '4.csv').set_index('beer').loc[x.index]
assert x.index.is_unique and y.index.is_unique
assert (x.tasting_category_fine == y.tasting_category_fine).all()
X = x.drop(columns=['beer_id','tasting_category_fine'])
Y = y.drop(columns=['beer_id','tasting_category_fine'])
sn = pd.read_excel(DATA / '8.xlsx',sheet_name='Sensory names')
sn = sn[sn.R_name.isin(Y.columns)].set_index('R_name').loc[Y.columns]
sn.index.name='R_name'
assert sn.index.is_unique and len(sn)==50 and sn.DataType.notna().all()
types = sn.DataType.to_dict()
sn.to_csv(OUT/'sensory_metadata.csv',encoding='utf-8-sig')
threshold = pd.read_excel(DATA / '8.xlsx',sheet_name='Sensory thresholds')
trans = pd.read_csv(Path(__file__).parent/'verified_transforms.csv').set_index('column')
dictionary = []
for col in X:
    rows = threshold[threshold.field_name_R == col]
    one = rows.iloc[0] if len(rows) else None
    record = {'field':col,'nonmissing_count':int(X[col].notna().sum()),
              'missing_count':int(X[col].isna().sum()),'status':'needs_review'}
    for target, source in [('chemical_name','chemical_name'),('unit','unit'),
                           ('origin','origin_paper'),('threshold_median','threshold_median'),
                           ('threshold_mean','threshold_mean')]:
        val = one[source] if one is not None else None
        record[target] = None if val is None or pd.isna(val) else val
    if col in trans.index:
        t=trans.loc[col]
        record.update({'candidate_transform':t.best_transform_by_median,
                       'positive_pair_agreement_fraction':float(t.agreement_fraction_1e_7),
                       'status':'verified_positive_pairs' if bool(t.all_positive_pairs_verified) else 'partial_agreement_needs_review'})
    dictionary.append(record)
(OUT/'chemical_dictionary.json').write_text(json.dumps(dictionary,ensure_ascii=False,indent=2),encoding='utf-8')
pd.concat([x,Y.add_prefix('sensory__')],axis=1).to_csv(OUT/'joined_analysis_data.csv',encoding='utf-8-sig')

models = {
    'PLS_quantile8':make_pipeline(SimpleImputer(strategy='median'),QuantileTransformer(
        n_quantiles=100,output_distribution='normal',random_state=20261024),PLSRegression(n_components=8,scale=False)),
    'ExtraTrees_leaf3':make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(
        n_estimators=160,min_samples_leaf=3,max_features=0.7,random_state=20261024,n_jobs=2))}
folds = list(StratifiedKFold(3,shuffle=True,random_state=20261024).split(X,x.tasting_category_fine))
rng=np.random.default_rng(20261024)
masked_tests=[]
for train,test in folds:
    xx=X.iloc[test].copy();mask=(rng.random(xx.shape)<.10)&xx.notna().to_numpy()
    masked_tests.append(xx.mask(mask))

records=[];summary={};checks=[]
for name, model in models.items():
    pred=np.empty(Y.shape);masked=np.empty(Y.shape);draft=np.empty(Y.shape)
    for fold,(train,test) in enumerate(folds):
        fit=clone(model).fit(X.iloc[train],Y.iloc[train])
        pred[test]=fit.predict(X.iloc[test]);masked[test]=fit.predict(masked_tests[fold])
        for j,col in enumerate(Y):
            if types[col]=='numeric':
                lo,hi=Y.iloc[train,j].quantile([.05,.95])
                assert hi>lo
                draft[test,j]=np.rint(np.clip(6*(pred[test,j]-lo)/(hi-lo),0,6))
                checks.append({'model':name,'fold':fold,'field':col,'type':'numeric',
                               'anchor_q05':float(lo),'anchor_q95':float(hi)})
            else:
                draft[test,j]=(pred[test,j]>0).astype(int)
                checks.append({'model':name,'fold':fold,'field':col,'type':'categorical',
                               'selection_threshold_standardized':0})
    for condition,pp in [('natural_missing',pred),('extra_10pct_mask',masked)]:
        rr=r2_score(Y,pp,multioutput='raw_values')
        aa=mean_absolute_error(Y,pp,multioutput='raw_values')
        for j,col in enumerate(Y):
            records.append({'model':name,'condition':condition,'field':col,'type':types[col],
                            'oof_R2':float(rr[j]),'oof_MAE_standardized':float(aa[j])})
        summary[name+'__'+condition]={}
        for kind in ['all','numeric','categorical']:
            idx=np.array([True if kind=='all' else types[c]==kind for c in Y])
            summary[name+'__'+condition][kind]={'n_dimensions':int(idx.sum()),
                'mean_oof_R2':float(rr[idx].mean()),'mean_oof_MAE_standardized':float(aa[idx].mean())}
    pd.DataFrame(pred,index=Y.index,columns=Y.columns).to_csv(OUT/(name+'_oof_standardized.csv'),encoding='utf-8-sig')
    dd=pd.DataFrame(draft.astype(int),index=Y.index,columns=Y.columns)
    assert all(dd[c].between(0,6 if types[c]=='numeric' else 1).all() for c in dd)
    dd.to_csv(OUT/(name+'_draft_relative_scores.csv'),encoding='utf-8-sig')
    display=dd.copy()
    for col in Y:
        if types[col]=='categorical':
            parent='_'.join(col.split('_')[:2])+'_all'
            assert parent in dd and types[parent]=='numeric'
            display[col]=dd[col]*dd[parent]
    display.to_csv(OUT/(name+'_draft_template_display_scores.csv'),encoding='utf-8-sig')
    print(name, json.dumps({k:v for k,v in summary.items() if k.startswith(name)},ensure_ascii=False),flush=True)
pd.DataFrame(records).to_csv(OUT/'baseline_metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(checks).to_csv(OUT/'fold_score_anchors.csv',index=False,encoding='utf-8-sig')
summary.update({'type_counts':sn.DataType.value_counts().to_dict(),
    'protocol':'fixed parameters; 3-fold style-stratified OOF; train-fold-only imputation and quantile transform; no nested tuning',
    'score_rule':'DRAFT: numeric q05->0, q95->6 linear clipped and rounded; categorical standardized prediction>0; not original ratings or validated binary labels',
    'template_display_rule':'categorical displayed score = draft checkbox * parent numeric strength, following template formulas; checkbox is stored separately',
    'n_rows':len(X),'n_features':X.shape[1],'n_targets':Y.shape[1]})
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('DONE',summary['type_counts'],flush=True)
