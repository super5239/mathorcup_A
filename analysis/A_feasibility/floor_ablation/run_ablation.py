"""One fixed paired ablation. AI assistance: OpenAI Codex, 2026-10-25.
See 实验协议.md for frozen decision rule. Does not modify original data.
"""
from pathlib import Path
from time import perf_counter
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error,r2_score

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
DATA=next(ROOT.glob('2026年第七届*/2026*赛道A*'))/'附件'
protocol_sha=hashlib.sha256((OUT/'实验协议.md').read_bytes()).hexdigest()
x=pd.read_csv(DATA/'1.csv').set_index('beer')
y=pd.read_csv(DATA/'4.csv').set_index('beer').loc[x.index]
X=x.drop(columns=['beer_id','tasting_category_fine']);Y=y.drop(columns=['beer_id','tasting_category_fine'])
meta=pd.read_csv(OUT.parent/'day1/sensory_metadata.csv',index_col=0)
assert meta.index.is_unique and set(Y.columns)<=set(meta.index)
types=meta.loc[Y.columns,'DataType'].to_numpy()
splitter=RepeatedStratifiedKFold(n_splits=3,n_repeats=5,random_state=20261025)
splits=list(splitter.split(X,x.tasting_category_fine))
split_log=[];fold_rows=[];dim_rows=[]
predictions={name:np.empty((5,len(Y),Y.shape[1])) for name in ['baseline','floor_indicators']}
start=perf_counter()
for idx,(train,test) in enumerate(splits):
    repeat,fold=divmod(idx,3)
    a=X.iloc[train].to_numpy();b=X.iloc[test].to_numpy()
    mins=np.nanmin(a,axis=0)
    flags_train=np.isfinite(a)&np.isclose(a,mins,rtol=0,atol=1e-12)
    freq=flags_train.sum(axis=0)/np.isfinite(a).sum(axis=0)
    selected=freq>=.4
    flags_test=np.isfinite(b)&np.isclose(b,mins,rtol=0,atol=1e-12)
    names=X.columns[selected].tolist()
    split_log.append({'repeat':repeat,'fold':fold,'train_beer':X.index[train].tolist(),
        'test_beer':X.index[test].tolist(),'floor_columns':names,
        'training_floors':dict(zip(names,mins[selected].tolist()))})
    row={'repeat':repeat,'fold':fold,'n_train':len(train),'n_test':len(test),'n_floor_columns':len(names)}
    for model_name,aa,bb in [
        ('baseline',a,b),
        ('floor_indicators',np.column_stack([a,flags_train[:,selected].astype(float)]),
         np.column_stack([b,flags_test[:,selected].astype(float)]))]:
        tic=perf_counter();imputer=SimpleImputer(strategy='median')
        at=imputer.fit_transform(aa);bt=imputer.transform(bb)
        fit=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,max_features=.7,
                               random_state=20261025+idx,n_jobs=2)
        fit.fit(at,Y.iloc[train]);pred=fit.predict(bt)
        assert pred.shape==(len(test),50) and np.isfinite(pred).all()
        predictions[model_name][repeat,test,:]=pred
        err=mean_absolute_error(Y.iloc[test],pred,multioutput='raw_values')
        r2=r2_score(Y.iloc[test],pred,multioutput='raw_values')
        row[model_name+'_MAE']=float(err.mean());row[model_name+'_R2']=float(r2.mean())
        row[model_name+'_seconds']=perf_counter()-tic
        for kind in ['numeric','categorical']:
            row[model_name+'_'+kind+'_MAE']=float(err[types==kind].mean())
        for j,col in enumerate(Y):dim_rows.append({'repeat':repeat,'fold':fold,
            'model':model_name,'field':col,'type':types[j],'MAE':float(err[j]),'R2':float(r2[j])})
    row['MAE_improvement']=row['baseline_MAE']-row['floor_indicators_MAE'];fold_rows.append(row)
    print(f'{idx+1}/15 MAE improvement={row["MAE_improvement"]:.6f}, indicators={len(names)}',flush=True)

df=pd.DataFrame(fold_rows);d=df.MAE_improvement.to_numpy();m=len(d)
ratio=float((df.n_test/df.n_train).mean());se=float(np.sqrt((1/m+ratio)*d.var(ddof=1)))
mean=float(d.mean());crit=float(t.ppf(.975,m-1));lo,hi=mean-crit*se,mean+crit*se
rel=mean/float(df.baseline_MAE.mean());keep=bool(rel>=.01 and lo>0)
assert len(df)==15 and all(len(set(s['train_beer'])&set(s['test_beer']))==0 for s in split_log)
summary={'protocol_sha256':protocol_sha,'n_splits':m,
    'baseline_mean_fold_MAE':float(df.baseline_MAE.mean()),
    'floor_indicators_mean_fold_MAE':float(df.floor_indicators_MAE.mean()),
    'mean_paired_MAE_improvement':mean,'relative_MAE_improvement':rel,
    'corrected_approx_95pct_CI':[lo,hi],'corrected_standard_error':se,
    'mean_test_train_ratio':ratio,'folds_improved':int((d>0).sum()),
    'keep_floor_indicators':keep,'decision':'use_floor_indicators' if keep else 'retain_baseline',
    'total_seconds':perf_counter()-start,
    'indicator_count_range':[int(df.n_floor_columns.min()),int(df.n_floor_columns.max())],
    'note':'Internal selection only; corrected CI approximate; outcomes standardized aggregates, not raw grades.'}
for name in predictions:
    summary[name+'_mean_repeat_pooled_R2']=float(np.mean([r2_score(Y,p) for p in predictions[name]]))
    for kind in ['numeric','categorical']:
        summary[name+'_'+kind+'_mean_fold_MAE']=float(df[name+'_'+kind+'_MAE'].mean())
    np.savez_compressed(OUT/(name+'_oof.npz'),prediction=predictions[name],beer=Y.index.to_numpy(),fields=Y.columns.to_numpy())
df.to_csv(OUT/'paired_fold_metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(dim_rows).to_csv(OUT/'dimension_diagnostics.csv',index=False,encoding='utf-8-sig')
(OUT/'split_and_floor_log.json').write_text(json.dumps(split_log,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
