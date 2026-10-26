"""Supplement existing Q1 OOF predictions; no estimator training.
AI assistance: OpenAI Codex, 2026-10-25. Labels remain standardized aggregates.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OUT=BASE/'q1_metrics'
OUT.mkdir(exist_ok=True)
DATA=next(ROOT.glob('2026年第七届*/2026*赛道A*'))/'附件'
xx=pd.read_csv(DATA/'1.csv').set_index('beer')
yy=pd.read_csv(DATA/'4.csv').set_index('beer').loc[xx.index]
Y=yy.drop(columns=['beer_id','tasting_category_fine'])
meta=pd.read_csv(BASE/'day1/sensory_metadata.csv',index_col=0).loc[Y.columns]
types=meta.DataType.to_numpy()
style=xx.tasting_category_fine
loaded=np.load(BASE/'floor_ablation/baseline_oof.npz',allow_pickle=True)
assert np.array_equal(loaded['beer'],Y.index.to_numpy())
assert np.array_equal(loaded['fields'],Y.columns.to_numpy())
preds={'training_mean':np.full((5,len(Y),50),np.nan),
       'style_mean':np.full((5,len(Y),50),np.nan),
       'ExtraTrees':loaded['prediction']}
splits=json.loads((BASE/'floor_ablation/split_and_floor_log.json').read_text(encoding='utf-8'))
seen=np.zeros((5,len(Y)),dtype=int)
fold_rows=[]
for split in splits:
    rep,fold=split['repeat'],split['fold']
    tr=Y.index.get_indexer(split['train_beer']);te=Y.index.get_indexer(split['test_beer'])
    assert (tr>=0).all() and (te>=0).all() and not set(tr)&set(te)
    seen[rep,te]+=1
    ym=Y.iloc[tr].mean().to_numpy()
    preds['training_mean'][rep,te,:]=ym
    group=Y.iloc[tr].groupby(style.iloc[tr]).mean()
    assert set(style.iloc[te])<=set(group.index)
    preds['style_mean'][rep,te,:]=group.loc[style.iloc[te]].to_numpy()
    for name,arr in preds.items():
        pp=arr[rep,te,:]
        fold_rows.append({'repeat':rep,'fold':fold,'model':name,
                         'MAE':float(mean_absolute_error(Y.iloc[te],pp)),
                         'R2':float(r2_score(Y.iloc[te],pp))})
assert np.all(seen==1) and all(np.isfinite(p).all() for p in preds.values())

rows=[];aggregate=[]
for name,arr in preds.items():
    for rep,pp in enumerate(arr):
        mae=mean_absolute_error(Y,pp,multioutput='raw_values')
        r2=r2_score(Y,pp,multioutput='raw_values')
        mse=mean_squared_error(Y,pp,multioutput='raw_values')
        rho=np.array([float(spearmanr(Y.iloc[:,j],pp[:,j]).statistic) for j in range(50)]) if name!='training_mean' else np.full(50,np.nan)
        for j,col in enumerate(Y):
            rows.append({'model':name,'repeat':rep,'field':col,'type':types[j],
                         'MAE':float(mae[j]),'MSE':float(mse[j]),'R2':float(r2[j]),'Spearman':float(rho[j])})
        for kind in ['all','numeric','categorical']:
            mask=np.ones(50,dtype=bool) if kind=='all' else types==kind
            aggregate.append({'model':name,'repeat':rep,'group':kind,'MAE':float(mae[mask].mean()),
                              'R2':float(r2[mask].mean()),
                              'Spearman':float(np.mean(rho[mask])) if name!='training_mean' else np.nan})
detail=pd.DataFrame(rows)
dim=detail.groupby(['model','field','type'],as_index=False)[['MAE','MSE','R2','Spearman']].mean()
agg=pd.DataFrame(aggregate).groupby(['model','group'],as_index=False)[['MAE','R2','Spearman']].mean()
model_dim=dim[dim.model=='ExtraTrees'].set_index('field').loc[Y.columns]
style_dim=dim[dim.model=='style_mean'].set_index('field').loc[Y.columns]
model_dim['relative_MAE_gain_vs_style']=(style_dim.MAE-model_dim.MAE)/style_dim.MAE
model_dim['R2_gain_vs_style']=model_dim.R2-style_dim.R2
model_dim['MSE_skill_vs_style']=1-model_dim.MSE/style_dim.MSE
model_dim['MSE_style_mean']=style_dim.MSE
fold_df=pd.DataFrame(fold_rows)
paired=fold_df.pivot(index=['repeat','fold'],columns='model',values='MAE')
absolute=(paired.style_mean-paired.ExtraTrees).to_numpy()
relative=(absolute/paired.style_mean).to_numpy()
ratios=np.array([len(s['test_beer'])/len(s['train_beer']) for s in splits])
def corrected_interval(values):
    m=len(values);avg=float(np.mean(values))
    se=float(np.sqrt((1/m+ratios.mean())*np.var(values,ddof=1)))
    width=float(t.ppf(.975,m-1)*se)
    return {'mean':avg,'approx_95pct_CI':[avg-width,avg+width],'corrected_standard_error':se}
comparison={'absolute_MAE_improvement':corrected_interval(absolute),
            'mean_fold_relative_MAE_improvement':corrected_interval(relative),
            'note':'Supplementary approximate inference on existing 15 paired folds, not independent confirmation. Mean fold ratios differ slightly from pooled ratio.'}
summ={}
for kind in ['all','numeric','categorical']:
    subset=model_dim if kind=='all' else model_dim[model_dim.type==kind]
    summ[kind]={'n_dimensions':len(subset),
        'R2_distribution':{'min':float(subset.R2.min()),'q25':float(subset.R2.quantile(.25)),
                           'median':float(subset.R2.median()),'q75':float(subset.R2.quantile(.75)),
                           'max':float(subset.R2.max())},
        'Spearman_distribution':{'min':float(subset.Spearman.min()),'q25':float(subset.Spearman.quantile(.25)),
                           'median':float(subset.Spearman.median()),'q75':float(subset.Spearman.quantile(.75)),
                           'max':float(subset.Spearman.max())},
        'positive_R2_dimensions':int((subset.R2>0).sum()),
        'dimensions_better_than_style_MAE':int((subset.relative_MAE_gain_vs_style>0).sum()),
        'dimensions_better_than_style_R2':int((subset.R2_gain_vs_style>0).sum())}
global_row=agg[(agg.model=='training_mean')&(agg.group=='all')].iloc[0]
style_row=agg[(agg.model=='style_mean')&(agg.group=='all')].iloc[0]
et_row=agg[(agg.model=='ExtraTrees')&(agg.group=='all')].iloc[0]
result={'protocol':'Existing 5 repeated 3-fold style-stratified splits; training-fold mean and style mean; no retraining. Per-repeat full-250 OOF metrics then averages.',
        'MAE_relative_gain_vs_global_mean':float((global_row.MAE-et_row.MAE)/global_row.MAE),
        'MAE_relative_gain_vs_style_mean':float((style_row.MAE-et_row.MAE)/style_row.MAE),
        'R2_gain_vs_style_mean':float(et_row.R2-style_row.R2),'groups':summ,
        'style_baseline_comparison_uncertainty':comparison,
        'best5_by_R2':model_dim.sort_values('R2',ascending=False).head(5).reset_index().to_dict('records'),
        'worst5_by_R2':model_dim.sort_values('R2').head(5).reset_index().to_dict('records'),
        'note':'Mean-baseline Spearman omitted: each held-out fold has a constant prediction; pooling different fold constants gives artificial ranks. Group type means metadata type; targets are not raw binary labels.'}
agg.to_csv(OUT/'aggregate_metrics.csv',index=False,encoding='utf-8-sig')
detail.to_csv(OUT/'per_repeat_dimension_metrics.csv',index=False,encoding='utf-8-sig')
model_dim.to_csv(OUT/'ExtraTrees_dimension_summary.csv',encoding='utf-8-sig')
fold_df.to_csv(OUT/'same_split_baseline_metrics.csv',index=False,encoding='utf-8-sig')
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

fontpath=Path(r'C:\Windows\Fonts\msyh.ttc')
if fontpath.exists():
    font=FontProperties(fname=str(fontpath))
    plt.rcParams['font.family']=font.get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10.5,5.5))
rng=np.random.default_rng(20261025)
colors=['#2E6FAD','#D18432']
for ax,metric,title in zip(axes,['R2','Spearman'],['逐维折外 $R^2$','逐维 Spearman 秩相关']):
    data=[model_dim.loc[model_dim.type==kind,metric].to_numpy() for kind in ['numeric','categorical']]
    bp=ax.boxplot(data,positions=[1,2],widths=.45,patch_artist=True,showfliers=False,
                  medianprops={'color':'#222222','linewidth':1.8})
    for idx,(vals,col,patch) in enumerate(zip(data,colors,bp['boxes'])):
        patch.set_facecolor(col);patch.set_alpha(.15);patch.set_edgecolor(col)
        ax.scatter(idx+1+rng.uniform(-.12,.12,len(vals)),vals,s=28,color=col,alpha=.8,zorder=3,edgecolors='white',linewidths=.4)
    ax.axhline(0,color='#777777',linewidth=.8,linestyle='--')
    ax.set_xticks([1,2],['numeric\n26维','categorical\n24维'])
    ax.set_title(title,pad=12);ax.grid(axis='y',alpha=.16)
fig.suptitle('Extra Trees：感官预测能力在维度间差异明显',fontsize=15,y=.98)
fig.text(.5,.015,'每点为一个维度；先计算每次重复的250样本折外指标，再对5次重复取平均。\n标签为标准化感官汇总值；categorical组并非原始0/1分类准确率。',ha='center',va='bottom',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.10,1,.92])
fig.savefig(OUT/'逐维预测指标分布.png',dpi=180,bbox_inches='tight')
fig.savefig(OUT/'逐维预测指标分布.svg',bbox_inches='tight')
plt.close(fig)
print('AGGREGATE\n'+agg.to_string(index=False))
print(json.dumps(result,ensure_ascii=False,indent=2))
