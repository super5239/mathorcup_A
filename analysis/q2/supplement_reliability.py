"""Reorganize saved Q1/Q2 outputs; no predictive refitting or new bootstrap.
AI assistance: OpenAI Codex, 2026-10-26.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
q=pd.read_csv(ROOT/'analysis/A_feasibility/q1_metrics/ExtraTrees_dimension_summary.csv',index_col=0)
top=pd.read_csv(OUT/'top3_chemical_candidates.csv')
mapping=pd.read_csv(OUT/'chemical_mapping_and_modules.csv').set_index('field')
counts=mapping.groupby('module').size()
fields=list(mapping.index);eligible=np.flatnonzero(mapping.eligible_primary_top3.to_numpy())
targets=list(pd.read_csv(OUT/'all_factor_mean_permutation_importance.csv',index_col=0).index)
q=q.loc[targets]

def wilson(p,n=100):
    z=1.959963984540054;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-width,center+width

rows=[]
for name in targets:
    a=top[top.target==name].sort_values('rank')
    probs=a.top3_frequency.to_numpy();lo,hi=wilson(probs)
    modules=a[['module','module_top3_frequency']].drop_duplicates('module')
    nonsingle=modules[modules.module.map(counts)>1]
    maxmodule=float(nonsingle.module_top3_frequency.max()) if len(nonsingle) else 0.
    skill=float(q.loc[name,'MSE_skill_vs_style'])
    A=(skill>0 and probs.min()>=.5-1e-10 and (a.mean_OOB_permutation_MAE_increase>0).all())
    B=(not A and skill>0 and maxmodule>=.7-1e-10)
    tier='A' if A else ('B' if B else 'C')
    reason=('相对风格均值有MSE提升；三个候选均有至少50%入选率' if A else
            ('相对风格均值有MSE提升；至少一个非单点模块命中率≥70%' if B else
             ('未优于风格均值' if skill<=0 else '单体及非单点模块未达到预设描述门槛')))
    rows.append({'target':name,'tier':tier,'Q1_R2':float(q.loc[name,'R2']),
        'Q1_MSE_skill_vs_style':skill,'Q1_relative_MAE_gain_vs_style':float(q.loc[name,'relative_MAE_gain_vs_style']),
        'mean_final_top3_selection_frequency':float(probs.mean()),'min_final_top3_selection_frequency':float(probs.min()),
        'max_non_singleton_module_frequency':maxmodule,
        'n_top3_Wilson_intervals_covering_0_5':int(((lo<=.5)&(hi>=.5)).sum()),'reason':reason})
    for j,(idx,r) in enumerate(a.iterrows()):
        top.loc[idx,'frequency_MC_Wilson_lower']=lo[j]
        top.loc[idx,'frequency_MC_Wilson_upper']=hi[j]
        top.loc[idx,'tier']=tier
df=pd.DataFrame(rows)
df.to_csv(OUT/'reliability_tiers.csv',index=False,encoding='utf-8-sig')
top.to_csv(OUT/'top3_with_MC_frequency_intervals.csv',index=False,encoding='utf-8-sig')

cp=np.load(OUT/'bootstrap_checkpoint.npz')
imps={key:cp[key].mean(0) for key in ['extra_trees','pls']}
overlap=[]
def jac(a,b):return len(a&b)/len(a|b)
for k in [3,5,10]:
    ranks={key:eligible[np.argsort(-v[:,eligible],axis=1,kind='stable')[:,:k]] for key,v in imps.items()}
    for j,name in enumerate(targets):
        a=set(ranks['extra_trees'][j]);b=set(ranks['pls'][j])
        ma=set(mapping.iloc[list(a)].module);mb=set(mapping.iloc[list(b)].module)
        overlap.append({'target':name,'top_k':k,'chemical_Jaccard':jac(a,b),'module_Jaccard':jac(ma,mb),
                        'ET_unique_modules':len(ma),'PLS_unique_modules':len(mb)})
ov=pd.DataFrame(overlap);ov.to_csv(OUT/'relaxed_cross_method_jaccard.csv',index=False,encoding='utf-8-sig')
assoc={name:float(spearmanr(df[col],df.mean_final_top3_selection_frequency).statistic)
       for name,col in [('skill_vs_selection','Q1_MSE_skill_vs_style'),('R2_vs_selection','Q1_R2')]}
summary={'date':'2026-10-26','n_dimensions':len(df),'tier_counts':df.tier.value_counts().to_dict(),
    'descriptive_Spearman_across_dimensions':assoc,
    'jaccard_medians':ov.groupby('top_k')[['chemical_Jaccard','module_Jaccard']].median().to_dict('index'),
    'jaccard_means':ov.groupby('top_k')[['chemical_Jaccard','module_Jaccard']].mean().to_dict('index'),
    'retrained_models':0,'style_confounding_sensitivity_completed':False,
    'tier_rule':'A: skill>0, min chemical frequency>=.5, positive average importance for all 3; B: not A, skill>0, at least one non-singleton module frequency>=.7; C: remaining',
    'qualification':'Post-analysis descriptive prioritization, not independent validation or causal evidence; Wilson intervals quantify 100-run conditional Monte Carlo precision only.'}
(OUT/'reliability_supplement_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
colors={'A':'#168377','B':'#d28b20','C':'#89939c'}
fig,axes=plt.subplots(1,2,figsize=(12.2,5.4),sharey=True)
for ax,col,label in zip(axes,['Q1_MSE_skill_vs_style','Q1_R2'],['Q1 相对风格均值的 MSE skill','Q1 逐维 R²']):
    for tier in ['A','B','C']:
        a=df[df.tier==tier]
        ax.scatter(a[col],a.mean_final_top3_selection_frequency,c=colors[tier],s=42,alpha=.85,label=f'{tier} 档（{len(a)}维）',edgecolors='white',linewidth=.5)
    ax.axvline(0,color='#b7bdc4',lw=.8);ax.set_xlabel(label);ax.set_ylim(0,.9);ax.grid(alpha=.16)
    for name in ['acidity','alcohol','stale_hops']:
        a=df[df.target==name].iloc[0]
        ax.annotate(name,(a[col],a.mean_final_top3_selection_frequency),
                    xytext=(-6,6) if name=='acidity' else (5,6),
                    ha='right' if name=='acidity' else 'left',textcoords='offset points',fontsize=8)
    rho=assoc['skill_vs_selection' if col.startswith('Q1_MSE') else 'R2_vs_selection']
    ax.set_title(f'跨50个维度的描述性 Spearman ρ={rho:.3f}',fontsize=11)
axes[0].set_ylabel('最终 Top-3 的平均 bootstrap 入选率')
axes[1].legend(frameon=False,loc='upper left')
fig.suptitle('可预测性与关键因素稳定性：使用已有折外及重采样结果',fontsize=13)
fig.text(.5,.025,'分档用于安排解释优先级；没有排除风格混杂，也不表示因果可信等级。',ha='center',fontsize=9,color='#59636d')
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/'可预测性与Top3稳定性.png',dpi=190);fig.savefig(OUT/'可预测性与Top3稳定性.svg');plt.close(fig)

lines=['# Q2可靠性分档与已有结果补充','', '日期：2026-10-26。本次重训模型0个；没有重做bootstrap或风格残差化。','',
'## 描述性分档规则','',
'分档在已看到总体稳定性结果后制定，用于论文解释和人工核查的优先级，不称为预注册检验、显著性或因果证据。',
'', '- A：Q1相对风格均值MSE skill>0；最终三个候选平均置换重要性均为正；三个入选率均≥50%。可以重点报告稳定的模型关联，仍需化学核查。',
'- B：未入A；skill>0；最终候选涉及的至少一个非单点模块命中率≥70%。报告模块线索，不把整个Top-3当作稳定发现。',
'- C：其余目标。保留候选清单，但不据此下关键化学因素结论。',
'', '这些是描述门槛，不是客观置信度。大模块命中机会更高，模块命中不等于社区划分稳定；B不能解释为已经验证共线替换。','',
f'本次分档：'+ '、'.join(f'{t}档{int((df.tier==t).sum())}维' for t in ['A','B','C'])+'。','',
'![可预测性与稳定性](可预测性与Top3稳定性.png)','',
f'跨50维的描述性Spearman：MSE skill对入选率={assoc["skill_vs_selection"]:.3f}，R²对入选率={assoc["R2_vs_selection"]:.3f}。维度彼此相关，不作50个独立重复的显著性检验；正相关也不能证明弱预测是低稳定性的唯一原因。','',
'## 跨模型一致性','', '| Top-k | 化合物Jaccard中位数 | 模块Jaccard中位数 | 化合物均值 | 模块均值 |','|---|---:|---:|---:|---:|']
for k,g in ov.groupby('top_k'):
    lines.append(f'| {k} | {g.chemical_Jaccard.median():.3f} | {g.module_Jaccard.median():.3f} | {g.chemical_Jaccard.mean():.3f} | {g.module_Jaccard.mean():.3f} |')
lines += ['', '比较对象为Extra Trees与8成分PLS，均按已有袋外置换MAE增量排序，在同一217个单体化学候选内截取Top-k。模块Jaccard先对各名单去重映射到固定模块，再求交并比。模块压缩与大模块本身会提高重叠，不能单独证实共线替换。','',
'## 入选率的计算精度','',
'100次重采样下，p=50%的二项Monte Carlo标准误约5个百分点。表中增加Wilson区间，仅近似描述给定数据和协议下重复100次的计算精度；不是未知真实化合物效应或总体入选率的置信区间。50%边界附近的A档应降格措辞。','',
'## 逐维清单','', '| 目标 | 档 | Q1 skill | 平均入选率 | 最低入选率 | 非单点模块最高命中率 |','|---|---|---:|---:|---:|---:|']
for a in df.sort_values(['tier','mean_final_top3_selection_frequency'],ascending=[True,False]).itertuples():
    lines.append(f'| {a.target} | {a.tier} | {a.Q1_MSE_skill_vs_style:.3f} | {a.mean_final_top3_selection_frequency:.1%} | {a.min_final_top3_selection_frequency:.1%} | {a.max_non_singleton_module_frequency:.1%} |')
lines += ['', '## 尚未解决的混杂','',
'尚未进行风格内置换、风格残差化或模块整体置换。酸度的乳酸候选与酚类候选须分开解释；酚类可能是发酵/风格共现标志物。A、B档也不能因此写成已经排除风格代理效应。酒精感虽R²高，但相对风格均值skill为负，按本规则不列入可强调的增量解释结论。']
(OUT/'Q2可靠性补充.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
