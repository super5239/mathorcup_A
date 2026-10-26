"""Assemble a readable Q2 statistical report from completed saved results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
assert s['bootstrap_completed']==100
top=pd.read_csv(OUT/'top3_chemical_candidates.csv')
target=pd.read_csv(OUT/'target_stability_and_jaccard.csv').set_index('target')
mapping=pd.read_csv(OUT/'chemical_mapping_and_modules.csv').set_index('field')
imps=pd.read_csv(OUT/'all_factor_mean_permutation_importance.csv',index_col=0)
freq=pd.read_csv(OUT/'chemical_top3_frequencies.csv',index_col=0)
mfreq=pd.read_csv(OUT/'module_top3_frequencies.csv',index_col=0)
assert len(top)==150 and top.groupby('target')['rank'].apply(lambda v:sorted(v)==[1,2,3]).all()
assert top.field.isin(mapping[mapping.eligible_primary_top3].index).all()
assert freq.to_numpy().min()>=0 and freq.to_numpy().max()<=1+1e-9
assert np.allclose(freq.sum(axis=1),3)
assert mfreq.to_numpy().min()>=0 and mfreq.to_numpy().max()<=1+1e-9
assert all(mfreq.loc[r.target,r.module]+1e-9>=r.top3_frequency for r in top.itertuples())
aux=[]
for field,row in imps.iterrows():
    for rank,compound in enumerate(row.sort_values(ascending=False,kind='stable').index[:3],1):
        aux.append({'target':field,'rank':rank,'field':compound,'importance':float(row[compound]),
                    'eligible_primary_top3':bool(mapping.loc[compound,'eligible_primary_top3'])})
pd.DataFrame(aux).to_csv(OUT/'supplementary_all_factor_top3.csv',index=False,encoding='utf-8-sig')

lines=['# Q2网络、关键化合物与稳定性：统计主流程结果','',
       '日期：2026-10-25。已完成100次分层bootstrap、每列3次袋外置换、Extra Trees与稳健PLS对照。本文给出统计候选和稳定性，尚不能替代逐项化学机理核查。','',
       '## 网络与名称核验','',
       f'231个指标全部通过附件8名称映射，修复希腊字母的可逆乱码后，与附件2逐项对齐。对齐矩阵与附件1重算Spearman的最大差异为{s["graph"]["matrix_spearman_max_error"]:.2g}。没有使用附件6标签相关性筛选变量。','',
       '固定|rho|≥0.6阈值，得到231个节点、266条边（3条负相关边），Louvain划分为19个非单点模块与101个单点模块。101个孤立节点仍参与模型和统计，只为图的可读性未在左图绘制。阈值没有按输出结果调整。','',
       '![化合物网络与模块](化合物网络与模块.png)','',
       '网络仅用于描述和把每次名单映射到固定模块，不进入预测拟合。模块频率是固定划分下的名单命中率，并非模块划分本身在重采样中的稳定性。','',
       '## 评价方法','',
       '每次按22种风格分层有放回重采样250条训练记录，训练折中位数插补，未抽中的样本作为袋外评价集。两个模型使用同一重采样及同一置换顺序；置换重要性为MAE增加量，逐目标计算。保存全部训练/袋外编号，训练与袋外编号不交叉。','',
       'Extra Trees参数与Q1保持一致；对照为已有8成分稳健PLS。PLS是固定线性预测器，其置换预测以系数精确计算，并用显式置换预测校验；没有以系数大小冒充验证集置换重要性。','',
       '模型输入保留全部231列。主名单在217个单体化学候选中排序，排除14个派生、物理、混合或元素指标；完整因子Top-3另存supplementary_all_factor_top3.csv，不隐去模型对这些指标的依赖。','',
       '主名单按100次平均袋外置换MAE增量排序；入选频率是每次重拟合后进入前三的比例。模块每次最多命中一次。同模块可出现多个化合物，不强制分散；大模块命中机会较大。','',
       '逐列置换仍受共线性及不真实化学组合的影响，重采样不能自动消除这种偏差。重要性可以为负，不截断成零。重要性的2.5%和97.5%数值是重采样分布百分位，不称为因果效应置信区间。','',
       '## 稳定性概况','',
       f'- 50个目标均输出3个候选，共150条主结果。',
       f'- 三个候选入选频率均不低于50%的目标：{s["targets_with_all_top3_frequency_at_least_0_5"]}/50。',
       f'- 三个候选入选频率均不低于60%的目标：{s["targets_with_all_top3_frequency_at_least_0_6"]}/50。',
       f'- 两模型最终Top-3的Jaccard中位数：{s["median_final_Jaccard"]:.3f}；均值：{s["mean_final_Jaccard"]:.3f}。',
       f'- Q1平均R²低于0.1的目标：{s["targets_Q1_R2_below_0_1"]}/50，清单明确标记为探索性。','',
       '50%和60%仅作稳定性描述，不是显著性阈值。跨方法一致性与重采样稳定性不是同一件事，不因某个候选重复出现就把它称为真实驱动。','',
       '![Top3与模块稳定性](Top3与模块稳定性.png)','',
       '## 代表性目标','']
selected=['acidity','bitternes','alcohol','A_hops_citrus','A_esters_isoaa','diacetyl','dms','X4vg','t2n','stale_hops']
for name in selected:
    g=top[top.target==name].sort_values('rank');tt=target.loc[name]
    lines += [f'### {name}：Q1 R²={tt.Q1_R2:.3f}，跨方法Jaccard={tt.Jaccard_final:.2f}','',
              '| 排名 | 化合物 | 袋外MAE增量 | 入选频率 | 模块 | 模块命中频率 |','|---|---|---:|---:|---|---:|']
    for r in g.itertuples():
        lines.append(f'| {r.rank} | {r.chemical_name} | {r.mean_OOB_permutation_MAE_increase:.5f} | {r.top3_frequency:.0%} | {r.module} | {r.module_top3_frequency:.0%} |')
    lines += ['', '上表是模型关联候选，不能仅凭化合物名称判定直接感官机制。低预测目标尤其不能据此作干预保证。','']
lines += ['## 全50维主名单','',
          '| 感官目标 | Q1 R² | Top-1（入选率） | Top-2（入选率） | Top-3（入选率） | 最终Jaccard |',
          '|---|---:|---|---|---|---:|']
for name in target.index:
    g=top[top.target==name].sort_values('rank');cells=[f'{r.field}（{r.top3_frequency:.0%}）' for r in g.itertuples()]
    lines.append(f'| {name} | {target.loc[name,"Q1_R2"]:.3f} | '+ ' | '.join(cells)+f' | {target.loc[name,"Jaccard_final"]:.2f} |')
lines += ['', '## 对下一阶段的约束','',
          '保留低稳定性或低预测可靠性目标，明确其探索性，不为得到更好数字追加模型、重选图阈值或重抽随机种子。未做的组置换、SHAP和图阈值敏感性不写成已经验证。',
          '第三问只能把稳定名单作为候选化学关系与模块约束的辅助，不能把置换重要性当成因果导数。DMS等缺陷仍优先使用真实单位浓度阈值，Stale hops的弱感官代理不能构成严格保证。','',
          '本次统计完成后，下一步按较高预测可靠性目标和实际Top-3逐项核对化学作用、单位及阈值；尚未完成的机理文献核查不得写作已证实。']
(OUT/'Q2统计主流程结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(s,ensure_ascii=False,indent=2))
print(top[top.target.isin(['acidity','bitternes','alcohol','dms','stale_hops'])][['target','field','top3_frequency','module_top3_frequency']].to_string(index=False))
