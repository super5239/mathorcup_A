"""v3.1 第2项：Q2结果表改用统一预测支持度 + 解释稳定性 + 高度相关备选。

只重组已有100次bootstrap结果（analysis/q2/），不重训模型、不重做bootstrap。
规则见 docs/解题思路.md 第1.6节与问题2。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
Q2 = ROOT / 'analysis' / 'q2'
OUT = ROOT / 'results' / 'q2'
OUT.mkdir(parents=True, exist_ok=True)
ALT_RHO, ALT_MAX = 0.8, 3

support = pd.read_csv(ROOT / 'results' / 'q1' / 'q1_prediction_support_tiers.csv', encoding='utf-8-sig', index_col=0)
old = pd.read_csv(Q2 / 'reliability_tiers.csv', encoding='utf-8-sig').set_index('target')
top3 = pd.read_csv(Q2 / 'top3_with_MC_frequency_intervals.csv', encoding='utf-8-sig')
mapping = pd.read_csv(Q2 / 'chemical_mapping_and_modules.csv', encoding='utf-8-sig').set_index('field')
rho = pd.read_csv(ROOT / '2.csv', index_col=0)
assert mapping.matrix_name.isin(rho.index).all()

# ---------- 解释稳定性（不含 skill 条件；skill 在预测支持度维度单列） ----------
imp_pos = top3.groupby('target').mean_OOB_permutation_MAE_increase.apply(lambda s: bool((s > 0).all()))
min_freq = top3.groupby('target').top3_frequency.min()
stab = pd.Series('探索性', index=old.index)
stab[old.max_non_singleton_module_frequency >= 0.7] = '模块线索'
stab[imp_pos & (min_freq >= 0.5)] = '单体稳定'

# ---------- 高度相关备选：同模块且 |rho|>=0.8，最多3个 ----------
# 非单体化学指标（理化/派生/混合，如 kcalperc）不作为化合物备选，单列为“派生相关指标”
def correlated_peers(field, eligible):
    mod = mapping.loc[field, 'module']
    peers = mapping.index[(mapping.module == mod) & (mapping.index != field)
                          & (mapping.eligible_primary_top3 == eligible)]
    if len(peers) == 0:
        return ''
    r = rho.loc[mapping.loc[field, 'matrix_name'], mapping.loc[peers, 'matrix_name']]
    r.index = peers
    r = r[r.abs() >= ALT_RHO]
    r = r.reindex(r.abs().sort_values(ascending=False).index)[:ALT_MAX]
    return '; '.join(f'{f}({"+" if v > 0 else "−"}{abs(v):.2f})' for f, v in r.items())


top3['high_corr_alternatives'] = top3.field.map(lambda f: correlated_peers(f, True))
top3['derived_correlated_indicators'] = top3.field.map(lambda f: correlated_peers(f, False))
top3['module_size'] = top3.module.map(mapping.module.value_counts())

# ---------- 汇总表 ----------
t = pd.DataFrame(index=old.index)
t['type'] = support.loc[t.index, 'type']
t['R2_mean_oof'] = support.loc[t.index, 'R2_mean_oof']
t['support_tier'] = support.loc[t.index, 'support_tier']
t['MSE_skill_vs_style'] = support.loc[t.index, 'MSE_skill_vs_style']
t['skill_flag'] = np.where(t.MSE_skill_vs_style > 0, '', '未优于风格均值')
t['interpretation_stability'] = stab
t['mean_top3_freq'] = old.mean_final_top3_selection_frequency
t['min_top3_freq'] = min_freq
t['all_top3_importance_positive'] = imp_pos
t['max_non_singleton_module_freq'] = old.max_non_singleton_module_frequency
t['old_combined_tier_historical'] = old.tier
t['top3'] = top3.sort_values('rank').groupby('target').field.apply(' | '.join)


def usage(r):
    if r.support_tier == 'C':
        return '探索性（预测支持度C，不支撑主要化学结论）'
    if r.skill_flag:
        return '谨慎：未优于风格均值'
    return {'单体稳定': '可重点报告单体候选', '模块线索': '报告模块线索', '探索性': '仅列候选'}[r.interpretation_stability]


t['usage'] = t.apply(usage, axis=1)
order = {'A': 0, 'B': 1, 'C': 2}
t = t.sort_values(['support_tier', 'R2_mean_oof'], key=lambda s: s.map(order) if s.name == 'support_tier' else -s)
t.to_csv(OUT / 'q2_v31_target_summary.csv', encoding='utf-8-sig')

cols = ['target', 'rank', 'field', 'chemical_name', 'origin', 'module', 'module_size',
        'mean_OOB_permutation_MAE_increase', 'top3_frequency', 'frequency_MC_Wilson_lower', 'frequency_MC_Wilson_upper',
        'module_top3_frequency', 'observed_spearman_descriptive', 'high_corr_alternatives',
        'derived_correlated_indicators']
detail = top3[cols].merge(t[['support_tier', 'skill_flag', 'interpretation_stability', 'usage']],
                          left_on='target', right_index=True)
detail.to_csv(OUT / 'q2_v31_top3_detail.csv', index=False, encoding='utf-8-sig')

cross = pd.crosstab(t.support_tier, t.interpretation_stability)
summary = {
    'support_tier_counts': t.support_tier.value_counts().sort_index().to_dict(),
    'interpretation_stability_counts': t.interpretation_stability.value_counts().to_dict(),
    'crosstab': cross.to_dict(),
    'usage_counts': t.usage.value_counts().to_dict(),
    'n_top3_with_alternatives': int((detail.high_corr_alternatives != '').sum()),
    'n_top3_with_derived_indicators': int((detail.derived_correlated_indicators != '').sum()),
    'n_top3_total': len(detail),
    'stability_rule': '单体稳定: 3个候选平均置换重要性均>0且入选率均>=50%; 模块线索: 非单点模块最高命中率>=70%; 其余探索性',
    'alternatives_rule': f'same module, |Spearman rho| >= {ALT_RHO} (2.csv), at most {ALT_MAX}; non-compound indicators listed separately as derived_correlated_indicators',
}
json.dump(summary, open(OUT / 'summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print(t[['support_tier', 'skill_flag', 'interpretation_stability', 'top3']].to_string())
