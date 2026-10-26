"""Read-only source audit; writes analytical CSV/JSON/Markdown intermediates.
AI assistance: OpenAI Codex, 2026-10-26.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
DATA=next(ROOT.glob('2026年第七届*/2026*赛道A*'))/'附件'
x=pd.read_csv(DATA/'1.csv').set_index('beer')
c=pd.read_excel(DATA/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer').loc[x.index]
t=pd.read_excel(DATA/'8.xlsx',sheet_name='Sensory thresholds')
cfg=json.loads((OUT/'defect_thresholds.json').read_text(encoding='utf-8'))
rows=[]
for name,s in cfg['measured_defects'].items():
    f=s['field'];a=x[f];r=c[f];fac=.001 if s['declared_raw_unit']=='ug/L' else 1
    floor=float(a.min());mask=a.eq(floor)
    count=int((r*fac>s['threshold_mg_L']).sum());valid=int(r.notna().sum())
    zero=r.eq(0)
    paired_zero=zero & a.notna()
    corroborated=bool(paired_zero.any() and a[paired_zero].eq(floor).all())
    rows.append({'defect':name,'field':f,'valid_calibrated':valid,'above_threshold':count,
                 'above_fraction':count/valid,'threshold_mg_L':s['threshold_mg_L'],
                 'table1_floor':floor,'floor_count':int(mask.sum()),'floor_fraction':float(mask.mean()),
                 'floor_is_supported_by_zero_pairing':corroborated,
                 'calibrated_zero_count':int(zero.sum()),'table1_missing':int(a.isna().sum()),
                 'source_threshold_aggregate_rows':int(t.field_name_R.eq(f).sum()),
                 'underlying_threshold_observation_count':'not_provided',
                 'threshold_medium':'not_provided_in_attachment'})
audit=pd.DataFrame(rows)
audit.to_csv(OUT/'threshold_and_floor_prevalence.csv',index=False,encoding='utf-8-sig')
hits=t[t.chemical_name.astype(str).str.contains('acid',case=False,regex=False) | t.field_name_R.eq('acids_sum')]
hits[['field_name_R','chemical_name','unit']].to_csv(OUT/'measured_acid_fields.csv',index=False,encoding='utf-8-sig')
text=['# 阈值、下限状态与测量列补充核查','', '日期：2026-10-26。计数以附件8校准表为依据，按标注单位，不做1000倍修正。零值只是记录状态，不能解释为精确测量零。','',
'| 缺陷 | 超主阈值/有效数 | 比例 | 附件1最小值占比 | 下限替代佐证 |',
'|---|---:|---:|---:|---|']
for a in audit.itertuples():
    text.append(f'| {a.defect} | {a.above_threshold}/{a.valid_calibrated} | {a.above_fraction:.2%} | {a.floor_count}/250（{a.floor_fraction:.1%}） | {"有校准零值配对" if a.floor_is_supported_by_zero_pairing else "仅为最小观测值，不能认定检出下限"} |')
text += ['', 't2n较低阈值情景为249/249；主阈值几乎没有区分力。四项均以同尺度不增加为主，阈值判断为辅。',
'', '4VG有168/250个下限状态；Pils/Lager源谱的下限状态另在迁移表逐项标记。CS2虽非本次四项缺陷，最小值记录209/250=83.6%，显示朴素高斯对点质量的保真能力有局限。',
'', '## 阈值的聚合来源', '',
'附件8每项只有一行汇总记录，提供threshold_mean与threshold_median，没有各次阈值试验、底层记录条数、记录权重或介质（水/啤酒）的逐项信息。原研究补充工作簿同名表同样只有汇总字段。中位数直接取现成字段，不再对一行数据计算“中位数”；不能反推出底层有几条记录，也不能声称全部来自啤酒基质。',
'', 't2n主情景引用Santos等2003；低情景引用Saison等2009的lager flavour阈值，区别于odor终点。二者不是统计区间。详见原阈值报告与配置中的来源。',
'', '## 陈旧酒花相关测量', '',
'附件1及名称表不存在isovaleric acid/3-methylbutanoic acid字段。直接酸类只有乙酸、乳酸，另有总酸派生量。名称表中的Iso-alpha acids对应bitterness理化指标；名称含butyrate/isovalerate的多项是酯，不是相应游离酸。现有酸类没有足够特异性支持Stale hops代理约束，因此不新增代理化学保证。',
'', '## 下限和数值容差', '',
'对DMS、4VG经校准零值配对佐证的替代下限记录，要求候选不高于同一替代下限，单独报告“下限状态保持”；换算的替代浓度不等于真实浓度或实测检出限。对没有佐证的观测最小值不冒称检出限。',
'', '没有找到可用于本次四项验收的逐样本重复测量误差或校准不确定度，本版不设测量容差。计算最终采用边界投影与严格≤；这是一种保守设计口径，不是物理品评保证。']
(OUT/'阈值与下限补充核查.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(audit.to_string(index=False))
