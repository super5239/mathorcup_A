"""Independent saved-output integrity, missingness and constraint checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
DATA=next(ROOT.glob('2026年第七届*/2026*赛道A*'))/'附件'
x=pd.read_csv(DATA/'1.csv').set_index('beer')
cfg=json.loads((OUT/'defect_thresholds.json').read_text(encoding='utf-8'))
g=pd.read_csv(OUT/'generated_231_coordinates.csv')
m=pd.read_csv(OUT/'migration_231_coordinates.csv')
checks=pd.read_csv(OUT/'migration_metrics_all_weights.csv')
defects=pd.read_csv(OUT/'migration_defect_checks.csv')
fit=joblib.load(OUT/'style_gaussians.joblib')
fields=fit['fields']
assert len(g)==900 and len(m)==66
assert g.groupby('style').size().eq(300).all()
assert g.candidate_id.is_unique
assert np.isfinite(g[fields]).all().all()
assert (g[fields].to_numpy()>=fit['low']-1e-12).all()
assert (g[fields].to_numpy()<=fit['high']+1e-12).all()
count=0;missing=0
for row in m.itertuples(index=False):
    values=dict(zip(m.columns,row));beer=int(values['beer'])
    for name,spec in cfg['measured_defects'].items():
        field=spec['field'];a=x.loc[beer,field];b=values[field]
        if pd.notna(a):
            assert b<=a,(beer,values['alpha'],field,a,b)
            count+=1
        else:missing+=1
assert count==252 and missing==12
assert not checks[checks.beer.isin([119,28])].four_defects_evaluable_on_table1.any()
assert checks[~checks.beer.isin([119,28])].all_four_nonincrease.all()
assert not checks.Stale_hops_chemically_verified.any()
assert defects[defects.source_at_supported_floor].floor_status_retained.all()
front=joblib.load(OUT/'sensory_forward_model.joblib')
pred=front['model'].predict(front['imputer'].transform(g.iloc[:5][fields]))
saved=pd.read_csv(OUT/'generated_sensory_standardized.csv')
assert np.allclose(pred,saved.iloc[:5][front['targets']].to_numpy(),rtol=1e-10,atol=1e-10)
splits=json.loads((OUT/'generator_holdout_splits.json').read_text(encoding='utf-8'))
assert all(not set(a['train_beer'])&set(a['test_beer']) for a in splits)
assert sorted(b for a in splits for b in a['test_beer'])==sorted(x.index.tolist())
result={'saved_candidates':len(g),'migration_candidates':len(m),'observed_defect_pairs_checked':count,
        'missing_source_defect_pairs_not_certified':missing,'floor_caps_checked':int(defects.source_at_supported_floor.sum()),
        'portable_models_loaded':True,'forward_prediction_roundtrip_verified':True,'holdout_partition_verified':True,
        'interpretation':'File/algorithm integrity checks, not proof of chemical realism or sensory safety.'}
(OUT/'saved_output_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
