"""Prepare auditable defect units, references and real-data compatibility checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
DATA=next(ROOT.glob('2026年第七届*/2026*赛道A*'))/'附件'
x=pd.read_csv(DATA/'1.csv').set_index('beer')
c=pd.read_excel(DATA/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer').loc[x.index]
t=pd.read_excel(DATA/'8.xlsx',sheet_name='Sensory thresholds').set_index('field_name_R')
fields={'DMS':'DMS','Diacetyl':'X2.3.butanedione','4VG':'X4VG','t2n':'X.E..2.nonenal'}
spec={}
for name,field in fields.items():
    row=t.loc[field]
    if name=='t2n':
        threshold=.0001;alternatives=[.00003,.0001];source='Santos et al. 2003 reference 0.1 ug/L; Saison et al. 2009 Table 2 sensitivity 0.03 ppb'
    else:
        factor=.001 if row.unit=='ug/L' else 1.
        threshold=float(row.threshold_median)*factor
        alternatives=sorted(set([threshold,float(row.threshold_mean)*factor]));source='8.xlsx Sensory thresholds; median main, mean alternative'
    spec[name]={'field':field,'declared_raw_unit':row.unit,'threshold_mg_L':threshold,
                'sensitivity_thresholds_mg_L':alternatives,'threshold_source':source,
                'table1_mapping':'log10(raw value in declared raw unit)' if name in ['DMS','4VG'] else 'mostly identity, exceptional rows require review',
                'qualification':'t2n absolute scale flagged by near-universal exceedance' if name=='t2n' else 'conditional on calibration and detection processing'}
config={'analysis_date':'2026-10-25','canonical_unit':'mg/L','measured_defects':spec,
        'recommended_primary_transfer_rule':'concentration_nonincrease_on_same_measurement_scale',
        'also_report':['excess_nonworsening','below_threshold'],
        'Stale_hops':{'measured_chemical':'absent','candidate_external_marker':'isovaleric acid / 3-methylbutanoic acid',
                      'not_equivalent_to':'ethyl_isovalerate / ethyl 3-methylbutyrate',
                      'status':'not_chemically_verifiable; weak sensory proxy only'},
        'sources':[
          {'id':'Santos2003','doi':'10.1016/S0021-9673(02)01396-1','url':'https://pubmed.ncbi.nlm.nih.gov/12580508/'},
          {'id':'Saison2009','doi':'10.1016/j.foodchem.2008.10.078','url':'https://www.agraria.com.br/extranet_2016/uploads/AgromalteArquivo/contribution_of_staling_compounds_to_the_aged_flavour_1601494914447.pdf','locator':'Table 2, journal p.1209; thresholds are indicative'},
          {'id':'BrewersAssociation2025','url':'https://cdn.brewersassociation.org/wp-content/uploads/2025/07/14095250/BAtech25-Off-Flavor-Isovaleric-201.pdf','locator':'p.1; isovaleric acid, hop degradation and matrix-dependent thresholds'}],
        'important':'No automatic 1000x rescaling; no missing-as-zero; no all-five sensory guarantee.'}
(OUT/'defect_thresholds.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
rows=[];summ={}
for name,field in fields.items():
    raw=c[field];obs=x[field]
    threshold=spec[name]['threshold_mg_L'];factor=.001 if t.loc[field,'unit']=='ug/L' else 1.
    for beer in x.index:
        r=raw.loc[beer];o=obs.loc[beer]
        flags=[]
        if pd.isna(r):flags.append('calibrated_value_missing')
        if pd.isna(o):flags.append('table1_value_missing_do_not_treat_as_zero')
        if pd.notna(r) and r==0:flags.append('calibrated_zero_is_not_proof_of_true_zero')
        consistent=None
        if pd.notna(r) and pd.notna(o) and r>0:
            expected=np.log10(r) if name in ['DMS','4VG'] else r
            consistent=bool(abs(o-expected)<1e-7)
            if not consistent:flags.append('table1_calibrated_discrepancy')
        rows.append({'beer':int(beer),'style':x.loc[beer,'tasting_category_fine'],'defect':name,'field':field,
                     'table1_value':None if pd.isna(o) else float(o),'calibrated_raw_value':None if pd.isna(r) else float(r),
                     'calibrated_mg_L':None if pd.isna(r) else float(r*factor),'threshold_mg_L':threshold,
                     'mapping_consistent_positive_pair':consistent,
                     'flags':' | '.join(flags)})
    mg=raw*factor
    summ[name]={'valid_calibrated':int(raw.notna().sum()),'table1_missing':int(obs.isna().sum()),
                'calibrated_zero':int(raw.eq(0).sum()),'above_main_threshold_under_declared_units':int((mg>threshold).sum()),
                'median_threshold_ratio':float(mg.median()/threshold)}
audit=pd.DataFrame(rows);audit.to_csv(OUT/'defect_concentration_audit.csv',index=False,encoding='utf-8-sig')
extra={'summary':summ,
       'specific_discrepancies':audit[audit['flags'].str.contains('discrepancy|missing')].replace({np.nan:None}).to_dict('records'),
       'isovaleric_acid_exact_measured_field_found':False,
       't2n_low_threshold_exceedance':int((c['X.E..2.nonenal']>.00003).sum())}
(OUT/'threshold_audit_summary.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summ,ensure_ascii=False,indent=2))
