"""Style shrinkage Gaussian baseline and conservative bounded migration.
AI assistance: OpenAI Codex, 2026-10-26. See 首版运行协议.md.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
import joblib

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
DATA=next(ROOT.glob('2026年第七届*/2026*赛道A*'))/'附件'
SEED=20261026;NGEN=300;STYLES=['Tripel','Kriek','Lambic'];ALPHAS=[0,.1,.3,1,3,10]
threadpool_limits(2)
x=pd.read_csv(DATA/'1.csv').set_index('beer')
y=pd.read_csv(DATA/'4.csv').set_index('beer').loc[x.index]
X=x.drop(columns=['beer_id','tasting_category_fine']);Y=y.drop(columns=['beer_id','tasting_category_fine'])
fields=list(X);targets=list(Y);labels=x.tasting_category_fine.to_numpy()
config=json.loads((OUT/'defect_thresholds.json').read_text(encoding='utf-8'))
floorinfo=pd.read_csv(OUT/'threshold_and_floor_prevalence.csv').set_index('defect')
quality=pd.read_csv(ROOT/'analysis/A_feasibility/q1_metrics/ExtraTrees_dimension_summary.csv',index_col=0).loc[targets]
meta=pd.read_csv(ROOT/'analysis/A_feasibility/day1/sensory_metadata.csv',index_col=0).loc[targets]
eligible=pd.read_csv(ROOT/'analysis/q2/chemical_mapping_and_modules.csv')
chemical_fields=eligible.loc[eligible.eligible_primary_top3,'field'].tolist()

class GaussianStyles:
    def fit(self,A,styles):
        self.imp=SimpleImputer(strategy='median');B=self.imp.fit_transform(np.asarray(A,dtype=float))
        self.scaler=StandardScaler().fit(B);self.Z=self.scaler.transform(B)
        self.low=B.min(axis=0);self.high=B.max(axis=0)
        self.models={'__all__':LedoitWolf().fit(self.Z)}
        for style in np.unique(styles):
            self.models[style]=LedoitWolf().fit(self.Z[styles==style])
        self.roots={}
        for key,m in self.models.items():
            v,Q=np.linalg.eigh(m.covariance_)
            self.roots[key]=Q*np.sqrt(np.maximum(v,0))[None,:]
        return self
    def sample(self,key,normals):
        m=self.models[key]
        B=self.scaler.inverse_transform(normals@self.roots[key].T+m.location_)
        flag=(B<self.low)|(B>self.high)
        return np.clip(B,self.low,self.high),float(flag.mean())
    def transform(self,A):return self.scaler.transform(self.imp.transform(np.asarray(A,dtype=float)))

def corr(A):
    B=A-A.mean(0);den=np.sqrt((B*B).sum(0))
    C=(B.T@B)/np.maximum(np.outer(den,den),1e-15)
    return C,den>1e-12

# The split definition is fixed and independent of output quality.
foldrows=[];foldlogs=[];centered={};cvgen=[]
for fold,(train,test) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(X,labels)):
    fit=GaussianStyles().fit(X.iloc[train],labels[train]);rng=np.random.default_rng(SEED+fold)
    foldlogs.append({'fold':fold,'train_beer':X.index[train].tolist(),'test_beer':X.index[test].tolist()})
    for method in ['style_conditional','unconditional']:centered[method]=[[],[]]
    for style in STYLES:
        at=test[labels[test]==style];testz=fit.transform(X.iloc[at])
        trainz=fit.Z[labels[train]==style];noise=rng.normal(size=(NGEN,len(fields)))
        realnn=cdist(testz,trainz).min(axis=1).mean()/np.sqrt(len(fields))
        for method,key in [('style_conditional',style),('unconditional','__all__')]:
            B,cliprate=fit.sample(key,noise);Z=fit.transform(B)
            dnn=cdist(Z,trainz).min(axis=1).mean()/np.sqrt(len(fields))
            floor=float(X.iloc[train]['X4VG'].min())
            foldrows.append({'fold':fold,'style':style,'method':method,'n_train_style':len(trainz),'n_test_style':len(at),
                'mean_coordinate_MAE_standardized':float(abs(Z.mean(0)-testz.mean(0)).mean()),
                'mean_coordinate_Wasserstein_standardized':float(np.mean([wasserstein_distance(Z[:,j],testz[:,j]) for j in range(len(fields))])),
                'synthetic_nearest_train_distance':float(dnn),'real_holdout_nearest_train_distance':float(realnn),
                'synthetic_to_real_NN_ratio':float(dnn/realnn),'range_projection_fraction':cliprate,
                'synthetic_4VG_floor_fraction':float((B[:,fields.index('X4VG')]==floor).mean()),
                'holdout_4VG_floor_fraction':float(X.iloc[at]['X4VG'].eq(floor).mean())})
            centered[method][0].append(Z-Z.mean(0));centered[method][1].append(testz-testz.mean(0))
    for method,(aa,bb) in centered.items():
        Cg,vg=corr(np.vstack(aa));Cr,vr=corr(np.vstack(bb));valid=vg&vr
        ids=np.where(valid)[0];i,j=np.triu_indices(len(ids),1)
        rmse=float(np.sqrt(np.mean((Cg[np.ix_(ids,ids)][i,j]-Cr[np.ix_(ids,ids)][i,j])**2)))
        cvgen.append({'fold':fold,'method':method,'pooled_within_style_correlation_RMSE':rmse,
                      'valid_coordinates':int(valid.sum()),'n_holdout':sum(len(v) for v in bb)})
    print(f'Completed generator holdout fold {fold+1}/3',flush=True)
pd.DataFrame(foldrows).to_csv(OUT/'generator_holdout_metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(cvgen).to_csv(OUT/'generator_holdout_correlation.csv',index=False,encoding='utf-8-sig')
(OUT/'generator_holdout_splits.json').write_text(json.dumps(foldlogs,ensure_ascii=False,indent=2),encoding='utf-8')

# Final design fit, not an out-of-fold predictor evaluation.
fit=GaussianStyles().fit(X,labels);rng=np.random.default_rng(SEED+100)
generated=[];descriptive=[]
for style in STYLES:
    B,cliprate=fit.sample(style,rng.normal(size=(NGEN,len(fields))))
    df=pd.DataFrame(B,columns=fields);df.insert(0,'style',style)
    df.insert(0,'candidate_id',[f'{style}_{i+1:03}' for i in range(NGEN)])
    generated.append(df)
    real=fit.Z[labels==style];syn=fit.transform(B)
    c1,v1=corr(real);c2,v2=corr(syn);idx=np.where(v1&v2)[0];i,j=np.triu_indices(len(idx),1)
    nn=cdist(syn,real).min(1)/np.sqrt(len(fields))
    descriptive.append({'style':style,'n_real':len(real),'n_generated':NGEN,
        'LedoitWolf_shrinkage':float(fit.models[style].shrinkage_),
        'mean_coordinate_Wasserstein_standardized':float(np.mean([wasserstein_distance(syn[:,j],real[:,j]) for j in range(len(fields))])),
        'correlation_RMSE_in_sample':float(np.sqrt(np.mean((c1[np.ix_(idx,idx)][i,j]-c2[np.ix_(idx,idx)][i,j])**2))),
        'mean_nearest_real_distance_standardized':float(nn.mean()),'exact_training_copy_count':int((nn<1e-12).sum()),
        'range_projection_fraction':cliprate,
        'real_4VG_floor_fraction':float(X.loc[labels==style,'X4VG'].eq(X.X4VG.min()).mean()),
        'synthetic_4VG_floor_fraction':float((B[:,fields.index('X4VG')]==X.X4VG.min()).mean())})
gen=pd.concat(generated,ignore_index=True)
gen.to_csv(OUT/'generated_231_coordinates.csv',index=False,encoding='utf-8-sig')
gen[['candidate_id','style']+chemical_fields].to_csv(OUT/'generated_217_chemical_coordinates.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(descriptive).to_csv(OUT/'generator_final_descriptive_metrics.csv',index=False,encoding='utf-8-sig')
joblib.dump({'imputer':fit.imp,'scaler':fit.scaler,'low':fit.low,'high':fit.high,
             'style_models':fit.models,'fields':fields},OUT/'style_gaussians.joblib')

def mg(name,value):
    if not np.isfinite(value):return np.nan
    return float(10.**value*(.001 if name=='DMS' else 1)) if name in ['DMS','4VG'] else float(value)

audit=pd.read_csv(OUT/'defect_concentration_audit.csv').set_index(['beer','defect'])
targetmodel=fit.models['Tripel'];P=targetmodel.precision_;mu=targetmodel.location_;p=len(fields)
low=(fit.low-fit.scaler.mean_)/fit.scaler.scale_
high=(fit.high-fit.scaler.mean_)/fit.scaler.scale_
sources=X.loc[labels=='Pils/Lager'];srcz=fit.transform(sources)
realtripel=fit.Z[labels=='Tripel']
rows=[];defrows=[];migrations=[]
for bi,(beer,source) in enumerate(sources.iterrows()):
    z0=srcz[bi];hi=high.copy();active=[];known=[]
    for name,spec in config['measured_defects'].items():
        field=spec['field'];j=fields.index(field);original=source[field]
        if pd.notna(original):
            hi[j]=min(hi[j],(original-fit.scaler.mean_[j])/fit.scaler.scale_[j]);known.append(name)
        if bool(floorinfo.loc[name,'floor_is_supported_by_zero_pairing']) and original==floorinfo.loc[name,'table1_floor']:
            active.append(name)
    source_md=float(np.sqrt(max((z0-mu)@P@(z0-mu),0)/p))
    source_nn=float(cdist(z0[None],realtripel).min()/np.sqrt(p))
    for alpha in ALPHAS:
        if alpha==0:
            z=z0.copy();success=True;nit=0
        else:
            def obj(z):
                diff=z-z0;delta=z-mu
                return .5*(diff@diff+alpha*(delta@P@delta)),diff+alpha*(P@delta)
            opt=minimize(obj,np.clip(z0,low,hi),method='L-BFGS-B',jac=True,
                         bounds=list(zip(low,hi)),options={'maxiter':1200,'ftol':1e-12,'gtol':1e-8,'maxls':40})
            z=np.clip(opt.x,low,hi);success=bool(opt.success);nit=int(opt.nit)
        B=np.clip(fit.scaler.inverse_transform(z[None])[0],fit.low,fit.high)
        # Projection in the original stored coordinate eliminates conversion roundoff.
        for name in known:
            field=config['measured_defects'][name]['field'];j=fields.index(field)
            B[j]=min(B[j],source[field])
        z=fit.transform(B[None])[0]
        delta=z-mu;md=float(np.sqrt(max(delta@P@delta,0)/p))
        nn=float(cdist(z[None],realtripel).min()/np.sqrt(p))
        changes=[];passes=[];below=[];excess=[];flags=[]
        for name,spec in config['measured_defects'].items():
            f=spec['field'];j=fields.index(f);a=source[f];b=B[j]
            aa=mg(name,a);bb=mg(name,b);threshold=spec['threshold_mg_L']
            valid=bool(np.isfinite(aa) and aa>=0)
            isfloor=name in active
            passed=bool(b<=a) if valid else None
            if valid:
                passes.append(passed);below.append(bool(bb<=threshold))
                excess.append(bool(max(bb/threshold-1,0)<=max(aa/threshold-1,0)))
                changes.append(float((b-a)/fit.scaler.scale_[j]))
            flag=str(audit.loc[(beer,name),'flags'])
            if flag!='nan':flags.append(name+':'+flag)
            defrows.append({'beer':int(beer),'alpha':alpha,'defect':name,
                'source_table1':None if pd.isna(a) else float(a),'candidate_table1':float(b),
                'source_mg_L_or_floor_surrogate':aa,'candidate_mg_L_or_floor_surrogate':bb,
                'threshold_mg_L':threshold,'concentration_nonincrease':passed,
                'threshold_pass':bool(bb<=threshold) if valid else None,
                'source_at_supported_floor':isfloor,'floor_status_retained':bool(b<=a) if isfloor else None,
                'interpretation':'floor_surrogate_cap_not_exact_measured_concentration' if isfloor else ('same_table1_coordinate' if valid else 'missing_source_not_certifiable'),
                'source_quality_flags':flag if flag!='nan' else '',
                't2n_low_scenario_pass':bool(bb<=.00003) if name=='t2n' and valid else None})
        complete=len(known)==4
        rows.append({'beer':int(beer),'alpha':alpha,'primary_alpha':alpha==1,'optimizer_success':success,'optimizer_iterations':nit,
            'source_target_Mahalanobis':source_md,'target_Mahalanobis':md,
            'target_Mahalanobis_relative_reduction':1-md/source_md,
            'source_nearest_real_Tripel_distance':source_nn,'nearest_real_Tripel_distance':nn,
            'nearest_real_Tripel_relative_reduction':1-nn/source_nn,
            'source_change_RMS_standardized':float(np.sqrt(np.mean((z-z0)**2))),
            'mean_defect_change_standardized':float(np.mean(changes)),
            'maximum_defect_change_standardized':float(max(changes)),
            'four_defects_evaluable_on_table1':complete,
            'all_observed_defects_nonincrease':bool(all(passes)),
            'all_four_nonincrease':bool(complete and all(passes)),
            'all_four_excess_nonworsening':bool(complete and all(excess)),
            'all_four_below_threshold':bool(complete and all(below)),
            'n_floor_defects':len(active),'source_quality_flags':' | '.join(flags),
            'Stale_hops_chemically_verified':False})
        record=dict(zip(fields,B));record.update({'beer':int(beer),'alpha':alpha,'source_style':'Pils/Lager','target_style':'Tripel'})
        migrations.append(record)
    print(f'Migration source {bi+1}/{len(sources)} complete, beer {beer}',flush=True)
mr=pd.DataFrame(rows);dr=pd.DataFrame(defrows);mdf=pd.DataFrame(migrations)
mr.to_csv(OUT/'migration_metrics_all_weights.csv',index=False,encoding='utf-8-sig')
dr.to_csv(OUT/'migration_defect_checks.csv',index=False,encoding='utf-8-sig')
mdf.to_csv(OUT/'migration_231_coordinates.csv',index=False,encoding='utf-8-sig')
mdf[mdf.alpha==1].to_csv(OUT/'migration_primary_alpha1.csv',index=False,encoding='utf-8-sig')
assert mr.optimizer_success.all(),mr[~mr.optimizer_success]
assert mr.all_observed_defects_nonincrease.all()
assert dr[dr.source_at_supported_floor].floor_status_retained.all()
assert not mr.Stale_hops_chemically_verified.any()
assert gen[fields].notna().all().all()
assert (gen[fields].to_numpy()>=fit.low).all() and (gen[fields].to_numpy()<=fit.high).all()

imputer=SimpleImputer(strategy='median');XX=imputer.fit_transform(X)
front=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,max_features=.7,random_state=20261024,n_jobs=2).fit(XX,Y)
joblib.dump({'imputer':imputer,'model':front,'fields':fields,'targets':targets},OUT/'sensory_forward_model.joblib')
out=gen[['candidate_id','style']].copy()
prediction=front.predict(imputer.transform(gen[fields]))
out=pd.concat([out,pd.DataFrame(prediction,columns=targets)],axis=1)
out.to_csv(OUT/'generated_sensory_standardized.csv',index=False,encoding='utf-8-sig')
mp=front.predict(imputer.transform(mdf[fields]));sens=mdf[['beer','alpha']].copy()
sens=pd.concat([sens,pd.DataFrame(mp,columns=targets)],axis=1)
sens.to_csv(OUT/'migration_sensory_standardized.csv',index=False,encoding='utf-8-sig')
sensrows=[]
for style in STYLES:
    a=prediction[gen['style'].eq(style)]
    for j,t in enumerate(targets):
        sensrows.append({'style':style,'target':t,'type':meta.loc[t,'DataType'],
            'prediction_mean':float(a[:,j].mean()),'candidate_q10':float(np.quantile(a[:,j],.1)),
            'candidate_q90':float(np.quantile(a[:,j],.9)),
            'Q1_R2':float(quality.loc[t,'R2']),'Q1_MSE_skill_vs_style':float(quality.loc[t,'MSE_skill_vs_style']),
            'reliable_for_emphasis':bool(quality.loc[t,'R2']>=.1 and quality.loc[t,'MSE_skill_vs_style']>0 and t!='stale_hops')})
pd.DataFrame(sensrows).to_csv(OUT/'generated_sensory_summary_with_reliability.csv',index=False,encoding='utf-8-sig')
primary=mr[mr.alpha==1];fm=pd.DataFrame(foldrows);fc=pd.DataFrame(cvgen)
summ={'analysis_date':'2026-10-26','seed':SEED,'generated_each_style':NGEN,
      'generated_styles':STYLES,'n_generated':len(gen),'model':'native_stored_scale_standardization_and_style_LedoitWolf_Gaussian_with_range_projection',
      'holdout_mean_metrics':fm.groupby('method').mean(numeric_only=True).to_dict('index'),
      'holdout_mean_correlation':fc.groupby('method').pooled_within_style_correlation_RMSE.mean().to_dict(),
      'n_source_Pils_Lager':len(sources),'primary_alpha':1,'weights':ALPHAS,
      'primary_four_evaluable':int(primary.four_defects_evaluable_on_table1.sum()),
      'primary_four_nonincrease':int(primary.all_four_nonincrease.sum()),
      'primary_four_below_threshold':int(primary.all_four_below_threshold.sum()),
      'primary_mean_target_Mahalanobis_reduction':float(primary.target_Mahalanobis_relative_reduction.mean()),
      'primary_mean_real_Tripel_NN_reduction':float(primary.nearest_real_Tripel_relative_reduction.mean()),
      'source_at_floor_counts':dr[dr.alpha==1].groupby('defect').source_at_supported_floor.sum().astype(int).to_dict(),
      'all_optimizer_runs_successful':bool(mr.optimizer_success.all()),
      'all_observed_defects_nonincrease_all_weights':bool(mr.all_observed_defects_nonincrease.all()),
      'all_five_chemical_guarantee':False,
      'limitations':['missing_Diacetyl_sources_not_certified','floor_values_are_surrogates',
          't2n_absolute_scale_unresolved','Stale_hops_not_chemically_observed','derived_coordinate_identities_not_enforced',
          'Gaussian_cannot_exactly_reproduce_point_masses','no_external_sensory_or_brewing_validation']}
(OUT/'gaussian_baseline_summary.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summ,ensure_ascii=False,indent=2),flush=True)
