"""Fixed Q2 statistical pipeline. Never executes downloaded reference code.
AI assistance: OpenAI Codex, 2026-10-25. See 方法与运行协议.md.
"""
from pathlib import Path
from time import perf_counter
import json,re,unicodedata,hashlib,argparse
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
DATA=next(ROOT.glob('2026年第七届*/2026*赛道A*'))/'附件'
parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=100)
args=parser.parse_args();N=100;R=3;SEED=20261025
assert 1<=args.limit<=N
threadpool_limits(limits=2)
x=pd.read_csv(DATA/'1.csv').set_index('beer')
y=pd.read_csv(DATA/'4.csv').set_index('beer').loc[x.index]
X=x.drop(columns=['beer_id','tasting_category_fine']);Y=y.drop(columns=['beer_id','tasting_category_fine'])
fields=list(X);targets=list(Y);A=X.to_numpy(float);YY=Y.to_numpy(float)
threshold=pd.read_excel(DATA/'8.xlsx',sheet_name='Sensory thresholds')
provided=pd.read_csv(DATA/'2.csv',index_col=0)

def repaired(s):
    s=str(s)
    if any(c in s for c in ['Î','Ã','Â']):
        try:s=s.encode('latin1').decode('utf-8')
        except (UnicodeError,LookupError):pass
    return unicodedata.normalize('NFKC',s).strip()
def normalized(s):
    s=repaired(s).lower()
    for a,b in [('α','alpha'),('β','beta'),('γ','gamma'),('δ','delta')]:s=s.replace(a,b)
    return re.sub(r'[^a-z0-9]','',s)
name_index={}
for lab in provided.index:name_index.setdefault(normalized(lab),[]).append(lab)
mapping=[]
for field in fields:
    rr=threshold[threshold.field_name_R==field]
    assert len(rr)==1,(field,len(rr))
    rr=rr.iloc[0]
    candidates=set()
    for key in ['chemical_name','chemical_name_short']:
        candidates.update(name_index.get(normalized(rr[key]),[]))
    assert len(candidates)==1,(field,candidates)
    name=next(iter(candidates))
    mapping.append({'field':field,'matrix_name':name,'reference_name':repaired(rr.chemical_name),
        'origin':rr.origin_paper,'unit':None if pd.isna(rr.unit) else rr.unit,
        'mapping_method':'reference_names_with_punctuation_and_Greek_encoding_normalization'})
assert len(set(r['matrix_name'] for r in mapping))==231
C=provided.loc[[r['matrix_name'] for r in mapping],[r['matrix_name'] for r in mapping]].to_numpy(float)
Craw=X.corr(method='spearman').to_numpy()
maxerr=float(np.nanmax(abs(C-Craw)))
assert maxerr<1e-12,maxerr
G=nx.Graph();G.add_nodes_from(range(231))
for i in range(231):
    for j in range(i+1,231):
        if np.isfinite(C[i,j]) and abs(C[i,j])>=.6:
            G.add_edge(i,j,weight=float(abs(C[i,j])),rho=float(C[i,j]))
communities=nx.community.louvain_communities(G,weight='weight',resolution=1,seed=SEED)
communities=sorted(communities,key=lambda s:(-len(s),min(fields[i] for i in s)))
module=np.empty(231,dtype=int)
for k,comm in enumerate(communities):
    for j in comm:module[j]=k
exclude={'pH':'physical_parameter','color.EBC.':'physical_parameter','bitterness':'physical_parameter',
    'CO2.PSI.':'pressure_measurement','GFS':'mixture','unfermentables':'mixture',
    'protein.g.L.':'mixture','betaglucan.mg.L.':'polymer_class','iron':'element_metric',
    'kcalperc':'derived','esters_sum':'derived','aroma_hops_sum':'derived','acids_sum':'derived','sulfur_sum':'derived'}
eligible=np.array([j for j,f in enumerate(fields) if f not in exclude],dtype=int)
for j,row in enumerate(mapping):
    row.update({'module':f'M{module[j]+1:02d}','degree':G.degree(j),
                'eligible_primary_top3':fields[j] not in exclude,'exclusion_reason':exclude.get(fields[j],'')})
pd.DataFrame(mapping).to_csv(OUT/'chemical_mapping_and_modules.csv',index=False,encoding='utf-8-sig')
edge_rows=[{'source':fields[i],'target':fields[j],'rho':v['rho'],'weight':v['weight']} for i,j,v in G.edges(data=True)]
pd.DataFrame(edge_rows).to_csv(OUT/'network_edges.csv',index=False,encoding='utf-8-sig')
graph_info={'n_nodes':231,'n_edges':G.number_of_edges(),'n_modules':len(communities),
            'n_singletons':sum(len(c)==1 for c in communities),'module_sizes':[len(c) for c in communities],
            'name_mapping_count':len(mapping),'matrix_spearman_max_error':maxerr,'rho_threshold':.6,
            'eligible_compounds':len(eligible)}
(OUT/'graph_summary.json').write_text(json.dumps(graph_info,ensure_ascii=False,indent=2),encoding='utf-8')
print('GRAPH '+json.dumps(graph_info,ensure_ascii=False),flush=True)

signature=hashlib.sha256((OUT/'方法与运行协议.md').read_bytes()).hexdigest()
checkpoint=OUT/'bootstrap_checkpoint.npz'
shape=(N,50,231)
if checkpoint.exists():
    old=np.load(checkpoint,allow_pickle=False)
    assert str(old['protocol_sha'])==signature
    imp_et=old['extra_trees'];imp_pls=old['pls'];completed=int(old['completed']);elapsed=float(old['elapsed_seconds'])
    logs=json.loads((OUT/'bootstrap_samples.json').read_text(encoding='utf-8'))
else:
    imp_et=np.full(shape,np.nan,dtype=np.float32);imp_pls=np.full(shape,np.nan,dtype=np.float32)
    completed=0;elapsed=0.;logs=[]
groups=[np.flatnonzero(x.tasting_category_fine.to_numpy()==s) for s in sorted(x.tasting_category_fine.unique())]
start=perf_counter()
for b in range(completed,args.limit):
    rng=np.random.default_rng(SEED+1000+b)
    train=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
    oob=np.setdiff1d(np.arange(len(X)),np.unique(train))
    assert len(oob)>0 and not set(train)&set(oob)
    imputer=SimpleImputer(strategy='median');at=imputer.fit_transform(A[train]);bt=imputer.transform(A[oob])
    et=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,max_features=.7,random_state=SEED+b,n_jobs=2)
    et.fit(at,YY[train]);ep=et.predict(bt);eb=np.mean(abs(ep-YY[oob]),axis=0)
    qt=QuantileTransformer(n_quantiles=100,output_distribution='normal',random_state=SEED+b)
    pat=qt.fit_transform(at);pbt=qt.transform(bt)
    pls=PLSRegression(n_components=8,scale=False,max_iter=1000)
    pls.fit(pat,YY[train]);pp=pls.predict(pbt);pb=np.mean(abs(pp-YY[oob]),axis=0)
    coef=pls.coef_
    if coef.shape==(231,50):coef=coef.T
    assert coef.shape==(50,231)
    n=len(oob)
    for first in range(0,231,12):
        cols=list(range(first,min(first+12,231)))
        batch=np.tile(bt,(len(cols)*R,1));pi=np.empty((len(cols),R,50))
        for u,j in enumerate(cols):
            for r in range(R):
                perm=rng.permutation(n);sl=slice((u*R+r)*n,(u*R+r+1)*n)
                batch[sl,j]=bt[perm,j]
                diff=pbt[perm,j]-pbt[:,j]
                predicted=pp+diff[:,None]*coef[:,j][None,:]
                pi[u,r]=np.mean(abs(predicted-YY[oob]),axis=0)-pb
                if b==0 and first==0 and u==0 and r==0:
                    check=pbt.copy();check[:,j]=pbt[perm,j]
                    assert np.max(abs(pls.predict(check)-predicted))<1e-8
        perturbed=et.predict(batch).reshape(len(cols),R,n,50)
        delta=np.mean(abs(perturbed-YY[oob][None,None,:,:]),axis=(1,2))-eb[None,:]
        imp_et[b][:,cols]=delta.T
        imp_pls[b][:,cols]=pi.mean(axis=1).T
    assert np.isfinite(imp_et[b]).all() and np.isfinite(imp_pls[b]).all()
    logs.append({'bootstrap':b,'train_beer_with_replacement':X.index[train].tolist(),
                 'oob_beer':X.index[oob].tolist(),'n_oob':n,
                 'ET_oob_macro_MAE':float(eb.mean()),'PLS_oob_macro_MAE':float(pb.mean())})
    completed=b+1
    if completed%5==0 or completed==args.limit:
        total=elapsed+perf_counter()-start
        print(f'BOOTSTRAP {completed}/100; elapsed={total:.1f}s; mean OOB count={np.mean([a["n_oob"] for a in logs]):.1f}',flush=True)
    if completed%10==0 or completed==args.limit:
        np.savez_compressed(checkpoint,extra_trees=imp_et,pls=imp_pls,completed=completed,
                            elapsed_seconds=elapsed+perf_counter()-start,protocol_sha=np.array(signature))
        (OUT/'bootstrap_samples.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding='utf-8')
if completed<100:
    print('Checkpoint saved; resume with --limit 100.',flush=True)
    raise SystemExit(0)

def top_indices(values):return eligible[np.argsort(-values[:,eligible],axis=1,kind='stable')[:,:3]]
tops_et=np.stack([top_indices(v) for v in imp_et]);tops_pls=np.stack([top_indices(v) for v in imp_pls])
mean_et=imp_et.mean(axis=0);mean_pls=imp_pls.mean(axis=0)
final_et=top_indices(mean_et);final_pls=top_indices(mean_pls)
freq=np.zeros((50,231));modulefreq=np.zeros((50,len(communities)))
for b in range(100):
    for t in range(50):
        freq[t,tops_et[b,t]]+=.01
        modulefreq[t,np.unique(module[tops_et[b,t]])]+=.01
quality=pd.read_csv(ROOT/'analysis/A_feasibility/q1_metrics/ExtraTrees_dimension_summary.csv',index_col=0).loc[targets]
observed=pd.concat([X,Y.add_prefix('target__')],axis=1).corr(method='spearman').iloc[:231,231:].to_numpy()
rows=[];target_rows=[]
for t,target in enumerate(targets):
    jaccards=[]
    for b in range(100):
        a,c=set(tops_et[b,t]),set(tops_pls[b,t]);jaccards.append(len(a&c)/len(a|c))
    a,c=set(final_et[t]),set(final_pls[t]);jaccard=len(a&c)/len(a|c)
    target_rows.append({'target':target,'Q1_R2':float(quality.loc[target,'R2']),
        'Q1_Spearman':float(quality.loc[target,'Spearman']),
        'ET_top3':' | '.join(fields[j] for j in final_et[t]),
        'PLS_top3':' | '.join(fields[j] for j in final_pls[t]),
        'Jaccard_final':jaccard,'Jaccard_bootstrap_mean':float(np.mean(jaccards)),
        'min_top3_frequency':float(min(freq[t,final_et[t]])),
        'exploratory_Q1_R2_below_0_1':bool(quality.loc[target,'R2']<.1)})
    for rank,j in enumerate(final_et[t],1):
        vals=imp_et[:,t,j];info=mapping[j]
        rows.append({'target':target,'rank':rank,'field':fields[j],'chemical_name':info['matrix_name'],
            'origin':info['origin'],'unit':info['unit'],'module':info['module'],
            'mean_OOB_permutation_MAE_increase':float(vals.mean()),
            'bootstrap_percentile_2_5':float(np.quantile(vals,.025)),
            'bootstrap_percentile_97_5':float(np.quantile(vals,.975)),
            'top3_frequency':float(freq[t,j]),'module_top3_frequency':float(modulefreq[t,module[j]]),
            'observed_spearman_descriptive':float(observed[j,t]),
            'Q1_R2':float(quality.loc[target,'R2']),
            'flag':'exploratory_weak_predictor' if quality.loc[target,'R2']<.1 else ('low_selection_stability' if freq[t,j]<.5 else 'candidate')})
pd.DataFrame(rows).to_csv(OUT/'top3_chemical_candidates.csv',index=False,encoding='utf-8-sig')
td=pd.DataFrame(target_rows);td.to_csv(OUT/'target_stability_and_jaccard.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(freq,index=targets,columns=fields).to_csv(OUT/'chemical_top3_frequencies.csv',encoding='utf-8-sig')
pd.DataFrame(modulefreq,index=targets,columns=[f'M{k+1:02d}' for k in range(len(communities))]).to_csv(OUT/'module_top3_frequencies.csv',encoding='utf-8-sig')
pd.DataFrame(mean_et,index=targets,columns=fields).to_csv(OUT/'all_factor_mean_permutation_importance.csv',encoding='utf-8-sig')
summary={'bootstrap_completed':100,'permutation_repeats':3,'graph':graph_info,
         'elapsed_seconds':elapsed+perf_counter()-start,
         'median_final_Jaccard':float(td.Jaccard_final.median()),
         'mean_final_Jaccard':float(td.Jaccard_final.mean()),
         'targets_with_all_top3_frequency_at_least_0_5':int((td.min_top3_frequency>=.5).sum()),
         'targets_with_all_top3_frequency_at_least_0_6':int((td.min_top3_frequency>=.6).sum()),
         'targets_Q1_R2_below_0_1':int(td.exploratory_Q1_R2_below_0_1.sum()),
         'protocol_sha256':signature,'note':'Statistical candidates, not causal drivers; source graph is descriptive only.'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('SUMMARY '+json.dumps(summary,ensure_ascii=False),flush=True)
