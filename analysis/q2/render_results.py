"""Render scientific figures from saved Q2 tables; no model training."""
from pathlib import Path
import argparse,json,textwrap
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--network-only',action='store_true');args=parser.parse_args()
font=Path(r'C:\Windows\Fonts\msyh.ttc')
if font.exists():plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
mapping=pd.read_csv(OUT/'chemical_mapping_and_modules.csv').set_index('field')
edges=pd.read_csv(OUT/'network_edges.csv')
g=nx.Graph();g.add_nodes_from(mapping.index)
for row in edges.itertuples():g.add_edge(row.source,row.target,weight=abs(row.rho),rho=row.rho)
active=[n for n in g if g.degree(n)>0];sub=g.subgraph(active)
counts=mapping.groupby('module').size();nontriv=counts[counts>1].sort_values(ascending=False,kind='stable')
colors=plt.get_cmap('tab20');mod_color={m:colors(i%20) for i,m in enumerate(nontriv.index)}
fig,(ax,bar)=plt.subplots(1,2,figsize=(13,7.4),gridspec_kw={'width_ratios':[2.1,1]})
pos=nx.spring_layout(sub,seed=20261025,weight='weight',iterations=100,k=.27)
positive=[(a,b) for a,b,d in sub.edges(data=True) if d['rho']>=0]
negative=[(a,b) for a,b,d in sub.edges(data=True) if d['rho']<0]
nx.draw_networkx_edges(sub,pos,edgelist=positive,ax=ax,alpha=.20,width=.7,edge_color='#65717c')
nx.draw_networkx_edges(sub,pos,edgelist=negative,ax=ax,alpha=.55,width=.9,edge_color='#bb4848',style='dashed')
nx.draw_networkx_nodes(sub,pos,ax=ax,node_color=[mod_color[mapping.loc[n,'module']] for n in sub],
    node_size=[25+9*sub.degree(n) for n in sub],edgecolors='white',linewidths=.5,alpha=.9)
ax.set_title(f'相关网络：{len(active)}个非孤立节点、{g.number_of_edges()}条边',pad=12);ax.axis('off')
bar.barh(np.arange(len(nontriv)),nontriv.values,color=[mod_color[m] for m in nontriv.index],alpha=.85)
bar.set_yticks(np.arange(len(nontriv)),nontriv.index);bar.invert_yaxis();bar.set_xlabel('模块内指标数');bar.set_title('19个非单点模块',pad=12)
for i,n in enumerate(nontriv.values):bar.text(n+.25,i,str(n),va='center',fontsize=9)
bar.set_xlim(0,float(nontriv.max())+3);bar.grid(axis='x',alpha=.15)
fig.suptitle('化合物相关网络与固定模块划分',fontsize=16,y=.98)
fig.text(.5,.018,'附件2；保留 |Spearman ρ| ≥ 0.6；Louvain，固定随机种子；字段名称见配套映射表。\n101个孤立节点未绘入左图，但保留在映射表和稳定性统计中；红虚线为负相关，相关边不表示因果或反应路径。',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.08,1,.93]);fig.savefig(OUT/'化合物网络与模块.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'化合物网络与模块.svg',bbox_inches='tight');plt.close(fig)
if args.network_only:raise SystemExit(0)

top=pd.read_csv(OUT/'top3_chemical_candidates.csv');targets=top.target.drop_duplicates().tolist()
chemical=top.pivot(index='target',columns='rank',values='top3_frequency').loc[targets].to_numpy()
modules=top.pivot(index='target',columns='rank',values='module_top3_frequency').loc[targets].to_numpy()
fields=top.pivot(index='target',columns='rank',values='field').loc[targets]
module_names=top.pivot(index='target',columns='rank',values='module').loc[targets]
quality=top.groupby('target').Q1_R2.first()
fig,axes=plt.subplots(1,2,figsize=(13,17),gridspec_kw={'width_ratios':[1.5,1]})
for idx,(ax,data,title) in enumerate(zip(axes,[chemical,modules],['化合物Top-3入选频率','对应模块命中频率'])):
    im=ax.imshow(data,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_xticks([0,1,2],['Top-1','Top-2','Top-3']);ax.xaxis.tick_top();ax.set_title(title,pad=30,fontsize=13)
    if idx==0:ax.set_yticks(range(50),[t+(' †' if quality[t]<.1 else '') for t in targets],fontsize=8)
    else:ax.set_yticks([])
    for row in range(50):
        for col in range(3):
            label=('\n'.join(textwrap.wrap(fields.iloc[row,col],24))+f'\n{data[row,col]:.0%}') if idx==0 else f'{module_names.iloc[row,col]}  {data[row,col]:.0%}'
            ax.text(col,row,label,ha='center',va='center',fontsize=6.5 if idx==0 else 8,
                    color='white' if data[row,col]>.62 else '#17222e')
    ax.set_yticks(np.arange(-.5,50,1),minor=True);ax.grid(which='minor',axis='y',color='white',linewidth=.7);ax.tick_params(which='minor',left=False)
fig.suptitle('100次分层bootstrap：化合物与模块两级稳定性',fontsize=16,y=.99)
fig.text(.5,.013,'每次袋外置换3次；主名单按平均袋外MAE增量排序。† 表示Q1平均R²<0.1，名单仅作探索。\n模块每次最多计一次命中；大模块更容易被命中；这些频率不是因果效应或显著性概率。',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.045,1,.965]);fig.savefig(OUT/'Top3与模块稳定性.png',dpi=170,bbox_inches='tight');plt.close(fig)
print('Figures saved.')
