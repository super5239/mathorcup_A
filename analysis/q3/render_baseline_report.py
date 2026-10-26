"""Scientific figures and report from saved first-version results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
DATA=next(ROOT.glob('2026年第七届*/2026*赛道A*'))/'附件'
s=json.loads((OUT/'gaussian_baseline_summary.json').read_text(encoding='utf-8'))
m=pd.read_csv(OUT/'migration_metrics_all_weights.csv');d=pd.read_csv(OUT/'migration_defect_checks.csv')
g=pd.read_csv(OUT/'generator_final_descriptive_metrics.csv');cv=pd.read_csv(OUT/'generator_holdout_metrics.csv')
cvcor=pd.read_csv(OUT/'generator_holdout_correlation.csv')
quality=pd.read_csv(ROOT/'analysis/A_feasibility/q1_metrics/ExtraTrees_dimension_summary.csv',index_col=0)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
palette=['#247e8a','#c48b35','#8063a5']

# Source/target preservation and defect control must be visible together.
group=m.groupby('alpha').mean(numeric_only=True).reset_index()
fig,axes=plt.subplots(1,2,figsize=(12.6,5.2))
axes[0].scatter(100*m.nearest_real_Tripel_relative_reduction,m.mean_defect_change_standardized,
                c=m.alpha,cmap='viridis',alpha=.24,s=23)
axes[0].plot(100*group.nearest_real_Tripel_relative_reduction,group.mean_defect_change_standardized,
             '-o',color='#247e8a',lw=1.8,ms=6)
for a in group.itertuples():
    axes[0].annotate(f'α={a.alpha:g}',(100*a.nearest_real_Tripel_relative_reduction,a.mean_defect_change_standardized),
                     xytext=(5,-13 if a.alpha in [1,10] else 6),textcoords='offset points',fontsize=9)
axes[0].axhline(0,color='#9aa1a7',lw=.8)
axes[0].set_xlabel('到真实 Tripel 最近邻距离的平均下降（%）')
axes[0].set_ylabel('已观测缺陷坐标平均变化（各列训练标准差单位）')
axes[0].set_title('风格接近与缺陷变化同时报告')
axes[1].plot(group.source_change_RMS_standardized,100*group.nearest_real_Tripel_relative_reduction,'-o',color='#8063a5')
for a in group.itertuples():
    axes[1].annotate(f'{a.alpha:g}',(a.source_change_RMS_standardized,100*a.nearest_real_Tripel_relative_reduction),
                     xytext=(6,-12 if a.alpha==10 else 5),textcoords='offset points',fontsize=9)
axes[1].set_xlabel('相对源谱改变量（231维标准化 RMS）')
axes[1].set_ylabel('到真实 Tripel 最近邻距离的平均下降（%）')
axes[1].set_title('向目标接近，需要改变多少源谱')
for ax in axes:ax.grid(alpha=.17)
fig.suptitle('Pils/Lager → Tripel：固定六档权重，主结果 α=1',fontsize=13)
fig.text(.5,.025,'实点为均值，淡点为单款。11款源谱中2款缺双乙酰，缺陷均值只含已观测项；距离下降不等于独立品评验证。',ha='center',fontsize=9,color='#59636d')
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/'迁移风格接近与缺陷权衡.png',dpi=190);fig.savefig(OUT/'迁移风格接近与缺陷权衡.svg');plt.close(fig)

# Keep the failed floor-mass reproduction prominent.
fig,axes=plt.subplots(1,2,figsize=(11.8,4.8));pos=np.arange(len(g));w=.34
axes[0].bar(pos-w/2,100*g.real_4VG_floor_fraction,width=w,label='实测记录',color='#5a7185')
axes[0].bar(pos+w/2,100*g.synthetic_4VG_floor_fraction,width=w,label='生成候选',color='#cf994a')
axes[0].set_xticks(pos,g['style']);axes[0].set_ylim(0,112);axes[0].set_ylabel('4VG 下限状态占比（%）');axes[0].legend(frameon=False)
for i in range(len(g)):
    axes[0].text(i-w/2,100*g.real_4VG_floor_fraction.iloc[i]+2,f'{g.real_4VG_floor_fraction.iloc[i]:.1%}',ha='center',fontsize=9)
    axes[0].text(i+w/2,100*g.synthetic_4VG_floor_fraction.iloc[i]+2,f'{g.synthetic_4VG_floor_fraction.iloc[i]:.1%}',ha='center',fontsize=9)
axes[0].set_title('首版明确短板：下限点质量未保真')
methods=['style_conditional','unconditional'];agg=cv.groupby('method')
for i,method in enumerate(methods):
    data=cv[cv.method==method].mean_coordinate_Wasserstein_standardized
    axes[1].scatter(np.full(len(data),i)+np.linspace(-.08,.08,len(data)),data,s=28,alpha=.6,color=['#247e8a','#9a8aab'][i])
    axes[1].plot([i-.17,i+.17],[data.median()]*2,color='#263b49',lw=2)
axes[1].set_xticks([0,1],['风格条件高斯','无条件高斯']);axes[1].set_ylabel('留出平均单变量 Wasserstein 距离')
axes[1].set_title('三折 × 三风格：保留异常值影响');axes[1].grid(axis='y',alpha=.15)
fig.text(.5,.015,'左图为全数据设计拟合的描述比较；右图为训练折内预处理的留出结果。横线为中位数。',ha='center',fontsize=9,color='#59636d')
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(OUT/'首版生成保真与下限偏差.png',dpi=190);plt.close(fig)

sens=pd.read_csv(OUT/'generated_sensory_summary_with_reliability.csv')
numeric=quality[quality.type=='numeric'].index.tolist()
matrix=sens[sens.type=='numeric'].pivot(index='style',columns='target',values='prediction_mean').loc[['Tripel','Kriek','Lambic'],numeric]
reliable=sens[sens.type=='numeric'].drop_duplicates('target').set_index('target').reliable_for_emphasis
ticklabels=[t+(' †' if not reliable[t] else '') for t in numeric]
bound=max(abs(matrix.to_numpy()).max(),.1)
fig,ax=plt.subplots(figsize=(15,4.8));im=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(0,-bound,bound))
ax.set_yticks(range(3),matrix.index);ax.set_xticks(range(len(numeric)),ticklabels,rotation=55,ha='right',fontsize=8)
for i in range(3):
    for j in range(len(numeric)):
        if not reliable[numeric[j]]:
            ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor='#dddddd',alpha=.65,edgecolor='none'))
ax.set_title('生成候选的感官画像：Q1前向模型预测，保留逐维可靠性标记',pad=12)
fig.colorbar(im,ax=ax,fraction=.025,pad=.02,label='标准化感官预测均值')
fig.text(.5,.02,'†及灰色覆盖：Q1 R²<0.1或相对风格均值MSE skill≤0。图中数值是模型输出，不是原始0–6分或品评验证。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.08,1,1));fig.savefig(OUT/'生成感官画像与可靠性.png',dpi=190);plt.close(fig)

# Scaling diagnostics retained as evidence, never used to delete test observations.
x=pd.read_csv(DATA/'1.csv').set_index('beer');X=x.drop(columns=['beer_id','tasting_category_fine'])
logs=json.loads((OUT/'generator_holdout_splits.json').read_text(encoding='utf-8'));diag=[]
for log in logs:
    ii=SimpleImputer(strategy='median');sc=StandardScaler().fit(ii.fit_transform(X.loc[log['train_beer']]))
    z=sc.transform(ii.transform(X.loc[log['test_beer']]));maxabs=abs(z).max(0)
    for j in np.argsort(-maxabs)[:5]:
        i=int(np.argmax(abs(z[:,j])));diag.append({'fold':log['fold'],'field':X.columns[j],
            'test_beer':log['test_beer'][i],'max_abs_train_scaled_holdout':float(maxabs[j]),
            'training_scale':float(sc.scale_[j]),'holdout_stored_value':float(X.loc[log['test_beer'][i],X.columns[j]])})
pd.DataFrame(diag).to_csv(OUT/'holdout_scaling_diagnostics.csv',index=False,encoding='utf-8-sig')

pc=m[m.alpha==1];cvmean=cv.groupby('method').mean(numeric_only=True)
maegain=1-cvmean.loc['style_conditional','mean_coordinate_MAE_standardized']/cvmean.loc['unconditional','mean_coordinate_MAE_standardized']
wgain=1-cvmean.loc['style_conditional','mean_coordinate_Wasserstein_standardized']/cvmean.loc['unconditional','mean_coordinate_Wasserstein_standardized']
lines=['# Q3收缩高斯与约束迁移：首版结果','',
'日期：2026-10-26。已跑通生成、留出保真评价、跨风格迁移、缺陷验收及感官画像。首版约束可运行，但生成分布仍有明显不足，不能宣称所有保真检验通过。','',
'## 已实现方法与输出','',
'231维储存坐标训练集内中位数插补、标准化，按风格拟合Ledoit–Wolf收缩高斯；样本投影到训练观测范围。Tripel、Kriek、Lambic各300个，共900个候选。另输出217个单体化学候选字段，去除Q2定义的14个理化/派生/混合指标。',
'', '它是固定参数的最小基线，没有潜变量维数选择、GAN或CVAE。完整231维包含派生指标，但未施加派生指标恒等式、化学计量或酿造工艺约束，因此交付是候选特征谱，不能称为生产配方。','',
'## 阈值与下限补充','',
'| 缺陷 | 按附件8主阈值超出/有效数 | 比例 |', '|---|---:|---:|',
'| DMS | 156/250 | 62.4% |','| Diacetyl | 172/250 | 68.8% |','| 4VG | 18/250 | 7.2% |','| t2n | 248/249 | 99.6% |','',
'这些是附件记录和标注单位下的条件性计数。双乙酰校准零值不等于真实零；t2n尺度疑点未解决，不擅自作1000倍修正。附件阈值只提供一行均值/中位数汇总，底层记录数量和介质不详，不编造为“n次啤酒试验的中位数”。',
'', '4VG下限状态168/250=67.2%，CS2最小值209/250=83.6%。迁移中9/11款源谱4VG处于下限，候选强制不高于相同替代下限，不能解释为精确浓度不增加。没有测量误差依据，不额外添加测量容差。','',
'## 风格条件生成的留出保真','',
'三折风格分层；所有预处理、观测范围和高斯参数只来自训练折。对三个目标风格分别评价，以9个“折×风格”单元等权汇总。无条件高斯作为比较，共用标准正态随机数。','',
'| 指标（越小越好） | 风格条件高斯 | 无条件高斯 |', '|---|---:|---:|',
f'| 标准化逐坐标均值误差 | {cvmean.loc["style_conditional","mean_coordinate_MAE_standardized"]:.4f} | {cvmean.loc["unconditional","mean_coordinate_MAE_standardized"]:.4f} |',
f'| 平均单变量Wasserstein | {cvmean.loc["style_conditional","mean_coordinate_Wasserstein_standardized"]:.4f} | {cvmean.loc["unconditional","mean_coordinate_Wasserstein_standardized"]:.4f} |',
f'| 去风格中心相关矩阵RMSE | {s["holdout_mean_correlation"]["style_conditional"]:.4f} | {s["holdout_mean_correlation"]["unconditional"]:.4f} |','',
f'风格条件化使前两项分别下降{maegain:.1%}和{wgain:.1%}，但相关结构没有优于无条件基线。因此只能说风格中心与边际分布有所改善，不能说联合结构已得到充分保真。小风格留出只有3—4条，相关指标本身也很不稳定。',
'', '第四项为到同风格训练样本的最近邻距离。保存了生成/真实留出各自距离及比值，不设置事后通过线。生成训练样本精确复制数为0，仅能排除精确复制，不能证明没有近邻依赖或模式收缩。',
'', '训练折尺度诊断发现个别留出值可达训练标准差的约475倍或3330倍，宏观均值会受其影响。没有删掉这些样本或按测试表现改缩放；同时保留均值、中位数、逐折表及诊断字段。',
'', '![生成保真与下限偏差](首版生成保真与下限偏差.png)','',
'| 风格 | 实测4VG下限占比 | 生成4VG下限占比 | 范围投影的坐标占比 |', '|---|---:|---:|---:|']
for a in g.itertuples():
    lines.append(f'| {a.style} | {a.real_4VG_floor_fraction:.1%} | {a.synthetic_4VG_floor_fraction:.1%} | {a.range_projection_fraction:.1%} |')
lines += ['', '高斯模型不能重现大量重复下限形成的点质量。这里的实测与生成占比是全数据拟合的描述比较，仍足以暴露问题。范围投影较多也说明矩形范围约束不足以恢复真实联合分布。','',
'## Pils/Lager → Tripel迁移','',
'在标准化坐标z中，最小化 0.5||z−z₀||²+0.5α(z−μ_T)ᵀΣ_T⁻¹(z−μ_T)，同时限制训练观测范围及四项缺陷上界。采用L-BFGS-B求解此有界凸二次问题，再在原储存坐标最终投影。α固定为0、0.1、0.3、1、3、10；主结果预先指定α=1，不按效果挑权重。',
'', f'共11款源谱×6档权重=66个结果，优化均成功。主权重下，到目标高斯的Mahalanobis距离平均下降{s["primary_mean_target_Mahalanobis_reduction"]:.1%}；到真实Tripel的最近邻标准化距离平均下降{s["primary_mean_real_Tripel_NN_reduction"]:.1%}。前一距离直接参与目标，后一距离虽不是优化项，仍只是在同一数据几何空间内的接近度，不能替代独立风格品评。','',
f'- 9/11款源谱四项字段齐全，9/9在主权重下通过四项同尺度非增；涉及下限的项目另标记为下限状态保持。',
'- beer 119、28缺少源双乙酰，2款只验证其余已观测项，四项全部通过状态为不可完整验收。',
f'- 主权重下四项同时低于主工作阈值：{s["primary_four_below_threshold"]}/9。浓度非增不等于达标；t2n阈值情景单独保存。',
'- 所有66个候选已观测缺陷都不增加；9个源4VG下限状态在全部权重下均保持。',
'- 没有任何候选被认证为五项缺陷感官不恶化；Stale hops没有异戊酸测量，现有乳酸/乙酸不新增为特异代理。','',
'![风格与缺陷权衡](迁移风格接近与缺陷权衡.png)','',
'图同时展示风格接近、缺陷变化和源谱改变量，因此不能通过把缺陷压低而忽略目标风格。它是设计权重扫描，不称为已求得全局Pareto前沿。','',
'| α | 到真实Tripel最近邻距离平均下降 | 源谱改变量RMS | 已观测缺陷坐标平均变化 |', '|---|---:|---:|---:|']
for a in group.itertuples():
    lines.append(f'| {a.alpha:g} | {a.nearest_real_Tripel_relative_reduction:.1%} | {a.source_change_RMS_standardized:.3f} | {a.mean_defect_change_standardized:.4f} |')
lines += ['', '缺陷平均变化以各列训练标准差归一化，缺失项不纳入；不是把不同mg/L量纲直接相加，也不是新的缺陷感官总分。逐项原值/替代浓度与质量标志见验收表。','',
'## 感官画像与可靠性','',
'固定Q1 Extra Trees参数在全体实测样本上拟合最终前向模型，给生成与迁移谱输出50维标准化感官预测。逐项附原Q1折外R²和相对风格均值MSE skill；候选间分位数不称为预测置信区间。','',
'![感官画像可靠性](生成感官画像与可靠性.png)','',
'酒精感虽R²高但相对风格均值skill<0，灰色标记；Stale hops也保留低可靠性标记，不能参与严格缺陷保证。这次不将新候选的标准化模型输出伪称为真实0–6评分。','',
'## 首版判断与下一步取舍','',
'首版完整跑通，可供论文报告方法、约束与权衡。保真仍存在明确问题：重复下限的点质量、训练尺度对稀有极端值敏感、相关结构未胜无条件基线、派生量恒等式未施加。不能把900条输出直接称为900条验证通过的合理化学谱。',
'', '暂不升级到CVAE、GAN或更复杂潜变量模型。若继续改进，优先考虑保留重复下限质量的边际映射及已知派生关系，属于针对数据结构的修正；需要另记协议并与首版同折比较，不回写本次预设协议。',
'', 'Q2可靠性分档、可预测性关联图和宽松Jaccard已用保存结果完成，没有重训主模型。风格混杂仍未排除，不把相关性写成因果。','',
'## 复现与文件','',
'依次运行supplement_threshold_audit.py、run_gaussian_baseline.py、render_baseline_report.py；Q2补充运行supplement_reliability.py。建模运行环境Python3.9、NumPy、Pandas、SciPy、scikit-learn、Matplotlib、joblib，具体版本另存environment_versions.json。CSV为数据与统计中间输出，原始附件未改动。']
(OUT/'Q3首版生成与迁移结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
import sys,scipy,sklearn,joblib
versions={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
          'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__,'joblib':joblib.__version__}
(OUT/'environment_versions.json').write_text(json.dumps(versions,ensure_ascii=False,indent=2),encoding='utf-8')
print('Report and three scientific figures completed.')
