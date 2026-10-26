"""论文新增图表（v3.1 结果）。读取 results/ 下已保存结果，不重新计算模型。
AI assistance: Claude (Claude Opus 5.5), Anthropic, 2026-10-26. 用于代码实现，详见支撑材料《AI 工具使用详情说明》。
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / 'results', ROOT / 'paper' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'WenQuanYi Zen Hei', 'axes.unicode_minus': False, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'savefig.dpi': 220, 'savefig.bbox': 'tight', 'figure.facecolor': 'white'})
S1, S2, S3, MUTED, INK = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b'


# ---------- 图：技术路线 ----------
def roadmap():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2)

    def box(x, y, w, h, text, fc='#eef4fc', ec=S1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9, color=INK, linespacing=1.4)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.1))

    box(0.2, 5.1, 9.6, 0.85, '数据核查与预处理：对照附件8核实70个共享指标尺度（37列log10、33列未取对数）；逐列变换字典\n'
        '训练折内插补/标准化；5次重复×3折风格分层；2.csv经数值重算确认为Spearman', fc='#f4f4f2', ec=MUTED)
    cols = [(0.2, '问题一  感官评分', S1, '#eef4fc',
             ['Extra Trees / PLS / 风格均值基线\n50维连续感官响应（折外）',
              'numeric：OOF锚点P5/P95\n相对评分0–6',
              'categorical：父项门控\n相对勾选倾向',
              '预测支持度A/B/C + MSE skill\n12张评分卡 + 250款主表']),
            (3.45, '问题二  化合物–风味', S2, '#fdf0ea',
             ['2.csv共表达网络\n|ρ|≥0.6，Louvain模块',
              '100次bootstrap袋外置换\nTop-3入选率、模块命中率',
              '解释稳定性 + 高度相关备选\nET与PLS的Jaccard',
              '敏感性：全局置换 vs 风格内置换']),
            (6.7, '问题三  生成与迁移', S3, '#e9f7f1',
             ['共同经验CDF映射G\n（保留下限点质量）',
              '风格条件收缩高斯生成\n与首版基线同折配对评价',
              '共同空间高斯OT + 缺陷投影\n冻结四条验收标准',
              '陈化方向指数；离散度\n容许域敏感性（负面结果）'])]
    for x, title, ec, fc, steps in cols:
        ax.text(x + 1.55, 4.75, title, ha='center', va='center', fontsize=11, color=INK, fontweight='bold')
        arrow(x + 1.55, 5.1, x + 1.55, 4.95)
        for k, s in enumerate(steps):
            y = 3.75 - k * 1.0
            box(x, y, 3.1, 0.78, s, fc=fc, ec=ec)
            if k < len(steps) - 1:
                arrow(x + 1.55, y, x + 1.55, y - 0.2)
    arrow(1.75, 0.75, 1.75, 0.45); arrow(1.75, 0.45, 8.0, 0.45); arrow(8.0, 0.45, 8.0, 0.75)
    ax.text(4.9, 0.25, 'Q1预测支持度用于Q2分档；最终前向模型 + 冻结锚点 → Q3生成谱感官画像', ha='center', fontsize=8, color=MUTED)
    fig.savefig(FIG / 'fig_roadmap.png'); plt.close(fig)


# ---------- 图：Q1 预测支持度与相对评分分布 ----------
def q1():
    s = pd.read_csv(RES / 'q1' / 'q1_prediction_support_tiers.csv', encoding='utf-8-sig', index_col=0).sort_values('R2_mean_oof')
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 7.2), gridspec_kw={'width_ratios': [1.5, 1]})
    y = np.arange(len(s))
    for t, c, lab in [('numeric', S1, 'numeric（强度，26项）'), ('categorical', S2, 'categorical（勾选，24项）')]:
        m = s.type == t
        a.scatter(s.R2_mean_oof[m], y[m], s=26, color=c, label=lab, zorder=3, edgecolor='white', linewidth=0.6)
    for v, lab in [(0.1, 'B/C 分界 0.1'), (0.3, 'A/B 分界 0.3')]:
        a.axvline(v, color=MUTED, ls='--', lw=0.9); a.text(v + 0.005, len(s) - 0.3, lab, fontsize=8, color=MUTED)
    neg = s.index[~s.skill_positive]
    for f in neg:
        a.annotate('未优于风格均值', (s.R2_mean_oof[f], y[list(s.index).index(f)]), xytext=(8, -3),
                   textcoords='offset points', fontsize=7.5, color='#52514e')
    a.set_yticks(y); a.set_yticklabels(s.index, fontsize=7); a.set_xlabel('5次重复折外平均 $R^2$')
    a.set_title('预测支持度：A 15 / B 21 / C 14', fontsize=11); a.legend(loc='lower right', fontsize=8, frameon=False)
    sc = pd.read_csv(RES / 'q1' / 'q1_numeric_relative_scores_250.csv', encoding='utf-8-sig', index_col=0).drop(columns='tasting_category_fine')
    dist = sc.stack().value_counts(normalize=True).sort_index()
    b.bar(dist.index, dist.values * 100, color=S1, width=0.7)
    for k, v in dist.items():
        b.text(k, v * 100 + 0.4, f'{v * 100:.1f}%', ha='center', fontsize=8, color='#52514e')
    b.set_xticks(range(7)); b.set_xlabel('相对评分等级'); b.set_ylabel('占比（%）')
    b.set_title('26项numeric相对评分分布（250款）', fontsize=11); b.grid(axis='x', visible=False)
    fig.text(0.5, -0.01, '锚点为250款5次重复OOF均值的P5/P95，冻结后统一使用；评分为基于折外预测的相对展示尺度，非原始品评分。',
             ha='center', fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / 'fig_q1_support.png'); plt.close(fig)

    # 12 张评分卡热图
    v = pd.read_csv(RES / 'q1' / 'q1_scorecards_12_values.csv', encoding='utf-8-sig', index_col=0)
    cols = ['A_malt_all', 'A_hops_all', 'A_esters_all', 'F_malt_all', 'F_hops_all', 'F_esters_all', 'body', 'bitternes',
            'co2', 'aftertaste', 'alcohol', 'sweetness', 'acidity', 'dms', 'metallic', 'X4vg', 't2n', 'diacetyl',
            'stale_hops', 'orange', 'coriander', 'clove', 'lactic', 'acetic', 'barnyard', 'overall']
    M = v[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=6, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, int(M[i, j]), ha='center', va='center', fontsize=7.5, color='white' if M[i, j] >= 4 else INK)
    tiers = s.support_tier
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f'{c}{"*" if tiers[c] == "C" else ""}' for c in cols], rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(v))); ax.set_yticklabels([f'beer {b}（{st}）' for b, st in zip(v.index, v.tasting_category_fine)], fontsize=8)
    ax.grid(False); cb = fig.colorbar(im, ax=ax, fraction=0.025); cb.set_label('相对评分')
    ax.set_title('12张评分卡的numeric相对评分（* 为预测支持度C档，仅作参考分）', fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / 'fig_q1_scorecards.png'); plt.close(fig)


# ---------- 图：Q2 全局置换 vs 风格内置换（排名） ----------
def q2():
    r = pd.read_csv(RES / 'q2' / 'q2_within_style_permutation_top3.csv', encoding='utf-8-sig')
    pairs = [('acidity', 'lactic_acid.mg.L.'), ('lactic', 'lactic_acid.mg.L.'), ('acetic', 'lactic_acid.mg.L.'),
             ('X4vg', 'X4VG'), ('F_hops_all', 'ethyl_isovalerate'), ('bitternes', 'ethyl_octanoate'),
             ('alcohol', 'ethanol..v.v.'), ('X4vg', 'lactic_acid.mg.L.'), ('alcohol', 'lactic_acid.mg.L.'),
             ('bitternes', 'lactic_acid.mg.L.')]
    d = pd.DataFrame([r[(r.target == t) & (r.field == f)].iloc[0] for t, f in pairs])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(d))[::-1]
    for yi, (_, row) in zip(y, d.iterrows()):
        ax.plot([row.global_rank_217, row.within_rank_217], [yi, yi], color='#c9c8c2', lw=2, zorder=1)
    ax.scatter(d.global_rank_217, y, s=48, color=S1, label='全局置换排名', zorder=3, edgecolor='white')
    ax.scatter(d.within_rank_217, y, s=48, color=S2, label='风格内置换排名', zorder=3, edgecolor='white', marker='D')
    ax.set_xscale('log'); ax.set_xticks([1, 3, 10, 30, 100, 217]); ax.set_xticklabels(['1', '3', '10', '30', '100', '217'])
    ax.set_yticks(y); ax.set_yticklabels([f'{t} ← {f}' for t, f in pairs], fontsize=8.5)
    ax.set_xlabel('在217个单体化学候选中的排名（对数刻度，越靠左越重要）')
    ax.axvline(3, color=MUTED, ls='--', lw=0.8); ax.text(3.1, y.max() + 0.5, 'Top-3', fontsize=8, color=MUTED)
    ax.legend(loc='center right', frameon=False, fontsize=8.5); ax.set_ylim(-0.7, y.max() + 0.9)
    ax.set_title('原Top-3候选在风格内置换下的排名变化（15个划分、同一模型与评价集）', fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / 'fig_q2_within_style.png'); plt.close(fig)


# ---------- 图：Q3 生成保真 ----------
def q3_gen():
    gm = pd.read_csv(RES / 'q3' / 'generator_holdout_metrics_paired.csv', encoding='utf-8-sig')
    gc = pd.read_csv(RES / 'q3' / 'generator_holdout_correlation_paired.csv', encoding='utf-8-sig')
    agg = gm.groupby('method')[['mean_coordinate_Wasserstein_standardized', 'floor_fraction_MAE_floor_columns',
                                'mean_coordinate_MAE_standardized']].mean()
    agg['corr'] = gc.groupby('method').pooled_within_style_correlation_RMSE.mean()
    methods = [('latent_style', '潜在高斯（本文）', S1), ('baseline_style', '首版风格高斯', S2), ('baseline_unconditional', '无条件高斯', S3)]
    metrics = [('mean_coordinate_MAE_standardized', '逐坐标均值误差'), ('mean_coordinate_Wasserstein_standardized', '单变量Wasserstein'),
               ('floor_fraction_MAE_floor_columns', '下限占比误差'), ('corr', '相关矩阵RMSE')]
    fig = plt.figure(figsize=(12, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.55, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)] + [fig.add_subplot(gs[1, 1:3])]
    for ax, (k, lab) in zip(axes[:4], metrics):
        vals = [agg.loc[m, k] for m, _, _ in methods]
        ax.bar(range(3), vals, color=[c for *_, c in methods], width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, color='#52514e')
        ax.set_xticks([]); ax.set_title(lab, fontsize=10); ax.grid(axis='x', visible=False)
        ax.set_ylim(0, max(vals) * 1.18); ax.set_xlim(-0.6, 2.6)
    axes[0].set_ylabel('留出评价（越小越好）')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in methods], loc='upper center', bbox_to_anchor=(0.5, 0.535),
               ncol=3, frameon=False, fontsize=9)
    d = pd.read_csv(RES / 'q3' / 'generator_final_descriptive.csv', encoding='utf-8-sig')
    base = {'Tripel': 0.337, 'Kriek': 0.547, 'Lambic': 0.457}   # 首版基线（analysis/q3 首版结果）
    ax = axes[4]; x = np.arange(3); w = 0.26
    ax.bar(x - w, d.real_4VG_floor_fraction * 100, w, color='#52514e', label='实测')
    ax.bar(x, d.synthetic_4VG_floor_fraction * 100, w, color=S1, label='潜在高斯')
    ax.bar(x + w, [base[s] * 100 for s in d['style']], w, color=S2, label='首版风格高斯')
    ax.set_xticks(x); ax.set_xticklabels(d['style']); ax.set_ylabel('4VG处于下限的比例（%）')
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 118)
    ax.set_title('4VG下限点质量（全数据拟合）', fontsize=10); ax.legend(frameon=False, fontsize=8, loc='upper center', ncol=3)
    ax.grid(axis='x', visible=False)
    fig.savefig(FIG / 'fig_q3_generator.png'); plt.close(fig)


# ---------- 图：Q3 迁移 ----------
def q3_mig():
    m = pd.read_csv(RES / 'q3' / 'migration_acceptance_all.csv', encoding='utf-8-sig')
    disp = pd.read_csv(RES / 'q3' / 'migration_dispersion.csv', encoding='utf-8-sig')
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for meth, lab, c, mk in [('latent_OT', '共同空间OT', S1, 'o'), ('baseline_optimizer', '首版优化基线', S2, 's')]:
        g = m[m.method == meth].groupby('alpha')[['source_change_RMS_latent', 'nn_reduction']].mean()
        a.plot(g.source_change_RMS_latent, g.nn_reduction * 100, marker=mk, color=c, lw=2, ms=7, label=lab)
        for al, row in g.iterrows():
            if al > 0:
                a.annotate(f'α={al:g}', (row.source_change_RMS_latent, row.nn_reduction * 100), xytext=(4, -11),
                           textcoords='offset points', fontsize=7.5, color='#52514e')
    a.axhline(10, color=MUTED, ls='--', lw=0.8); a.text(0.02, 11, '判据3门槛 10%', fontsize=8, color=MUTED)
    a.set_xlabel('源谱改变量（共同正态分数空间RMS）'); a.set_ylabel('到真实Tripel最近邻距离下降（%）')
    a.set_title('目标接近与干预幅度（11款平均）', fontsize=10.5); a.legend(frameon=False, fontsize=8.5, loc='lower right')
    rows = []
    for _, r in disp.iterrows():
        g = r['group']
        if g.startswith('latent_OT') or g.startswith('baseline_optimizer'):
            meth, al = g.split(' alpha=')
            rows.append((meth, float(al), r.mean_pairwise_dist))
    dd = pd.DataFrame(rows, columns=['method', 'alpha', 'pair'])
    for meth, lab, c, mk in [('latent_OT', '共同空间OT', S1, 'o'), ('baseline_optimizer', '首版优化基线', S2, 's')]:
        g = dd[dd.method == meth]
        b.plot(np.arange(len(g)), g.pair, marker=mk, color=c, lw=2, ms=7, label=lab)
        for i, (al, v) in enumerate(zip(g.alpha, g.pair)):
            b.annotate(f'{al:g}', (i, v), xytext=(0, 7 if meth == 'latent_OT' else -13), textcoords='offset points',
                       fontsize=7.5, ha='center', color='#52514e')
    real = disp.set_index('group').loc['real_Tripel(28)', 'mean_pairwise_dist']
    b.axhline(real, color='#52514e', ls='--', lw=1); b.text(0.05, real + 0.03, f'真实Tripel {real:.3f}', fontsize=8, color='#52514e')
    b.set_xticks([]); b.set_xlabel('权重/强度递增（点旁数字为α）')
    b.set_ylabel('迁移群体平均两两距离'); b.set_ylim(0, 1.35)
    b.set_title('多样性保留：是否向中心塌缩', fontsize=10.5); b.legend(frameon=False, fontsize=8.5, loc='lower left')
    fig.text(0.5, -0.02, '两法的α含义与网格不同，仅作描述性对照；离散度只说明是否塌缩，不能独立说明是否属于Tripel。',
             ha='center', fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / 'fig_q3_migration.png'); plt.close(fig)


if __name__ == '__main__':
    roadmap(); q1(); q2(); q3_gen(); q3_mig()
    print(sorted(p.name for p in FIG.glob('*.png')))
