"""论文补充图表：数据探索、预测对比、生成画像与迁移细节。读取已保存结果，不重新训练模型。
AI assistance: Claude (Claude Opus 5.5), Anthropic, 2026-10-26. 用于代码实现，详见支撑材料《AI 工具使用详情说明》。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'q3_style_generation'))
sys.path.insert(0, str(ROOT / 'src'))
from make_paper_figures import FIG, RES, S1, S2, S3, MUTED, INK   # noqa: E402  共用样式

x = pd.read_csv(ROOT / '1.csv').set_index('beer')
X = x.drop(columns=['beer_id', 'tasting_category_fine'])
style = x.tasting_category_fine
Y = pd.read_csv(ROOT / '4.csv').set_index('beer').loc[x.index].drop(columns=['beer_id', 'tasting_category_fine'])
GRAY = '#c9c8c2'


def eda_overview():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={'width_ratios': [1.25, 1, 1]})
    a = axes[0]; vc = style.value_counts().sort_values()
    a.barh(vc.index, vc.values, color=S1, height=0.7)
    for i, v in enumerate(vc.values):
        a.text(v + 0.3, i, str(v), va='center', fontsize=7.5, color='#52514e')
    a.set_title('(a) 22 种风格的样本数', fontsize=10); a.tick_params(axis='y', labelsize=7.5); a.grid(axis='y', visible=False)
    b = axes[1]; na = X.isna().sum(); na = na[na > 0].sort_values()
    b.barh(na.index, na.values, color=S2, height=0.6)
    for i, v in enumerate(na.values):
        b.text(v + 0.3, i, str(v), va='center', fontsize=8, color='#52514e')
    b.set_title('(b) 含缺失值的 9 列（共 73 个）', fontsize=10); b.tick_params(axis='y', labelsize=7.5); b.grid(axis='y', visible=False)
    c = axes[2]; fl = (X == X.min()).mean() * 100
    c.hist(fl, bins=np.arange(0, 90, 5), color=S3, edgecolor='white')
    c.axvline(10, color=MUTED, ls='--', lw=0.8)
    c.text(11, c.get_ylim()[1] * 0.9, f'下限占比≥10%：{(fl >= 10).sum()} 列\n下限占比≥50%：{(fl >= 50).sum()} 列', fontsize=8, color='#52514e')
    c.set_xlabel('取值等于列最小值（检出下限）的样本占比（%）'); c.set_ylabel('列数')
    c.set_title('(c) 231 列的下限点质量分布', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / 'fig_eda_overview.png'); plt.close(fig)


def eda_logcheck():
    c = pd.read_excel(ROOT / '8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    m = X.join(c, rsuffix='_c', how='inner')
    cols = [('ethyl_acetate', 'log10'), ('X4VG', 'log10'), ('ethanol..v.v.', 'log10'), ('linalool', 'raw')]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, (col, kind) in zip(axes, cols):
        a, b = m[col], m[col + '_c']; ok = b > 0
        xv = np.log10(b[ok]) if kind == 'log10' else b[ok]
        ax.scatter(xv, a[ok], s=10, color=S1, alpha=0.7, edgecolor='none')
        lo, hi = float(min(xv.min(), a[ok].min())), float(max(xv.max(), a[ok].max()))
        ax.plot([lo, hi], [lo, hi], color=S2, lw=1.2, ls='--')
        ax.set_xlabel('log10(校准浓度)' if kind == 'log10' else '校准浓度（原值）', fontsize=8.5)
        ax.set_ylabel('附件1取值', fontsize=8.5)
        ax.set_title(f'{col}：{"log10 列" if kind == "log10" else "未取对数列"}', fontsize=9.5)
    fig.text(0.5, -0.03, '虚线为 y=x；四列中位绝对偏差均小于 1e-6。70 个共享指标中 37 列落在 log10 关系上、33 列落在原值关系上。',
             ha='center', fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / 'fig_eda_logcheck.png'); plt.close(fig)


def eda_corr():
    s5 = pd.read_csv(ROOT / '5.csv', index_col=0)
    s6 = pd.read_csv(ROOT / '6.csv', index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={'width_ratios': [1, 1.05]})
    im = a.imshow(s5.values, cmap='RdBu_r', vmin=-1, vmax=1)
    a.set_xticks(range(50)); a.set_yticks(range(50))
    short = [c.replace(' aroma', 'A').replace(' taste', 'F').replace(' - ', '-')[:18] for c in s5.columns]
    a.set_xticklabels(short, rotation=90, fontsize=4.8); a.set_yticklabels(short, fontsize=4.8); a.grid(False)
    a.set_title('(a) 50 个感官属性间 Spearman 相关（附件 5）', fontsize=10)
    keep = s6.abs().max(axis=1).sort_values(ascending=False).index[:25]
    sub = s6.loc[keep]
    im2 = b.imshow(sub.values, cmap='RdBu_r', vmin=-0.85, vmax=0.85, aspect='auto')
    b.set_yticks(range(len(sub))); b.set_yticklabels(sub.index, fontsize=6.5)
    b.set_xticks(range(50)); b.set_xticklabels(short, rotation=90, fontsize=4.8); b.grid(False)
    b.set_title('(b) 化学指标×感官交叉相关（附件 6，最大|ρ|前 25 行）', fontsize=10)
    fig.colorbar(im2, ax=b, fraction=0.03).set_label('Spearman ρ')
    fig.tight_layout(); fig.savefig(FIG / 'fig_eda_corr.png'); plt.close(fig)


def q1_pred_true():
    oof = np.load(ROOT / 'analysis/A_feasibility/floor_ablation/baseline_oof.npz', allow_pickle=True)
    ref = pd.DataFrame(oof['prediction'].mean(0), index=oof['beer'], columns=list(oof['fields'])).loc[x.index]
    tiers = pd.read_csv(RES / 'q1/q1_prediction_support_tiers.csv', encoding='utf-8-sig', index_col=0)
    cols = ['acidity', 'bitternes', 'F_hops_all', 'sweetness', 'X4vg', 'dms']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for ax, c in zip(axes.flat, cols):
        ax.scatter(Y[c], ref[c], s=12, color=S1, alpha=0.65, edgecolor='none')
        lo, hi = float(min(Y[c].min(), ref[c].min())), float(max(Y[c].max(), ref[c].max()))
        ax.plot([lo, hi], [lo, hi], color=S2, ls='--', lw=1)
        r2 = tiers.loc[c, 'R2_mean_oof']
        ax.set_title(f'{c}（{tiers.loc[c, "support_tier"]} 档，R²={r2:.2f}）', fontsize=9.5)
        ax.set_xlabel('真实标准化值', fontsize=8.5); ax.set_ylabel('折外预测（5 次重复均值）', fontsize=8.5)
    fig.text(0.5, -0.01, '虚线为 y=x。预测点向均值收缩、斜率小于 1，是回归预测的正常现象，也是 6.3 节采用折外预测分位锚点的原因。',
             ha='center', fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / 'fig_q1_pred_true.png'); plt.close(fig)


def q2_summary():
    d = pd.read_csv(RES / 'q2/q2_v31_top3_detail.csv', encoding='utf-8-sig')
    j = pd.read_csv(ROOT / 'analysis/q2/relaxed_cross_method_jaccard.csv', encoding='utf-8-sig')
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    vc = d.field.value_counts().head(10).sort_values()
    a.barh(vc.index, vc.values, color=S2, height=0.65)
    for i, v in enumerate(vc.values):
        a.text(v + 0.3, i, str(v), va='center', fontsize=8, color='#52514e')
    a.set_xlabel('出现在多少个感官维度的 Top-3 中'); a.set_title('(a) Top-3 中出现最多的 10 个化合物', fontsize=10)
    a.tick_params(axis='y', labelsize=8); a.grid(axis='y', visible=False)
    cols = ['chemical_Jaccard', 'module_Jaccard']
    k = j.groupby('top_k')[cols].median()
    if True:
        xs = np.arange(len(k)); w = 0.36
        b.bar(xs - w / 2, k[cols[0]], w, color=S1, label='化合物层面')
        b.bar(xs + w / 2, k[cols[1]], w, color=S3, label='模块层面')
        for xi, (v1, v2) in enumerate(zip(k[cols[0]], k[cols[1]])):
            b.text(xi - w / 2, v1 + 0.005, f'{v1:.3f}', ha='center', fontsize=8); b.text(xi + w / 2, v2 + 0.005, f'{v2:.3f}', ha='center', fontsize=8)
        b.set_xticks(xs); b.set_xticklabels([f'Top-{int(v)}' for v in k.index]); b.legend(frameon=False, fontsize=8.5)
    b.set_ylabel('50 维 Jaccard 中位数'); b.set_title('(b) Extra Trees 与 PLS 名单的一致性', fontsize=10); b.grid(axis='x', visible=False)
    fig.tight_layout(); fig.savefig(FIG / 'fig_q2_summary.png'); plt.close(fig)


def q3_space_and_details():
    import latent_gaussian_ot as L   # 重跑主脚本（确定性），取得共同映射 G
    fit = L.fit
    Z = fit.Z
    mu = Z.mean(0)
    U, Sv, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    P = lambda A: (A - mu) @ Vt[:2].T
    ev = Sv ** 2 / (Sv ** 2).sum()
    gen = pd.read_csv(RES / 'q3/generated_latent_231_coordinates.csv', encoding='utf-8-sig')
    mc = pd.read_csv(RES / 'q3/migration_latent_OT_231_coordinates.csv', encoding='utf-8-sig')
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4))
    pz = P(Z)
    a.scatter(pz[:, 0], pz[:, 1], s=10, color=GRAY, label='其他风格')
    for st, c in [('Tripel', S1), ('Kriek', S2), ('Lambic', S3)]:
        g = P(fit.G.transform(gen[gen['style'] == st][L.fields].to_numpy(float)))
        a.scatter(g[:, 0], g[:, 1], s=6, color=c, alpha=0.18, edgecolor='none')
        r = pz[labels_mask(st)]
        a.scatter(r[:, 0], r[:, 1], s=34, color=c, edgecolor='white', linewidth=0.8, label=f'{st}（实测●／生成·）')
    a.set_xlabel(f'PC1（{ev[0] * 100:.1f}%）'); a.set_ylabel(f'PC2（{ev[1] * 100:.1f}%）')
    a.set_title('(a) 共同正态分数空间：实测样本与 900 条生成谱', fontsize=10); a.legend(frameon=False, fontsize=8, loc='best')
    b.scatter(pz[:, 0], pz[:, 1], s=10, color=GRAY)
    rt = pz[labels_mask('Tripel')]; b.scatter(rt[:, 0], rt[:, 1], s=30, color=S1, edgecolor='white', label='真实 Tripel')
    for beer in mc.beer.unique():
        path = mc[mc.beer == beer].sort_values('alpha')
        pp = P(fit.G.transform(path[L.fields].to_numpy(float)))
        b.plot(pp[:, 0], pp[:, 1], color=S2, lw=1.1, alpha=0.8)
        b.scatter(pp[0, 0], pp[0, 1], s=30, color=S2, marker='s', edgecolor='white', zorder=3)
        b.scatter(pp[2, 0], pp[2, 1], s=36, color=S2, marker='*', edgecolor='white', zorder=3)
    b.scatter([], [], color=S2, marker='s', label='Pils/Lager 源谱'); b.scatter([], [], color=S2, marker='*', label='OT 候选（α=0.5）')
    b.set_xlabel(f'PC1（{ev[0] * 100:.1f}%）'); b.set_ylabel(f'PC2（{ev[1] * 100:.1f}%）')
    b.set_title('(b) Pils/Lager → Tripel 的 OT 迁移路径（α=0→1）', fontsize=10); b.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / 'fig_q3_space.png'); plt.close(fig)

    # 缺陷浓度比与陈化指数
    acc = pd.read_csv(RES / 'q3/migration_acceptance_all.csv', encoding='utf-8-sig')
    sel = mc[mc.alpha == 0.5].set_index('beer')
    defects = {'DMS': ('DMS', True), '双乙酰': ('X2.3.butanedione', False), '4VG': ('X4VG', True), 't2n': ('X.E..2.nonenal', False)}
    R = pd.DataFrame(index=sel.index, columns=list(defects), dtype=float)
    for name, (f, islog) in defects.items():
        s0 = X.loc[sel.index, f]; s1 = sel[f]
        R[name] = (10 ** (s1 - s0)) if islog else (s1 / s0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={'width_ratios': [1.1, 1]})
    M = R.to_numpy(float)
    im = a.imshow(M, cmap='Blues_r', vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):
            t = '缺失' if np.isnan(M[i, k]) else f'{M[i, k]:.2f}'
            a.text(k, i, t, ha='center', va='center', fontsize=8, color='white' if (not np.isnan(M[i, k]) and M[i, k] < 0.5) else INK)
    a.set_xticks(range(4)); a.set_xticklabels(list(defects)); a.set_yticks(range(len(R))); a.set_yticklabels([f'beer {b_}' for b_ in R.index], fontsize=8)
    a.grid(False); fig.colorbar(im, ax=a, fraction=0.04).set_label('候选/源 浓度比')
    a.set_title('(a) OT 候选（α=0.5）四项缺陷浓度与源谱之比（≤1 即非增）', fontsize=9.5)
    g = acc[(acc.method == 'latent_OT') & (acc.alpha == 0.5)].set_index('beer').loc[R.index]
    y = np.arange(len(g))
    for yi, (_, r) in zip(y, g.iterrows()):
        if np.isfinite(r.aging_index_source):
            b.plot([r.aging_index_source, r.aging_index_candidate], [yi, yi], color=GRAY, lw=2)
    b.scatter(g.aging_index_source, y, color=S2, s=40, label='源谱', zorder=3)
    b.scatter(g.aging_index_candidate, y, color=S1, s=40, marker='D', label='OT 候选', zorder=3)
    b.set_yticks(y); b.set_yticklabels([f'beer {b_}' for b_ in g.index], fontsize=8)
    b.set_xlabel('陈化方向指数 A(x)（越小越“新鲜”）'); b.legend(frameon=False, fontsize=8.5)
    b.invert_yaxis(); b.set_title('(b) 陈化方向指数变化（源缺项样本仅显示候选）', fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / 'fig_q3_defects.png'); plt.close(fig)


def labels_mask(st):
    return (style == st).to_numpy()


def q3_profile():
    p = pd.read_csv(RES / 'q3/generated_sensory_profile.csv', encoding='utf-8-sig')
    p = p[p.support_tier.isin(['A', 'B']) & (p.type == 'numeric')]
    order = p[p['style'] == 'Tripel'].sort_values('real_style_mean_standardized').target.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6), sharey=True)
    for ax, st, c in zip(axes, ['Tripel', 'Kriek', 'Lambic'], [S1, S2, S3]):
        g = p[p['style'] == st].set_index('target').loc[order]
        yy = np.arange(len(g))
        for yi, (_, r) in zip(yy, g.iterrows()):
            ax.plot([r.real_style_mean_standardized, r.pred_mean_standardized], [yi, yi], color=GRAY, lw=1.6)
        ax.scatter(g.real_style_mean_standardized, yy, color='#52514e', s=24, label='实测风格均值', zorder=3)
        ax.scatter(g.pred_mean_standardized, yy, color=c, s=30, marker='D', label='生成谱预测均值', zorder=3)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(st, fontsize=10.5); ax.set_xlabel('标准化感官值'); ax.legend(frameon=False, fontsize=8, loc='lower right')
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=8)
    fig.text(0.5, -0.01, '仅列 A/B 档 numeric 属性；生成谱感官值由问题一最终前向模型给出，幅度因回归收缩而偏小。', ha='center', fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / 'fig_q3_profile.png'); plt.close(fig)


if __name__ == '__main__':
    eda_overview(); eda_logcheck(); eda_corr(); q1_pred_true(); q2_summary(); q3_profile(); q3_space_and_details()
    print('done')
