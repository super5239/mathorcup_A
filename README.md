# 2026 MathorCup 大数据竞赛 · 赛道 A
啤酒酿造化合物分析与啤酒风味预测

## 目录结构
```
1.csv ~ 8.xlsx        附件原始数据（只读，不要修改）
数据说明.txt / 题目PDF
data/processed/       清洗、统一 log10 后的中间数据
src/common/           数据读取、预处理等公用代码
src/q1_sensory_imputation/  问题1：缺失条件下的感官评分（0–6 分级）
src/q2_compound_flavor/     问题2：化合物共表达图 + Top-3 关键化合物 + 稳定性
src/q3_style_generation/    问题3：风格条件生成 + 跨风格迁移
notebooks/            探索性分析
results/figures, tables  各问题输出的图表
paper/                论文及论文用图
```

## 数据核查结论
- 1.csv 共 250 个样本、22 种风格（题面写 226，以数据为准）。
- 对照附件 8「Calibrated chemical dataset」验证：37 列为 **log10(原始浓度)**（负值 ⇔ 原始浓度 < 1），
  其中包括 ethanol、pH、color、bitterness、CO2 等理化参数（与题面“保留原始量纲”不符）；
  另有 33 列（如 2,3-butanedione、linalool、trans-2-nonenal、damascenone）为**未取对数的原始值**。
- 3.xlsx 为 12 张空白评分卡 + summary，需由问题1模型填充 0–6 分。
- 4.csv（50 个感官属性）与 7.csv 为标准化/中心化值。

## 题目要点
- **问题1**：以 1.csv 为输入，给出 Aroma / Flavour 等感官维度的 0–6 分级评分，并考察缺失鲁棒性。
- **问题2**：用 2.csv 构建化合物共表达图，给出每个感官维度 Top-3 关键化合物；bootstrap 入选频率 + 方法间 Jaccard 论证稳定性，并做化学合理性论证。
- **问题3**：以 `tasting_category_fine` 为条件训练生成模型（Tripel、Kriek、Lambic 等），生成化合物谱与感官画像；Pils/Lager → Tripel 跨风格迁移，缺陷化合物（DMS / Diacetyl / 4VG / trans-2-nonenal / Stale hops）不恶化。
